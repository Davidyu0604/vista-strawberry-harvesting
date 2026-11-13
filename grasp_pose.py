"""Experimental robot-only jaw calibration and visual stalk pose IK.

No scene object IDs, object positions, contacts, or maturity are planner inputs.
The jaw aperture is the collision hull's section at the calibrated pad depth,
not a midpoint of body origins. This module does not alter collision geometry.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull


def line_interval(equations, y, z):
    """Intersect a convex solid with an X-directed line; None if uncovered."""
    normal = equations[:, 0]
    offset = equations[:, 1] * y + equations[:, 2] * z + equations[:, 3]
    parallel = abs(normal) < 1e-9
    if np.any(offset[parallel] > 1e-8):
        return None
    lower = np.max(-offset[normal < -1e-9] / normal[normal < -1e-9], initial=-np.inf)
    upper = np.min(-offset[normal > 1e-9] / normal[normal > 1e-9], initial=np.inf)
    return None if lower > upper else (float(lower), float(upper))


class JawCalibration:
    def __init__(self, env):
        self.env = env
        m = env.model
        self.data = mujoco.MjData(m)
        self.depth = float(m.site_pos[env.tcp, 2])
        pad = m.geom('peduncle_rubber_insert').id
        if not np.allclose(m.geom_quat[pad], [1, 0, 0, 0]):
            raise ValueError('Calibration requires the known axis-aligned pad')
        self.fixed_face = float(m.geom_pos[pad, 0] + m.geom_size[pad, 0])
        self.meshes = []
        for g in range(m.ngeom):
            if (m.geom_bodyid[g] != env.jaw_body or
                    not (m.geom_contype[g] or m.geom_conaffinity[g])):
                continue
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError('Unexpected moving jaw collision primitive')
            mesh = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            self.meshes.append((g, ConvexHull(m.mesh_vert[a:a+n]).equations))
        if not self.meshes:
            raise ValueError('No moving jaw collision surface')

    def aperture(self, opening, y=0.):
        e, d = self.env, self.data
        d.qpos[:] = e.data.qpos
        d.qpos[e.qa[-1]] = opening
        mujoco.mj_kinematics(e.model, d)
        grip_r = d.xmat[e.grip_body].reshape(3, 3)
        hits = []
        for g, equations in self.meshes:
            # Transform cached hull planes into the gripper body frame.
            rot = grip_r.T @ d.geom_xmat[g].reshape(3, 3)
            pos = grip_r.T @ (d.geom_xpos[g] - d.xpos[e.grip_body])
            normals = equations[:, :3] @ rot.T
            planes = np.c_[normals, equations[:, 3] - normals @ pos]
            interval = line_interval(planes, y, self.depth)
            if interval is not None:
                hits.append(interval[0])
        if not hits:
            raise ValueError('Moving finger does not cover the calibrated grasp depth')
        return min(hits) - self.fixed_face

    def center(self, opening):
        gap = self.aperture(opening)
        if gap <= 0:
            raise ValueError('No positive calibrated jaw gap')
        return np.array([self.fixed_face + gap / 2, 0., self.depth])


class RobotSelfCollision:
    """Robot collision geometry only; no fruit/plant geometry enters planning."""
    def __init__(self,env):
        self.e=env;self.data=mujoco.MjData(env.model)
        m=env.model;root=m.body_rootid[env.grip_body]
        geoms=[g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]]==root
               and (m.geom_contype[g] or m.geom_conaffinity[g])]
        excluded=set(map(int,m.exclude_signature));self.pairs=[]
        for i,g in enumerate(geoms):
            for h in geoms[i+1:]:
                a,b=sorted((int(m.geom_bodyid[g]),int(m.geom_bodyid[h])))
                wa,wb=int(m.body_weldid[a]),int(m.body_weldid[b])
                if wa==wb:continue
                if m.body_weldid[m.body_parentid[wa]]==wb or m.body_weldid[m.body_parentid[wb]]==wa:continue
                if (a<<16)+b in excluded:continue
                if not ((m.geom_contype[g]&m.geom_conaffinity[h]) or (m.geom_contype[h]&m.geom_conaffinity[g])):continue
                self.pairs.append((g,h))

    def collision(self,q):
        e=self.e;m=e.model;d=self.data
        d.qpos[:]=e.data.qpos;d.qpos[e.qa]=q;mujoco.mj_kinematics(m,d)
        for g,h in self.pairs:
            if np.linalg.norm(d.geom_xpos[g]-d.geom_xpos[h])>m.geom_rbound[g]+m.geom_rbound[h]+.001:continue
            distance=float(mujoco.mj_geomDistance(m,d,g,h,.001,None))
            if distance<-.0001:
                return {'bodies':[m.body(m.geom_bodyid[k]).name for k in (g,h)],
                        'robot_geom_indices':[g,h],'penetration_m':-distance}
        return None

    def path_collision(self,start,end):
        # Kinematic preflight, not a guarantee of collision-free dynamics.
        count=max(1,int(np.ceil(np.max(np.abs(np.asarray(end)-start))/.02)))
        for fraction in np.linspace(0,1,count+1):
            hit=self.collision(np.asarray(start)+fraction*(np.asarray(end)-start))
            if hit:return {**hit,'path_fraction':float(fraction)}
        return None


def solve_stalk_pose(env, point, axis, start, opening, local_center, direction=None):
    """Place a calibrated robot point on the visual target and align pad width."""
    point, axis = np.asarray(point, float), np.asarray(axis, float)
    if point.shape != (3,) or axis.shape != (3,) or not np.all(np.isfinite(np.r_[point, axis])):
        raise ValueError('Finite visual point and stalk axis required')
    if np.linalg.norm(axis) < 1e-8:
        raise ValueError('Empty visual stalk axis')
    axis = axis / np.linalg.norm(axis)
    data = env.ikdata
    data.qpos[:] = env.data.qpos
    low, high = env.model.actuator_ctrlrange[:5].T

    def residual(q):
        data.qpos[env.qa] = np.r_[q, opening]
        mujoco.mj_kinematics(env.model, data)
        approach = (point - data.body('upper_arm').xpos if direction is None
                    else np.array(direction, dtype=float, copy=True))
        approach[2] = 0
        if np.linalg.norm(approach) < 1e-8:
            raise ValueError('Undefined approach direction')
        approach /= np.linalg.norm(approach)
        width = axis - np.dot(axis, approach) * approach
        if np.linalg.norm(width) < .2:
            raise ValueError('Stalk too parallel to approach')
        width /= np.linalg.norm(width)
        rot = data.xmat[env.grip_body].reshape(3, 3)
        center = data.xpos[env.grip_body] + rot @ local_center
        return np.r_[center - point, .045 * (-rot[:, 2] - approach),
                     .04 * (rot[:, 1] - width), .0001 * (q - start[:5])]

    solution = least_squares(residual, np.clip(start[:5], low + .005, high - .005),
                             bounds=(low + .005, high - .005), max_nfev=150,
                             ftol=1e-9, xtol=1e-9, gtol=1e-9)
    error = float(np.linalg.norm(residual(solution.x)[:3]))
    return np.r_[solution.x, opening], error


def fixed_roll_ik(env, pos, gripper, roll, approach_pitch=0., start=None):
    """Robot-only position/approach IK with wrist roll held at grasp command."""
    data=env.ikdata;data.qpos[:]=env.data.qpos
    start=env.state() if start is None else np.asarray(start)
    low,high=env.model.actuator_ctrlrange[:4].T
    def residual(q):
        data.qpos[env.qa]=np.r_[q,roll,gripper]
        mujoco.mj_kinematics(env.model,data)
        direction=np.asarray(pos)-data.body('upper_arm').xpos
        direction[2]=0;direction/=max(np.linalg.norm(direction),1e-8)
        direction=direction*np.cos(approach_pitch)+[0,0,np.sin(approach_pitch)]
        approach=-data.xmat[env.grip_body].reshape(3,3)[:,2]
        return np.r_[data.site_xpos[env.tcp]-pos,.045*(approach-direction),.0001*(q-start[:4])]
    sol=least_squares(residual,np.clip(start[:4],low+.005,high-.005),
                      bounds=(low+.005,high-.005),max_nfev=150,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    error=float(np.linalg.norm(residual(sol.x)[:3]))
    return np.r_[sol.x,roll,gripper],error

