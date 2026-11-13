"""MuJoCo task environment. No network calls; all robot controls are simulation-only."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
import re
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scene import build, ROOT, JOINTS
from chassis import WHEEL_RADIUS, TRACK


def compile_scene(xml):
    """Keep valid XML unchanged; wake only roots rejected by init-sleep checks.

    MuJoCo 3.14 rejects an init-sleep tree touching an awake tree in qpos0.
    The robot is moved to its episode pose after compilation, so this can
    occur even for fruit far from the actual start. Do not disable contacts
    or global sleeping, and never retry unrelated compiler errors.
    """
    root = None
    awakened = []
    pattern = re.compile(
        r"\d+ trees were marked as sleep='init' but only \d+ could be slept\.\s*"
        r"Body '([^']+)' \(id=\d+\) is the root of the first tree that could not be slept\."
    )
    while True:
        try:
            return mujoco.MjModel.from_xml_string(xml), xml, awakened
        except ValueError as exc:
            match = pattern.search(str(exc))
            if match is None:
                raise
            if root is None:
                root = ET.fromstring(xml)
            name = match.group(1)
            body = next((b for b in root.iter('body') if b.get('name') == name), None)
            if body is None or body.get('sleep') != 'init':
                raise
            # Every retry consumes one explicit init attribute, so it is bounded.
            body.set('sleep', 'auto')
            awakened.append(name)
            xml = ET.tostring(root, encoding='unicode')


class StrawberryEnv:
    dt=.05
    def __init__(self,seed=0,full_physics=False,render_quality="full",scene_builder=None,initial_arm_pose='scene'):
        if initial_arm_pose not in ('scene','search_stow'):
            raise ValueError('Unknown initial arm pose')
        xml,self.meta=(scene_builder or build)(seed)
        if initial_arm_pose=='search_stow':
            if not self.meta.get('tilted_stalk_prior') or self.meta.get('basket_mount')!='chassis':
                raise ValueError('Search-stow observation requires full greenhouse and chassis basket')
            base=np.asarray(self.meta['initial_base_pose'],float)
            c,s=np.cos(base[2]),np.sin(base[2])
            local=np.array([.19,-.04,.34])
            point=local.copy();point[:2]=base[:2]+np.array([[c,-s],[s,c]])@local[:2]
            self.meta.update(initial_arm_pose='search_stow',observation_pose_profile='search-stow-v1',
                             scene_initial_tcp_world=self.meta['initial_tcp_world'],
                             initial_tcp_world=point.tolist())
        if full_physics:
            tree=ET.fromstring(xml);tree.find('option/flag').set('sleep','disable')
            for body in tree.iter('body'):body.attrib.pop('sleep',None)
            xml=ET.tostring(tree,encoding='unicode')
        self.render_quality=render_quality
        self.meta.update(physics_profile='full' if full_physics else 'adaptive_sleep',render_quality=render_quality)
        self.model,self.xml,awakened=compile_scene(xml)
        self.meta.update(initial_sleep_compatibility='contact-root-wakeup-v1',
                         initial_sleep_awakened_bodies=awakened)
        self.physics_steps=round(self.dt/self.model.opt.timestep)
        if not np.isclose(self.physics_steps*self.model.opt.timestep,self.dt):
            raise ValueError('Physics timestep must divide the control period')
        self.data=mujoco.MjData(self.model)
        self.ikdata=mujoco.MjData(self.model)
        self.rng=np.random.default_rng(seed)
        self.anchors=np.array(self.meta['fruit_positions'])
        self.attached=np.ones(len(self.anchors),dtype=bool)
        self.contact_ticks=np.zeros(len(self.anchors),dtype=int)
        self.grasp_origins={}
        self.events=[]; self.frame=0; self.in_box_ticks=0
        self.renderers={}
        self.worst_obstacle_penetration=0.; self.worst_obstacle_pair=None
        self.worst_vegetation_penetration=0.
        self.opt=mujoco.MjvOption(); self.opt.geomgroup[3]=0
        self.robot_ids=[self.model.joint(n).id for n in JOINTS]
        self.qa=np.array([self.model.jnt_qposadr[j] for j in self.robot_ids])
        self.da=np.array([self.model.jnt_dofadr[j] for j in self.robot_ids])
        self.fruit_bodies=[self.model.body(f'fruit_{i}').id for i in range(len(self.anchors))]
        self.fruit_geoms=[self.model.geom(f'fruit_collision_{i}').id for i in range(len(self.anchors))]
        self.fruit_bodies=np.array(self.fruit_bodies)
        self.fruit_geom_lookup=np.full(self.model.ngeom,-1,dtype=int)
        self.fruit_geom_lookup[self.fruit_geoms]=np.arange(len(self.anchors))
        self.fruit_dofs=np.array([self.model.joint(f'fruit_joint_{i}').dofadr[0] for i in range(len(self.anchors))])
        self.stem_bodies=np.array([self.model.body(f'stem_{i}').id for i in range(len(self.anchors))])
        self.anchor_sites=np.array([self.model.site(f'stem_anchor_{i}').id for i in range(len(self.anchors))])
        self.fruit_trees=self.model.body_treeid[self.fruit_bodies]
        self.stem_trees=self.model.body_treeid[self.stem_bodies]
        self.pull_directions={}
        self.grip_body=self.model.body('gripper').id
        self.jaw_body=self.model.body('moving_jaw_so101_v1').id
        self.tcp=self.model.site('tcp').id
        self.last_action=np.array([0,-.3,.45,.25,0,1.0])
        self.data.qpos[self.qa]=self.last_action; self.data.ctrl[:6]=self.last_action
        self.base_qa=np.array([self.model.joint(n).qposadr[0] for n in ['base_x','base_y','base_yaw']])
        self.data.qpos[self.base_qa]=self.meta.get('initial_base_pose',[0.,0.,0.])
        mujoco.mj_forward(self.model,self.data)
        home,error=self.ik(np.array(self.meta.get('initial_tcp_world',[.19,-.15,.33])),gripper=1.0)
        if error>.006:raise RuntimeError('Initial TCP pose unreachable')
        self.data.qpos[self.qa]=home; self.data.ctrl[:6]=home; self.last_action=home.copy()
        mujoco.mj_forward(self.model,self.data)
        self.base_qa=np.array([self.model.joint(n).qposadr[0] for n in ['base_x','base_y','base_yaw']])
        self.base_da=np.array([self.model.joint(n).dofadr[0] for n in ['base_x','base_y','base_yaw']])
        self.drive_command=np.zeros(2)
        self.park_pose=self.data.qpos[self.base_qa].copy()
        self.target=self.select_target()

    def select_target(self):
        tcp=self.data.site_xpos[self.tcp]
        candidates=sorted((np.linalg.norm(p-tcp),i) for i,p in enumerate(self.data.xpos[self.fruit_bodies]) if self.meta['ripe'][i] and self.attached[i] and np.linalg.norm(p-self.data.body('upper_arm').xpos)<.48)
        for _,i in candidates:
            q,error=self.ik(self.fruit_pos(i),gripper=1.0)
            if error<.004: return i
        if self.meta.get('allow_no_reachable_target'):return None
        raise RuntimeError('No reachable ripe fruit for this seed')

    def pull_direction(self,i):
        direction=self.data.body('upper_arm').xpos-self.fruit_pos(i);direction[2]=0
        return direction/max(np.linalg.norm(direction),1e-8)

    def state(self): return self.data.qpos[self.qa].copy()
    def fruit_pos(self,i): return self.data.xpos[self.fruit_bodies[i]].copy()
    def ik(self,pos,gripper=1.0,start=None,approach_pitch=0.):
        m=self.model; d=self.ikdata
        d.qpos[:]=self.data.qpos
        x0=self.state()[:5] if start is None else np.asarray(start)[:5]
        low=m.actuator_ctrlrange[:5,0]+.005; high=m.actuator_ctrlrange[:5,1]-.005
        def residual(q):
            d.qpos[self.qa[:5]]=q; d.qpos[self.qa[5]]=gripper; mujoco.mj_kinematics(m,d)
            # The jaw approaches along local -Z; yaw follows the target from shoulder.
            direction=np.asarray(pos)-d.body('upper_arm').xpos
            direction[2]=0; direction/=max(np.linalg.norm(direction),1e-8)
            direction=direction*np.cos(approach_pitch)+np.array([0,0,np.sin(approach_pitch)])
            axis=-d.xmat[self.grip_body].reshape(3,3)[:,2]
            return np.r_[d.site_xpos[self.tcp]-pos,.045*(axis-direction),.001*(q[4])]
        result=least_squares(residual,np.clip(x0,low,high),bounds=(low,high),max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        residual(result.x)
        return np.r_[result.x,gripper],float(np.linalg.norm(d.site_xpos[self.tcp]-pos))

    def bilateral_contacts(self):
        pairs=self.data.contact.geom
        flags=np.zeros(len(self.anchors),dtype=np.int8)
        for side in [0,1]:
            fruit=self.fruit_geom_lookup[pairs[:,side]]
            bodies=self.model.geom_bodyid[pairs[:,1-side]]
            mask=fruit>=0
            bits=(bodies==self.grip_body).astype(np.int8)+2*(bodies==self.jaw_body).astype(np.int8)
            np.bitwise_or.at(flags,fruit[mask],bits[mask])
        return flags==3

    def contacts(self,i):return bool(self.bilateral_contacts()[i])

    def step(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(6,) or not np.isfinite(action).all(): raise ValueError('Expected 6 finite radian joint targets')
        if np.any(action<self.model.actuator_ctrlrange[:6,0]) or np.any(action>self.model.actuator_ctrlrange[:6,1]):
            raise ValueError('Joint target outside robot limits')
        # Limit each control tick to 0.04 rad (~0.8 rad/s), including gripper.
        executed=np.clip(action,self.last_action-.04,self.last_action+.04)
        self.data.ctrl[:6]=executed; self.last_action=executed.copy()
        nearby=np.linalg.norm(self.data.xpos[self.fruit_bodies,:2]-self.data.body('base').xpos[:2],axis=1)<.65
        for _ in range(self.physics_steps):
            self.data.xfrc_applied[:]=0
            self.data.qfrc_applied[:]=0
            if np.allclose(self.drive_command,0):
                delta=self.park_pose-self.data.qpos[self.base_qa]
                delta[2]=np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
                self.data.qfrc_applied[self.base_da]=np.clip(np.array([5000,5000,800])*delta-np.array([200,200,50])*self.data.qvel[self.base_da],-200,200)
            self._fruit_forces(nearby)
            old_time=self.data.time
            mujoco.mj_step(self.model,self.data)
            if self.data.time<=old_time or any(w.number for w in self.data.warning):
                raise RuntimeError('MuJoCo numerical warning/reset; episode invalid')
        if not np.isfinite(self.data.qpos).all(): raise RuntimeError('Nonfinite simulation')
        # Audit sampled robot/chassis/world contacts; fruit contacts are intentional.
        for contact in self.data.contact:
            b1=int(self.model.geom_bodyid[contact.geom1]);b2=int(self.model.geom_bodyid[contact.geom2])
            def is_arm(b):
                while b:
                    if b==self.model.body('base').id:return True
                    b=int(self.model.body_parentid[b])
                return False
            arm1,arm2=is_arm(b1),is_arm(b2)
            other=b2 if arm1 else b1
            if arm1!=arm2 and self.model.body(other).name.startswith(('leaf_','branch_','stem_')):
                self.worst_vegetation_penetration=max(self.worst_vegetation_penetration,float(-contact.dist))
                continue
            if arm1!=arm2 and other not in self.fruit_bodies and contact.dist< -self.worst_obstacle_penetration:
                self.worst_obstacle_penetration=float(-contact.dist)
                self.worst_obstacle_pair=[self.model.body(b1).name+':'+(self.model.geom(contact.geom1).name or str(contact.geom1)),self.model.body(b2).name+':'+(self.model.geom(contact.geom2).name or str(contact.geom2))]
        if self.target is None:
            self.in_box_ticks=0;self.frame+=1
            return executed
        p=self.fruit_pos(self.target); basket=self.basket_position()
        relative=self.data.body('basket').xmat.reshape(3,3).T@(p-basket)
        jid=self.model.joint(f'fruit_joint_{self.target}').id; da=self.model.jnt_dofadr[jid]
        half=self.meta.get('basket_acceptance_half_xy_m',[.066,.048])
        inside=abs(relative[0])<half[0] and abs(relative[1])<half[1] and .012<relative[2]<.05
        settled=np.linalg.norm(self.data.qvel[da:da+3])<.05
        released=not self.contacts(self.target)
        self.in_box_ticks=self.in_box_ticks+1 if inside and settled and released and not self.attached[self.target] else 0
        self.frame+=1
        return executed

    def base_pose(self): return self.data.qpos[self.base_qa].copy()

    def basket_position(self): return self.data.body('basket').xpos.copy()

    def drive(self,linear=0.,angular=0.):
        command=np.array([linear,angular],dtype=float)
        if not np.isfinite(command).all() or abs(linear)>.15 or abs(angular)>.5:
            raise ValueError('Base speed limits: 0.15 m/s and 0.5 rad/s')
        if np.allclose(command,0) and not np.allclose(self.drive_command,0):
            self.park_pose=self.base_pose()
        self.drive_command=command
        left=-(linear-angular*TRACK/2)/WHEEL_RADIUS
        right=-(linear+angular*TRACK/2)/WHEEL_RADIUS
        self.data.ctrl[6:]=[left,left,right,right]

    @property
    def success(self): return self.target is not None and self.in_box_ticks>=10 and bool(self.meta['ripe'][self.target])

    def _camera_renderer(self,camera,width,height):
        size=(width,height)
        if size not in self.renderers:
            self.renderers[size]=mujoco.Renderer(self.model,height,width)
        renderer=self.renderers[size]
        self.opt.geomgroup[4]=self.render_quality!='fast'
        renderer.update_scene(self.data,camera=camera,scene_option=self.opt)
        for flag in [mujoco.mjtRndFlag.mjRND_SHADOW,mujoco.mjtRndFlag.mjRND_REFLECTION]:
            renderer.scene.flags[flag]=self.render_quality!='fast'
        return renderer

    def _fruit_forces(self, nearby):
        # Vectorized tether forces for all fruit, including the former background fruit.
        awake=(self.data.tree_asleep[self.fruit_trees]<0)|(self.data.tree_asleep[self.stem_trees]<0)
        awake |= nearby
        active=np.flatnonzero(self.attached & awake);bids=self.fruit_bodies[active]
        anchors=self.data.site_xpos[self.anchor_sites]
        displacement=self.data.xpos[self.fruit_bodies]-anchors
        velocity=self.data.qvel[self.fruit_dofs[:,None]+np.arange(3)]
        spring=-65*displacement-.9*velocity
        self.data.xfrc_applied[bids,:3]=spring[active]+np.c_[np.zeros((len(active),2)),self.model.body_mass[bids]*9.81]
        # cvel angular component is world-aligned; verified against mj_objectVelocity.
        self.data.xfrc_applied[bids,3:]=-.0001*self.data.cvel[bids,:3]
        stems=self.stem_bodies[active];reaction=-spring[active]
        self.data.xfrc_applied[stems,:3]=reaction
        self.data.xfrc_applied[stems,3:]=np.cross(anchors[active]-self.data.xipos[stems],reaction)
        bilateral=self.bilateral_contacts() & self.attached
        self.contact_ticks=np.where(bilateral,self.contact_ticks+1,0)
        for i in list(self.grasp_origins):
            if not bilateral[i]:self.grasp_origins.pop(i,None);self.pull_directions.pop(i,None)
        for i in np.flatnonzero(bilateral):
            if self.contact_ticks[i]>=20 and i not in self.grasp_origins:
                self.grasp_origins[i]=self.data.site_xpos[self.tcp].copy()
                direction=self.data.body('upper_arm').xpos-anchors[i];direction[2]=0
                self.pull_directions[i]=direction/max(np.linalg.norm(direction),1e-8)
            if i not in self.grasp_origins:continue
            tcp_pull=self.data.site_xpos[self.tcp]-self.grasp_origins[i]
            direction=self.pull_directions[i]
            pulled=np.linalg.norm(displacement[i])>.012 and displacement[i]@direction>.007 and tcp_pull@direction>.015
            if pulled:
                self.attached[i]=False;self.data.xfrc_applied[self.fruit_bodies[i],:]=0
                self.data.xfrc_applied[self.stem_bodies[i],:]=0
                self.events.append({'event':'detached','fruit':int(i),'time':float(self.data.time),'displacement':displacement[i].tolist(),'bilateral_contact':True})

    def render(self,camera='top',width=640,height=480):
        return self._camera_renderer(camera,width,height).render().copy()

    def close(self):
        for renderer in self.renderers.values(): renderer.close()
        self.renderers.clear()

    def depth_view(self, camera='top', width=480, height=360):
        """Metric optical-axis depth and camera calibration; no object identities."""
        renderer = self._camera_renderer(camera,width,height)
        renderer.enable_depth_rendering()
        try: depth = renderer.render().copy()
        finally: renderer.disable_depth_rendering()
        cid = self.model.camera(camera).id
        return depth, {'position': self.data.cam_xpos[cid].copy(),
                       'rotation': self.data.cam_xmat[cid].reshape(3,3).copy(),
                       'fovy': float(self.model.cam_fovy[cid])}

if __name__=='__main__':
    e=StrawberryEnv()
    print('target',e.target,'state',e.state())
    for i,p in enumerate(e.anchors): print(i,e.ik(p)[1])
    for _ in range(40): e.step(e.last_action)
    print('drift',np.max(np.linalg.norm(np.array([e.fruit_pos(i) for i in range(len(self.anchors))])-e.anchors,axis=1)))
    from PIL import Image
    for name in ['overview','detail','top','wrist']: Image.fromarray(e.render(name,960,720)).save(ROOT/'reports'/f'{name}.png')
    e.close()
