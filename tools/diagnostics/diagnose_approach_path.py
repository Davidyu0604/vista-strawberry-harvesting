"""Full-scene offline path/hand-pose comparison; JSON only, no images or video.

Planning uses the archived visual target/axis and robot kinematics. Fruit IDs and
positions are used ONLY in the diagnostic measurements, never to choose a path.
One process at a time: scene construction writes shared texture assets.
"""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from replay_source import replay_environment
from grasp_pose import JawCalibration, solve_stalk_pose
from scipy.optimize import brentq


def fk(env,q,data):
    data.qpos[:]=env.data.qpos;data.qpos[env.qa]=q
    mujoco.mj_kinematics(env.model,data)
    return data.site_xpos[env.tcp].copy()


def aligned_ik(env,point,axis,start,direction=None):
    """Align finger width (local Y) with visually fitted stalk, not its true pose."""
    data=env.ikdata;data.qpos[:]=env.data.qpos
    low,high=env.model.actuator_ctrlrange[:5].T
    def residual(q):
        data.qpos[env.qa]=np.r_[q,1.];mujoco.mj_kinematics(env.model,data)
        approach=(np.asarray(point)-data.body('upper_arm').xpos if direction is None else np.array(direction,copy=True));approach[2]=0
        approach/=np.linalg.norm(approach)
        width=axis-np.dot(axis,approach)*approach;width/=np.linalg.norm(width)
        rotation=data.xmat[env.grip_body].reshape(3,3)
        return np.r_[data.site_xpos[env.tcp]-point,.045*(-rotation[:,2]-approach),
                     .04*(rotation[:,1]-width),.0001*(q-start[:5])]
    solution=least_squares(residual,np.clip(start[:5],low+.005,high-.005),bounds=(low+.005,high-.005),
                           max_nfev=150,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    residual(solution.x)
    return np.r_[solution.x,1.],float(np.linalg.norm(data.site_xpos[env.tcp]-point))


def run(episode,variant,out,close_seconds=2.5,continuation=None):
    if out.exists():raise FileExistsError(f'Refusing to overwrite diagnostic result: {out}')
    config=json.loads((episode/'run-config.json').read_text());report=json.loads((episode/'report.json').read_text())
    rows=[json.loads(l) for l in (episode/'trajectory.jsonl').read_text().splitlines()]
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    pre=next(r['frame'] for r in rows if r['phase']=='visual_pregrasp')
    grasp=next(r['frame'] for r in rows if r['phase']=='visual_grasp')
    locks=[e['result'] for e in events if e['kind']=='receipt' and isinstance(e.get('result'),dict)
           and e['result'].get('target_confirmed') and e['result'].get('locked_target_m')
           and int(e['result']['frame_id'][1:])<=pre]
    lock=locks[-1];target=np.asarray(lock['locked_target_m']);axis=np.asarray(lock['stalk_fit']['axis_world'])
    start_frame=pre if variant.startswith('roll-') else grasp
    env,source=replay_environment(episode,config,report)
    centered=variant=='roll-centered'
    calibration=JawCalibration(env)
    opening=.2 if centered else 1.
    contact_opening=brentq(lambda q:calibration.aperture(q)-.003,-.15,.2) if centered else None
    samples=[];max_replay_error=0.;max_step=0.;max_joint_step=0.;max_requested_joint_step=0.;blocked=0
    try:
        for row in rows[:start_frame]:
            error=max(float(np.max(np.abs(env.state()-row['state']))),float(np.max(np.abs(env.base_pose()-row['base_pose']))),
                      float(np.linalg.norm(env.data.site_xpos[env.tcp]-row['tcp_position'])))
            max_replay_error=max(max_replay_error,error)
            if error>1e-6:raise RuntimeError('Recorded prefix differs')
            env.drive(*row['base_command']);env.step(row.get('requested_action',row['action']))
        reference=mujoco.MjData(env.model);reference.qpos[:]=env.data.qpos
        frozen_qpos=env.data.qpos.copy()
        # Offline association only. Never passed to a motion generator.
        distance=np.linalg.norm(env.data.site_xpos[env.grasp_sites]-target,axis=1)
        fruit=int(np.argmin(distance))
        if distance[fruit]>.015:raise ValueError('No unambiguous nearby stalk for offline audit')
        initial_site=env.data.site_xpos[env.grasp_sites[fruit]].copy()
        fruit_geom=int(env.fruit_geoms[fruit]);stalk_geom=int(env.stalk_geoms[fruit])
        fingers=[g for g in range(env.model.ngeom) if env.model.geom_bodyid[g] in (env.grip_body,env.jaw_body)
                 and (env.model.geom_contype[g] or env.model.geom_conaffinity[g])]
        scratch=mujoco.MjData(env.model)
        def measure(phase):
            gr=env.data.xmat[env.grip_body].reshape(3,3)
            r=env.data.xmat[env.fruit_bodies[fruit]].reshape(3,3)
            bottom=env.fruit_pos(fruit)+r@np.array([0,0,.022*env.meta['fruit_scales'][fruit]])
            local=r.T@(target-bottom)
            # Truth below is exclusively offline evaluation. Use a finite stalk
            # segment, not an infinite line that could intersect outside the stem.
            stalk_axis=gr.T@r[:,2]
            stalk_bottom=gr.T@(bottom-env.data.xpos[env.grip_body])
            try:
                gap=calibration.aperture(env.state()[5]);center=calibration.center(env.state()[5])
            except ValueError:
                gap=None;center=np.array([.001,0,calibration.depth])
            along=float(np.clip((center-stalk_bottom)@stalk_axis,0,.036))
            closest=stalk_bottom+along*stalk_axis
            contacts=[];touch={'fixed':False,'moving':False}
            for c in env.data.contact:
                if c.dist>0:continue
                gs=list(map(int,c.geom));bs=[int(env.model.geom_bodyid[g]) for g in gs]
                if not any(g in fingers for g in gs):continue
                if all(env.model.body_rootid[b]==env.model.body_rootid[env.grip_body] for b in bs):continue
                contacts.append([env.model.geom(g).name or f'geom_{g}' for g in gs])
                if stalk_geom in gs:
                    touch['fixed'] |= env.grip_body in bs;touch['moving'] |= env.jaw_body in bs
            samples.append({'frame':env.frame,'phase':phase,'tcp_m':env.data.site_xpos[env.tcp].tolist(),
                'tcp_error_mm':float(np.linalg.norm(env.data.site_xpos[env.tcp]-target)*1000),
                'stalk_displacement_mm':float(np.linalg.norm(env.data.site_xpos[env.grasp_sites[fruit]]-initial_site)*1000),
                'target_radial_error_mm':float(np.linalg.norm(local[:2])*1000),'target_height_mm':float(local[2]*1000),
                'closing_axis_dot_visual_stalk':float(gr[:,0]@axis),'target_stalk_contacts':touch,'contacts':contacts})
            samples[-1]['jaw_center_audit']={'gap_mm':None if gap is None else gap*1000,
                'center_gripper_mm':(center*1000).tolist(),
                'center_valid':gap is not None and gap>0,
                'closest_stalk_point_gripper_mm':(closest*1000).tolist(),
                'stalk_minus_center_mm':((closest-center)*1000).tolist(),
                'true_axis_gripper':stalk_axis.tolist(),'actual_opening_rad':float(env.state()[5])}
        def tick(q,phase,protect=True):
            nonlocal max_step,max_joint_step,max_requested_joint_step,blocked
            old=env.last_action.copy();expected=fk(env,q,scratch)
            step=float(np.linalg.norm(expected-fk(env,old,scratch)))
            max_step=max(max_step,step)
            max_requested_joint_step=max(max_requested_joint_step,float(np.max(np.abs(q-old))))
            env.drive(0,0);executed=env.step(q)
            max_joint_step=max(max_joint_step,float(np.max(np.abs(executed-old))))
            measure(phase)
            error=float(np.linalg.norm(env.data.site_xpos[env.tcp]-expected))
            blocked=blocked+1 if error>.008 else 0
            if protect and blocked>=3:raise ValueError('Tracking error persisted for 3 ticks')
        def goto(q,phase):
            for _ in range(180):
                old=env.last_action.copy();delta=q-old
                if np.max(np.abs(delta))<1e-8:return
                scale=min(1.,.04/max(np.max(np.abs(delta)),1e-10))
                origin=fk(env,old,scratch)
                while np.linalg.norm(fk(env,old+scale*delta,scratch)-origin)>.005:scale*=.8
                tick(old+scale*delta,phase)
            raise ValueError('Motion timeout')
        final_recorded=next(r for r in reversed(rows) if r['phase']=='visual_grasp')['requested_action']
        direction=None
        solve=lambda point,start:(solve_stalk_pose(env,point,axis,start,opening,calibration.center(opening),direction)
                                  if centered else aligned_ik(env,point,axis,start,direction)
                                  if variant.startswith('roll-') else env.ik(point,1,start=start,approach_pitch=0))
        final_q,ik_error=solve(target,env.state())
        if variant in ('recorded','joint-chord'):final_q=np.asarray(final_recorded)
        # Freeze fruit at pre-approach pose in scratch data. Diagnose the planned
        # final pose independently of any deflection caused by the live fingers.
        reference.qpos[:]=frozen_qpos;reference.qpos[env.qa]=final_q;mujoco.mj_kinematics(env.model,reference)
        gaps={env.model.geom(g).name or f'geom_{g}':float(mujoco.mj_geomDistance(env.model,reference,g,fruit_geom,.1,None)*1000) for g in fingers}
        endpoint={'finger_to_undisturbed_fruit_signed_gap_mm':gaps,'ik_error_mm':float(ik_error*1000),
                  'tcp_m':reference.site_xpos[env.tcp].tolist(),
                  'closing_axis_dot_visual_stalk':float(reference.xmat[env.grip_body].reshape(3,3)[:,0]@axis),
                  'joint_target':final_q.tolist()}
        if variant in ('roll-axial','roll-centered'):
            direction=-reference.xmat[env.grip_body].reshape(3,3)[:,2].copy()
            direction[2]=0;direction/=np.linalg.norm(direction)
        measure('start');failure=None
        try:
            if variant=='recorded':
                end=max(r['frame'] for r in rows if r['phase']=='gripper')
                for row in rows[start_frame:end+1]:
                    error=max(float(np.max(np.abs(env.state()-row['state']))),float(np.max(np.abs(env.base_pose()-row['base_pose']))),
                              float(np.linalg.norm(env.data.site_xpos[env.tcp]-row['tcp_position'])))
                    max_replay_error=max(max_replay_error,error)
                    if error>1e-6:raise RuntimeError('Recorded continuation differs')
                    tick(np.asarray(row.get('requested_action',row['action'])),row['phase'],False)
            else:
                if variant.startswith('roll-'):
                    q,error=solve(target-(.055*direction if direction is not None else [.055,0,0]),env.state())
                    if error>.006:raise ValueError('Aligned pregrasp unreachable')
                    goto(q,'pregrasp_rotate')
                    for _ in range(55):tick(q,'pregrasp_settle')
                if variant=='joint-chord':goto(final_q,'grasp')
                else:
                    origin=(env.data.xpos[env.grip_body]+env.data.xmat[env.grip_body].reshape(3,3)@calibration.center(opening)
                            if centered else env.data.site_xpos[env.tcp].copy());q=env.last_action.copy()
                    for f in np.linspace(0,1,29)[1:]:
                        q,error=solve(origin+f*(target-origin),q)
                        if error>.006:raise ValueError('Cartesian waypoint unreachable')
                        goto(q,'grasp')
                    final_q=q
                for _ in range(55):tick(final_q,'grasp_settle')
                for _ in range(round(close_seconds/env.dt)):
                    q=env.last_action.copy();q[5]=max(-.15,q[5]-.01 if centered else q[5]-.04)
                    if centered:
                        q,error=solve_stalk_pose(env,target,axis,q,q[5],calibration.center(max(contact_opening,q[5])),direction)
                        if error>.006:raise ValueError('Centered closing waypoint unreachable')
                        if np.max(np.abs(q-env.last_action))>.04+1e-9 or np.linalg.norm(fk(env,q,scratch)-fk(env,env.last_action,scratch))>.005:
                            raise ValueError('Centered closing exceeds motion limits')
                        tick(q,'gripper')
                    else:tick(q,'gripper')
            for _ in range(10):tick(env.last_action.copy(),'contact_hold')
        except ValueError as exc:failure=str(exc)
        hold=[s for s in samples if s['phase']=='contact_hold']
        # Include the initial close sample plus ten 50 ms intervals.
        stable=len(hold)==10 and len(samples)>=11 and all(all(s['target_stalk_contacts'].values()) for s in samples[-11:])
        result={'episode':str(episode),'variant':variant,'physics_source':source,'prefix_start_frame':start_frame,
                'max_recorded_state_error':max_replay_error,'visual_target_m':target.tolist(),'visual_stalk_axis':axis.tolist(),
                'offline_associated_fruit':fruit,'model_calls':0,'images_written':0,'videos_written':0,
                'close_seconds':close_seconds,'hold_seconds':.5,'hold_samples_include_initial':True,
                'grasp_reference':'calibrated jaw gap center' if centered else 'legacy TCP',
                'approach_opening_rad':opening,'center_compensation_stop_opening_rad':contact_opening,
                'planned_approach_direction_world':None if direction is None else direction.tolist(),
                'endpoint':endpoint,'max_requested_tcp_step_mm':max_step*1000,'max_joint_step_rad':max_joint_step,
                'max_requested_joint_step_rad':max_requested_joint_step,
                'stable_target_stalk_hold':stable,'failure':failure,'final':samples[-1],'samples':samples,
                'detached_fruits_offline':np.flatnonzero(~env.attached).tolist(),
                'note':'Offline full-scene counterfactual, no online success claim. Fruit truth is audit-only. Target height, pads and collisions unchanged.'}
        # Continuations receive the simulator, not the audit's associated fruit
        # or truth-based stability result. Their own robot/tactile gates decide.
        if continuation is not None:
            result['continuation']=continuation(env) if failure is None else {'error':'Approach failed; continuation skipped'}
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('variant','max_recorded_state_error','stable_target_stalk_hold','failure','final')}),flush=True)
        if continuation is not None:print(json.dumps({k:v for k,v in result['continuation'].items() if k!='trace'}),flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path)
    p.add_argument('--variant',choices=['recorded','joint-chord','cartesian','roll-aligned','roll-axial','roll-centered'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--close-seconds',type=float,choices=[2.5,3.],default=2.5);a=p.parse_args()
    if a.variant=='recorded' and a.close_seconds!=2.5:p.error('Recorded controls retain original close duration')
    run(a.episode.resolve(),a.variant,a.out,a.close_seconds)
