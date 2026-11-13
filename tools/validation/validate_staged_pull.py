"""Full-scene, same-state pull comparisons. Truth is recorded only by the audit."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse
import copy
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from replay_source import replay_environment


def validate(episode,out):
    config=json.loads((episode/'run-config.json').read_text())
    report=json.loads((episode/'report.json').read_text())
    rows=[json.loads(line) for line in (episode/'trajectory.jsonl').read_text().splitlines()]
    prefix=next(i for i,r in enumerate(rows) if r['phase']=='bend_pull')
    env,source=replay_environment(episode,config,report)
    sys.path.insert(0,str(PROJECT_ROOT))
    from vista import Session,jaw_contact_state
    from grasp_pose import fixed_roll_ik
    output={'episode':str(episode),'physics_source':source,'prefix_error':0.,'trials':[],
            'model_calls':0,'new_media':0,'note':'Offline control comparison, not online success.'}
    try:
        for row in rows[:prefix]:
            error=max(float(np.max(abs(env.state()-row['state']))),
                      float(np.max(abs(env.base_pose()-row['base_pose']))))
            output['prefix_error']=max(output['prefix_error'],error)
            if error>1e-6:raise ValueError('Original prefix differs')
            env.drive(*row['base_command']);env.step(row.get('requested_action',row['action']))
        saved=mujoco.MjData(env.model);mujoco.mj_copyData(saved,env.model,env.data)
        names=('attached','contact_ticks','grasp_origins','events','frame','in_box_ticks','last_action',
               'drive_command','park_pose','stalk_metrics','worst_obstacle_penetration','worst_obstacle_pair',
               'worst_vegetation_penetration','pull_directions')
        attrs={n:copy.deepcopy(getattr(env,n)) for n in names}
        for distance in (.02,.03,.035):
            mujoco.mj_copyData(env.data,env.model,saved)
            for name,value in attrs.items():setattr(env,name,copy.deepcopy(value))
            trace=[]
            class Recorder:
                def __init__(self):self.out=out.parent
                def tick(self,q,phase):
                    env.step(q)
                    trace.append({'frame':env.frame,'phase':phase,'tcp':env.data.site_xpos[env.tcp].tolist(),
                        'bilateral_ids_offline':np.flatnonzero(env.bilateral_contacts()).tolist(),
                        'detached_ids_offline':np.flatnonzero(~env.attached).tolist(),
                        'metrics_offline':{int(i):copy.deepcopy(env.stalk_metrics[int(i)]) for i in np.flatnonzero(env.bilateral_contacts())}})
            session=Session(env,Recorder(),None,centered_grasp=True)
            session.grasp_attempted=True;session.target_stage='grasp'
            controller=session.centered;controller.carry_roll=float(env.last_action[4])
            start=env.data.site_xpos[env.tcp].copy()
            rot=env.data.xmat[env.grip_body].reshape(3,3)
            pitch=float(np.arcsin(np.clip(-rot[2,2],-1,1)))
            error=None
            def move(point,angle,phase):
                q,ik_error=fixed_roll_ik(env,point,-.15,controller.carry_roll,angle)
                if ik_error>.006:raise ValueError('IK error')
                if controller.self_collision.path_collision(env.last_action,q):raise ValueError('Robot self collision')
                for _ in range(180):
                    if np.max(abs(q-env.last_action))<1e-8:break
                    session.tick(q,phase)
                    if not jaw_contact_state(env)['both']:raise ValueError('Contact lost')
                else:raise ValueError('Motion timeout')
            try:
                for f in np.linspace(0,1,9)[1:]:move(start+[-distance*f,0,0],pitch,'pull_translate')
                for angle in np.linspace(pitch,np.deg2rad(-37),17)[1:]:move(start+[-distance,0,0],angle,'pull_bend')
                for _ in range(20):session.tick(env.last_action,'pull_hold')
            except ValueError as exc:error=str(exc)
            trial={'distance_m':distance,'error':error,'detached_ids_offline':np.flatnonzero(~env.attached).tolist(),
                   'bilateral_ids_offline':np.flatnonzero(env.bilateral_contacts()).tolist(),
                   'tracking':controller.last_tracking,'warnings':[int(w.number) for w in env.data.warning],
                   'trace':trace}
            output['trials'].append(trial)
            out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(output,indent=2)+'\n')
            print(json.dumps({k:v for k,v in trial.items() if k!='trace'}),flush=True)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();validate(args.episode.resolve(),args.out.resolve())
