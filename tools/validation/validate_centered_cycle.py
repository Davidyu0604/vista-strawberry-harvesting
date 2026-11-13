"""Full-scene centered grasp -> pull -> rear basket -> return. JSON only.

Archived RGB-D target/axis start the experiment. No new model requests or media.
Object truth is sampled only by the audit, not by control decisions.
"""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse
from pathlib import Path
import sys
import mujoco
import numpy as np
from grasp_pose import fixed_roll_ik
from diagnose_approach_path import run, fk


def audit_cycle(result):
    """Read-only acceptance of a saved run; never used to choose actions."""
    c=result.get('continuation',{});evaluation=c.get('evaluation',{})
    target=result['offline_associated_fruit'];trace=c.get('trace',[])
    carried=[]
    for row in trace:
        if row['phase']=='delivery_release':break
        carried.append(row)
    continuous=bool(carried and any(r['phase']=='delivery_lower' for r in carried)
                    and all(target in r['stalk_bilateral_ids_offline'] for r in carried))
    matching=bool(evaluation.get('single_fruit_delivered') and
                  evaluation.get('ripe_fruit_in_basket')==[target])
    complete=bool(evaluation.get('success') and matching and continuous
                  and result.get('stable_target_stalk_hold')
                  and result.get('failure') is None and c.get('error') is None
                  and not any(c.get('warnings',[1])))
    return {'same_target_ripe_fruit_delivered':matching,
            'same_stalk_contact_every_sample_until_release':continuous,
            'full_cycle_passed':complete}


def continue_cycle(env,mode):
    # Source selection must already have imported the recorded physics before
    # importing the current policy/evaluator. Never build two scenes in parallel.
    sys.path.insert(0,str(PROJECT_ROOT / 'src'))
    from vista import Session, jaw_contact_state, evaluate
    e=env;trace=[];scratch=mujoco.MjData(e.model)
    roll=float(e.last_action[4]);original_ik=e.ik
    if mode=='hold-roll':
        e.ik=lambda pos,gripper=1.,start=None,approach_pitch=0.:fixed_roll_ik(e,pos,gripper,roll,approach_pitch,start)
    session=Session.__new__(Session)
    session.e=e;session.grasp_attempted=True;session.approach_pitch=0.
    blocked=0;max_step=0.;max_joint=0.;lost=0
    def tick(goal,phase):
        nonlocal blocked,max_step,max_joint,lost
        old=e.last_action.copy();delta=np.asarray(goal)-old
        scale=min(1.,.04/max(np.max(np.abs(delta)),1e-10))
        origin=fk(e,old,scratch)
        while np.linalg.norm(fk(e,old+scale*delta,scratch)-origin)>.005:scale*=.8
        q=old+scale*delta;expected=fk(e,q,scratch)
        max_step=max(max_step,float(np.linalg.norm(expected-origin)))
        max_joint=max(max_joint,float(np.max(np.abs(q-old))))
        e.drive(0,0);e.step(q)
        tactile=jaw_contact_state(e)
        error=float(np.linalg.norm(e.data.site_xpos[e.tcp]-expected))
        blocked=blocked+1 if error>.008 else 0
        # Audit-only fields below are never fed back into the controller.
        trace.append({'frame':e.frame,'phase':phase,'base_pose':e.base_pose().tolist(),
            'tcp':e.data.site_xpos[e.tcp].tolist(),'requested_action':q.tolist(),'state':e.state().tolist(),
            'tcp_tracking_error_m':error,'external_contact':tactile,
            'stalk_bilateral_ids_offline':np.flatnonzero(e.bilateral_contacts()).tolist(),
            'detached_ids_offline':np.flatnonzero(~e.attached).tolist()})
        if blocked>=3:raise ValueError('Tracking error above 8 mm for 3 ticks: '+phase)
        if phase=='bend_pull':
            lost=0 if tactile['both'] else lost+1
            if lost>=4:raise ValueError('External jaw contact lost for 0.2 s during pull')
    session.tick=tick
    failure=None;delivery=None;pull=None
    try:
        if not session.check_contact_hold()['stable']:raise ValueError('External contact hold failed before pull')
        pos=e.data.site_xpos[e.tcp].copy()+[-.035,0,0]
        q,error=e.ik(pos,-.15,approach_pitch=np.deg2rad(-37))
        if error>.006:raise ValueError('Pull waypoint unreachable')
        for _ in range(60):tick(q,'bend_pull')
        if np.linalg.norm(e.data.site_xpos[e.tcp]-pos)>.008:raise ValueError('Pull waypoint not reached')
        pull={'status':'executed','offset_m':[-.035,0,0],'pitch_deg':-37,'seconds':3.}
        delivery=session.deliver_to_basket()
    except (ValueError,RuntimeError) as exc:failure=str(exc)
    finally:
        e.drive(0,0);e.ik=original_ik
    return {'mode':mode,'roll_command_rad':roll,'pull':pull,'delivery':delivery,'error':failure,
            'evaluation':evaluate(e,trace),'warnings':[int(w.number) for w in e.data.warning],
            'events_offline':e.events,'max_requested_tcp_step_mm':max_step*1000,
            'max_requested_joint_step_rad':max_joint,'trace':trace,
            'note':'Offline continuation using robot/tactile feedback; no fresh vision or online-success claim.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['legacy','hold-roll'],default='hold-roll')
    a=p.parse_args()
    run(a.episode.resolve(),'roll-centered',a.out,3.,lambda e:continue_cycle(e,a.mode))
