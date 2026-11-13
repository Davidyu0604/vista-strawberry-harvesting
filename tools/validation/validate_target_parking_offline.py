"""Full-scene offline checkpoint test; no model calls or automatic training export.

Replays real controls to a visually confirmed target, scores parking, executes
its chosen move, remeasures via RGB-D and tries the existing harvest controller.
Projected correspondence replaces model reselection, so this is NOT a policy
success-rate experiment. Hidden object identity is used only by final evaluate.
"""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from peduncle import PeduncleEnv
from vista import Session, tilted_stalk_grasp_point, verify_projected_point, depth_point, evaluate, jaw_contact_state
from target_parking import optimized_parking_plan
from visual_maturity import assess_maturity


def validate(episode, out, call=None):
    out.mkdir(parents=True, exist_ok=False)
    config=json.loads((episode/'run-config.json').read_text())
    rows=[json.loads(l) for l in (episode/'trajectory.jsonl').read_text().splitlines()]
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    event=next(r for r in events if r['kind']=='receipt' and isinstance(r.get('result'),dict)
               and r['result'].get('target_confirmed') and (call is None or r.get('call')==call))
    target=np.array(event['result']['locked_target_m'])
    e=PeduncleEnv(seed=config['seed'],single=False,fixed_base=config.get('fixed_base',False),render_quality=config['quality'])
    result={'episode':str(episode.resolve()),'confirmed_call':event['call'],'seed':config['seed'],
            'model_calls':0,'scope':__doc__,'harvest_success':False,'last_completed_stage':'init'}
    try:
        drift=0.
        for row in rows:
            drift=max(drift,float(np.max(np.abs(e.state()-row['state']))))
            if e.data.time>=event['sim_time']-1e-6:break
            e.drive(*row['base_command']);e.step(row['requested_action'])
        result.update(checkpoint_frame=e.frame,max_replay_joint_error_rad=drift)
        if drift>1e-8:raise ValueError('Replay differs from recorded robot state')
        qpos=e.data.qpos.copy();qvel=e.data.qvel.copy();simtime=e.data.time
        started=time.monotonic();plan=optimized_parking_plan(e,target)
        result.update(plan=plan,planning_wall_seconds=time.monotonic()-started,
                      planning_did_not_mutate_live_state=bool(np.array_equal(qpos,e.data.qpos) and np.array_equal(qvel,e.data.qvel) and e.data.time==simtime))
        if not result['planning_did_not_mutate_live_state']:raise ValueError('Planning changed live state')
        print(json.dumps({'seed':config['seed'],'call':event['call'],'recommended_delta_y_m':plan['recommended_delta_y_m'],
                          'current':plan['current'],'selected':plan['selected'],'planning_seconds':result['planning_wall_seconds']}),flush=True)
        (out/'plan.json').write_text(json.dumps(result,indent=2))
        s=Session.__new__(Session);s.e=e;s.max_calls=32;s.calls=event['call']
        s.grasp_attempted=False;s.target_stage=None;s.target_confirmed=True
        s.locked_target=target.copy();s.target_camera='harvest_close';s.approach_pitch=0
        s.maturity_evidence=event['result']['maturity_check'];s.crops={};s.frames={};s.depth_frames={}
        s.target_aware_parking=True;s.parking_history=[];s.parking_anchor=None
        s.search_steps=sum(1 for r in events if r['kind']=='receipt' and r['sim_time']<event['sim_time']
                           and isinstance(r.get('result'),dict) and r['result'].get('stage')=='search' and r['result'].get('status')=='executed')
        trace=[]
        def tick(q,phase):
            e.step(q);trace.append({'base_pose':e.base_pose().tolist()})
            if any(w.number for w in e.data.warning):raise ValueError('Physics warning')
        s.tick=tick
        if plan['recommended_delta_y_m'] is not None:
            result['parking_receipt']=s.refine_parking(plan=plan)
        elif not plan['current']['reachable']:raise ValueError('No reachable parking candidate')
        result['last_completed_stage']='parked'
        rgb=Image.fromarray(e.render('harvest_close',480,360));depth,cal=e.depth_view('harvest_close',480,360)
        rgb.save(out/'after-parking.jpg')
        check=verify_projected_point(target,depth,cal);result['target_surface_check']=check
        if not check['consistent']:raise ValueError('Original measured target not visible after parking')
        x,y=check['sample_pixel'];point=depth_point(depth,cal,x,y)
        measured,fit=tilted_stalk_grasp_point(point,depth,cal,rgb)
        result['remeasurement_error_m']=float(np.linalg.norm(measured-target))
        if np.linalg.norm(measured-target)>.008:raise ValueError('RGB-D remeasurement disagrees with original target')
        maturity,marked=assess_maturity(measured,fit['axis_world'],depth,cal,rgb)
        result['maturity']=maturity;marked.save(out/'remeasured-target.jpg')
        if not maturity['passed']:raise ValueError('Fresh visual maturity check failed')
        checks={}
        for camera in ('top','wrist'):
            d,c=e.depth_view(camera,480,360);checks[camera]=verify_projected_point(measured,d,c)
        result['cross_view_checks']=checks
        if not any(c['consistent'] for c in checks.values()):raise ValueError('Cross-view confirmation failed')
        s.locked_target=measured;s.target_confirmed=True;s.maturity_evidence=maturity
        result['last_completed_stage']='relocalized'
        result['pregrasp']=s.execute_approach('pregrasp')
        result['grasp']=s.execute_approach('grasp');result['last_completed_stage']='approached'
        result['close']=s.execute([{'action':'gripper','opening':-.15,'seconds':2.5}])
        if result['close'][0]['status']!='executed':raise ValueError('Close rejected')
        result['contact_hold']=s.check_contact_hold()
        if not result['contact_hold']['stable']:raise ValueError('No stable external bilateral contact after closing')
        result['last_completed_stage']='stable_hold'
        for dx,dz,pitch,seconds in [(-.035,0,-37,3),(-.045,.005,-37,2.5),(-.05,.02,-30,2.5)]:
            receipt=s.execute([{'action':'bend_pull','dx':dx,'dy':0,'dz':dz,'pitch_deg':pitch,'seconds':seconds}])
            result.setdefault('pulls',[]).append(receipt)
            if receipt[0]['status']!='executed':raise ValueError('Pull rejected: '+receipt[0].get('reason',''))
            for _ in range(20):tick(e.last_action,'offline_inspect_pull')
        result['delivery']=s.deliver_to_basket()
        for _ in range(40):tick(e.last_action,'terminal_settle')
        result['evaluation']=evaluate(e,trace);result['harvest_success']=result['evaluation']['success']
        result['last_completed_stage']='delivery_evaluated'
    except (ValueError,RuntimeError) as exc:result['reason']=str(exc)
    finally:
        result['physics_warnings']=bool(any(w.number for w in e.data.warning))
        e.drive(0,0);e.close()
        (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k in ('seed','last_completed_stage','reason','harvest_success','physics_warnings','max_replay_joint_error_rad')}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--call',type=int)
    a=p.parse_args();validate(a.episode,a.out,a.call)
