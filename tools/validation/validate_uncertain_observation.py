"""Archived full-scene command replay plus bounded semantic observation; no model/video."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse,json,tempfile,sys
from pathlib import Path
import numpy as np
from replay_source import replay_environment


def validate(episode,out):
    config=json.loads((episode/'run-config.json').read_text());report=json.loads((episode/'report.json').read_text())
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    rows=[json.loads(l) for l in (episode/'trajectory.jsonl').read_text().splitlines()]
    e,physics_source=replay_environment(episode,config,report)
    sys.path.insert(0,str(PROJECT_ROOT))
    from vista import Session,robot_state
    from visual_proposals import reviewed_candidates
    from PIL import Image
    trace=[];max_error=0.;saved_states={}
    class Recorder:
        def __init__(self,path):self.out=path
        def tick(self,q,phase):
            trace.append({'phase':phase,'base_command':e.drive_command.tolist()})
            e.step(q)
    try:
        # End before the original rejected second search request; all recorded
        # joint and drive commands are unchanged, no target truth is consulted.
        for row in rows[:664]:
            if e.frame in (560,611):saved_states[e.frame]=robot_state(e)
            error=max(np.max(abs(e.state()-row['state'])),np.max(abs(e.base_pose()-row['base_pose'])))
            max_error=max(max_error,float(error))
            if error>1e-6:raise ValueError('Archived prefix differs')
            e.drive(*row['base_command']);e.step(row.get('requested_action',row['action']))
        with tempfile.TemporaryDirectory(prefix='uncertain-observation-') as tmp:
            s=Session(e,Recorder(Path(tmp)),None,centered_grasp=True,visual_proposals=True);s.calls=10
            args=json.loads(next(ev['message']['tool_calls'][0]['function']['arguments'] for ev in events if ev['kind']=='response' and ev.get('call')==9))
            review={k:v for k,v in args.items() if k!='correction_reason'}
            batch={k:v for k,v in next(ev for ev in events if ev['kind']=='body_proposals' and ev['frame_id']=='f0000611').items() if k not in ('kind','frame_id','sim_time')}
            with Image.open(episode/'observations/f0000611-harvest_close.jpg') as rgb:
                measured,blocked=reviewed_candidates(review,batch,'f0000611',rgb)
            local=s.local_targets;local.register(measured,saved_states[611]['base_x_y_yaw'],review['coverage'])
            local.selection_blocked_reason=blocked;local.survey_review=review;local.survey_batch=batch
            local.survey_state=saved_states[611];local.proposal_review_attempts=1;s.station_attempts=2
            s.depth_frames[('f0000611','harvest_close')]=episode/'observations/f0000611-harvest_close-depth.npz'
            key=local.candidates[0]['candidate_key'];base=e.base_pose().copy()
            before_frame,state,_=s.observe()
            crops={}
            for camera in s.policy_cameras:
                info,image=s.active_candidate_crop(before_frame,state,camera,review_only=True)
                if info:
                    image.save(f'/tmp/uncertain-seed1-before-{camera}.jpg');crops[camera]=info
            receipt,_=s.dispatch('adjust_observation',{'basis_frame_id':before_frame,'candidate_key':key,'reason':'Offline recovery test for recorded uncertain red fragment'},before_frame,state,0)
            after_frame,state,_=s.observe()
            after_crops={}
            for camera in s.policy_cameras:
                info,image=s.active_candidate_crop(after_frame,state,camera,review_only=True)
                if info:
                    image.save(f'/tmp/uncertain-seed1-after-{camera}.jpg');after_crops[camera]=info
            # Exercise post-motion review calibration with uncertainty retained.
            # This is intentionally not a scripted maturity approval.
            reviewed,_=s.dispatch('review_proposals',{**args,'frame_id':after_frame,'correction_reason':'Offline protocol check: retain uncertainty; online semantic re-review still needed'},after_frame,state,0)
            result={'source_episode':str(episode),'physics_source':physics_source,'prefix_max_robot_error':max_error,
                    'model_calls':0,'online_success_claimed':False,'before_crops':crops,'after_crops':after_crops,
                    'adjustment':receipt,'post_adjustment_review':reviewed,'trace':trace,
                    'base_displacement_m':float(np.linalg.norm(e.base_pose()[:2]-base[:2])),
                    'all_fruits_attached_offline':bool(e.attached.all()),'warnings':[int(w.number) for w in e.data.warning]}
            assert result['base_displacement_m']<.005 and result['all_fruits_attached_offline'] and not any(result['warnings'])
            assert local.active is None and not local.can_search
            out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
            print(json.dumps({k:result[k] for k in ('prefix_max_robot_error','base_displacement_m','all_fruits_attached_offline','warnings','model_calls')}),flush=True)
    finally:e.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('out',type=Path);a=p.parse_args();validate(a.episode,a.out)
