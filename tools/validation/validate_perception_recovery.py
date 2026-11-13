"""Offline full-scene recovery from archived RGB-D decisions; no model/video."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse
import json
from pathlib import Path
import tempfile
import sys
import numpy as np
from replay_source import replay_environment


def validate(episode, output):
    config=json.loads((episode/'run-config.json').read_text())
    report=json.loads((episode/'report.json').read_text())
    events=[json.loads(line) for line in (episode/'vista-events.jsonl').read_text().splitlines()]
    rows=[json.loads(line) for line in (episode/'trajectory.jsonl').read_text().splitlines()]
    survey=json.loads(next(e['message']['tool_calls'][0]['function']['arguments'] for e in events
                         if e['kind']=='response' and e['call']==1))
    e,source=replay_environment(episode,config,report)
    sys.path.insert(0,str(PROJECT_ROOT))
    from vista import Session,robot_state,evaluate,depth_point
    from PIL import Image
    from visual_proposals import propose_bodies
    trace=[];receipts=[];max_error=0.;failure=None
    class Recorder:
        def __init__(self,out):self.out=out
        def tick(self,q,phase):
            # All privileged evaluation happens after control has finished.
            trace.append({'phase':phase,'base_pose':e.base_pose().tolist()})
            e.step(q)
    with tempfile.TemporaryDirectory(prefix='perception-recovery-') as tmp:
        s=Session(e,Recorder(Path(tmp)),None,centered_grasp=True,visual_proposals=True)
        (Path(tmp)/'observations').mkdir()
        def observe():return s.observe()[:2]
        def archived_observe():
            frame=f'f{e.frame:07d}';state=robot_state(e)
            s.frames[frame]={}
            for camera in ('harvest_close','wrist'):
                s.frames[frame][camera]=episode/'observations'/f'{frame}-{camera}.jpg'
                s.depth_frames[(frame,camera)]=episode/'observations'/f'{frame}-{camera}-depth.npz'
            with Image.open(s.frames[frame]['harvest_close']) as rgb,np.load(s.depth_frames[(frame,'harvest_close')]) as sample:
                s.proposal_frames={frame:propose_bodies(rgb,sample['depth'],sample,state['base_x_y_yaw'],depth_point)}
            s.stalk_proposals={}
            for camera in ('harvest_close','wrist'):s.active_candidate_crop(frame,state,camera)
            return frame,state
        def replay_to(stop):
            nonlocal max_error
            for row in rows[e.frame:stop]:
                error=max(float(np.max(abs(e.state()-row['state']))),float(np.max(abs(e.base_pose()-row['base_pose']))))
                max_error=max(max_error,error)
                if error>1e-6:raise ValueError('Archived prefix mismatch')
                e.drive(*row['base_command']);e.step(row.get('requested_action',row['action']))
        def dispatch(name,args,frame,state):
            result,_=s.dispatch(name,args,frame,state,0)
            receipts.append({'tool':name,'result':result})
            return result
        try:
            # Exact original command prefix up to the initial survey image.
            prefix=int(survey['frame_id'][1:])
            replay_to(prefix)
            frame,state=archived_observe()
            dispatch('survey_candidates',survey,frame,state)
            next_frame=next(ev['basis_frame_id'] for ev in events if ev['kind']=='request' and ev['call']==2)
            replay_to(int(next_frame[1:]))
            frame,state=archived_observe()
            if s.local_targets.selection_blocked_reason:
                revised={**survey,'frame_id':frame,'missing_visible_ripe':False,
                         'coverage':survey['coverage']+' Offline scripted correction test: original model later retracted the omission claim.',
                         'correction_reason':'Offline protocol test using recorded model retraction; no new semantic validation.'}
                dispatch('correct_survey',revised,frame,state)
                frame,state=archived_observe()
            proposals=list(s.stalk_proposals.values())
            if not proposals:raise ValueError('No geometric stalk proposal for active fruit')
            # A deterministic scripted proposal is only offline interface evidence,
            # not a fresh model identification or success-rate observation.
            proposal=next((p for p in proposals if p['camera']=='wrist'),proposals[0])
            def locate(proposal,frame,state):
                return dispatch('locate_stalk_proposal',{'frame_id':frame,'candidate_key':proposal['candidate_key'],
                    'proposal_id':proposal['proposal_id'],'confirmed_short_stalk':True,
                    'reason':'Offline scripted geometric candidate; semantic identification not validated.'},frame,state)
            first=locate(proposal,frame,state)
            if not first.get('target_confirmed'):
                old_target=np.asarray(first['locked_target_m'])
                dispatch('adjust_observation',{'basis_frame_id':frame,'candidate_key':s.local_targets.active['candidate_key'],
                         'reason':'Recorded current cross-view depth inconsistency; offline recovery test.'},frame,state)
                assert s.locked_target is None and not s.target_confirmed and not s.frames
                frame,state=observe()
                proposals=list(s.stalk_proposals.values())
                if not proposals:raise ValueError('No stalk proposals after observation adjustment')
                # Correspondence to the previously measured RGB-D region only.
                proposal=min(proposals,key=lambda p:np.linalg.norm(np.asarray(p['target_m'])-old_target))
                locate(proposal,frame,state)
            if not s.target_confirmed:raise ValueError('Fresh two-view confirmation still failed')
            for stage in ('pregrasp','grasp'):
                frame,state=observe()
                result=dispatch('approach_target',{'basis_frame_id':frame,'stage':stage},frame,state)
                if result.get('status')!='executed':raise ValueError(str(result))
            result=s.execute([{'action':'gripper','opening':-.15,'seconds':3}]);receipts.append({'close':result})
            if result[0]['status']!='executed':raise ValueError(str(result))
            result=s.execute([{'action':'bend_pull','dx':-.035,'dy':0.,'dz':0.,'pitch_deg':-37,'seconds':3}]);receipts.append({'pull':result})
            if result[0]['status']!='executed':raise ValueError(str(result))
            # Mechanical continuation only; never fabricate an online visual review.
            receipts.append({'mechanical_delivery':s.deliver_to_basket()})
        except (ValueError,RuntimeError,AssertionError) as exc:
            failure=f'{type(exc).__name__}: {exc}'
        finally:
            output.parent.mkdir(parents=True,exist_ok=True)
            result={'episode':str(episode),'physics_source':source,'prefix_error':max_error,
                    'failure':failure,'evaluation':evaluate(e,trace),'warnings':[int(w.number) for w in e.data.warning],
                    'receipts':receipts,'recoveries':s.perception_recoveries,'local_selection':s.local_targets.status(),
                    'model_calls':0,'replay_videos':0,'temporary_observations_deleted':True,
                    'online_visual_review_validated':False,'note':'Scripted recovery and geometric proposal selection; not online success rate.'}
            output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k not in ('receipts','recoveries','local_selection')},ensure_ascii=False),flush=True)
            e.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();validate(a.episode.resolve(),a.out.resolve())
