"""Offline seed 12 prefix and bounded observation; no model or truth-based policy."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json,tempfile
from copy import deepcopy
import numpy as np
from replay_source import replay_environment


def validate(episode,out):
    cfg=json.loads((episode/'run-config.json').read_text())
    report=json.loads((episode/'report.json').read_text())
    events=[json.loads(x) for x in (episode/'vista-events.jsonl').read_text().splitlines()]
    rows=[json.loads(x) for x in (episode/'trajectory.jsonl').read_text().splitlines()]
    def request(call):
        e=next(e for e in events if e['kind']=='response' and e['call']==call)
        return json.loads(e['message']['tool_calls'][0]['function']['arguments'])
    review=request(3);frame='f0000108'
    original=next(e for e in events if e['kind']=='observation' and e['frame_id']==frame)
    env,source=replay_environment(episode,cfg,report)
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from vista import Session
    error=0.;trace=[]
    try:
        for row in rows[:108]:
            drift=max(float(np.max(abs(env.state()-row['state']))),
                      float(np.max(abs(env.base_pose()-row['base_pose']))))
            error=max(error,drift);assert drift<1e-9
            env.drive(*row['base_command']);env.step(row.get('requested_action',row['action']))
        with tempfile.TemporaryDirectory() as tmp:
            class Recorder:
                out=Path(tmp)
                def tick(self,q,phase):env.step(q);trace.append({'frame':env.frame,'phase':phase})
            s=Session(env,Recorder(),None,centered_grasp=True,visual_proposals=True)
            s.calls=4;s.station_attempts=3;local=s.local_targets
            local.surveyed=True;local.candidates=deepcopy(original['local_selection']['candidates'])
            local.coverage=review['coverage'];local.proposal_review_attempts=2
            local.survey_review={k:v for k,v in review.items() if k!='correction_reason'}
            local.survey_batch={k:v for k,v in next(e for e in events if e['kind']=='body_proposals'
                and e['frame_id']==review['frame_id']).items() if k not in ('kind','sim_time','frame_id')}
            local.survey_state=next(e for e in events if e['kind']=='observation' and e['frame_id']==review['frame_id'])
            local.survey_reference_depth=episode/'observations'/f"{review['frame_id']}-harvest_close-depth.npz"
            frame,state,_=s.observe();before_base=env.base_pose().copy()
            end,_=s.dispatch('finish',request(4),frame,state,0)
            assert not end['episode_ended'] and end['remaining_model_calls']==28
            item=next(c for c in local.candidates if c['proposal_id']=='P2')
            args={'basis_frame_id':frame,'candidate_key':item['candidate_key'],
                  'reason':'Offline fixture from recorded uncertainty: retain P2 and change arm occlusion before semantic review'}
            moved,_=s.dispatch('adjust_observation',args,frame,state,0)
            assert moved['recovery']['uncertain_region_review']['body_pixel_verified'] is False
            assert local.active is None and not local.can_search and not s.target_confirmed
            assert item['failures']==0 and not item['body_pixel_verified']
            after_frame,after_state,_=s.observe()
            # Keep the original uncertainty as an explicit offline fixture;
            # this validates the earned review, not a new model decision.
            follow=deepcopy(review);follow['frame_id']=after_frame
            follow['correction_reason']='Offline fixture: retain both maturity claims after adjustment; no model invoked'
            reviewed,_=s.dispatch('review_proposals',follow,after_frame,after_state,0)
            assert local.proposal_review_attempts==3 and local.active is None and not local.can_search
            assert not s.target_confirmed and env.attached.all() and not any(w.number for w in env.data.warning)
            delta=float(np.linalg.norm(env.base_pose()-before_base));assert delta<.002
            out.parent.mkdir(parents=True,exist_ok=True)
            result={'source_episode':str(episode),'physics_source':source,'prefix_frames':108,
                    'prefix_error':error,'finish_result':end,'offline_adjustment_args':args,
                    'adjustment':moved,'followup_review':reviewed,'trace':trace,
                    'base_pose_drift':delta,'all_attached_offline':True,'warnings':0,
                    'model_calls':0,'online_success_validated':False}
            out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            for camera in s.policy_cameras:
                import shutil
                shutil.copy2(s.frames[after_frame][camera],out.parent/f'after-{camera}.jpg')
            print('108-frame prefix exact; bounded observation and third review passed; uncertainty retained.',flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();validate(a.episode.resolve(),a.out.resolve())
