"""Replay an archived blocked observation and exercise current recovery offline.

No model calls; action choice is a scripted diagnostic, not a harvested episode.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / 'src'))


def validate(env, episode, output, prefix, call, mode, error):
    from vista import Session,robot_state
    from perception_recovery import uncertain_viewpoint_search_available
    report=json.loads((episode/'report.json').read_text())
    events=[json.loads(x) for x in (episode/'vista-events.jsonl').read_text().splitlines()]
    observation=next(e for e in events if e['kind']=='observation' and e['frame_id']==f'f{prefix:07d}')
    reviews=[e['review'] for e in report['perception_recoveries'] if e['kind']=='proposal_review' and int(e['review']['frame_id'][1:])<=prefix]
    review=reviews[-1]
    old_batch=next(e for e in events if e['kind']=='body_proposals' and e['frame_id']==review['frame_id'])
    trace=[]
    class Recorder:
        out=output.parent/f'{output.stem}-observations'
        def tick(self,q,phase):
            env.step(q)
            trace.append(dict(frame=env.frame,phase=phase,base_pose=env.base_pose().tolist()))
    Recorder.out.mkdir(parents=True,exist_ok=True)
    session=Session(env,Recorder(),None,centered_grasp=True,visual_proposals=True)
    local=session.local_targets
    status=observation['local_selection']
    for key in ('surveyed','candidates','coverage','selection_blocked_reason','survey_correction_attempts','proposal_review_attempts','duplicate_links','observation_adjustments'):
        if key in status:setattr(local,key,deepcopy(status[key]))
    local.survey_review=deepcopy(review)
    local.survey_batch={k:deepcopy(v) for k,v in old_batch.items() if k not in ('kind','sim_time','frame_id')}
    local.survey_state=robot_state(env)
    local.survey_reference_depth=episode/'observations'/f"{review['frame_id']}-harvest_close-depth.npz"
    session.calls=call;session.station_attempts=observation['station_search']['perception_calls']
    session.search_history=deepcopy(observation.get('row_search_guidance',{}).get('completed_searches',[]))
    session.search_steps=len(session.search_history)
    frame,state,_=session.observe()
    result=dict(episode=str(episode),prefix_frames=prefix,prefix_error=error,model_calls=0,
                mode=mode,old_next_action=observation['station_search']['next_action'],
                new_next_action=session.station_status()['next_action'],before=local.status(),
                note='Exact recorded physics prefix; saved RGB-D/semantic queue restored. Scripted offline recovery, not online harvest success.')
    if mode=='review':
        response=next(e for e in events if e['kind']=='response' and e['call']==call)
        tool=response['message']['tool_calls'][0]['function'];assert tool['name']=='review_proposals'
        args=json.loads(tool['arguments']);args['frame_id']=frame
        receipt,_=session.dispatch('review_proposals',args,frame,state,0)
        result['receipt']=receipt
        assert receipt['status']!='rejected',receipt
        assert any(r['proposal_id']=='P1' for r in receipt['review']['changed_depth_regions'])
        assert local.active and local.active['proposal_id']=='P2',local.status()
        assert local.active['body_pixel_verified'] is True
        assert not session.target_confirmed
        result['active_after_review']=local.active
        result['stalk_proposals_available']=deepcopy(session.stalk_proposals)
    else:
        assert uncertain_viewpoint_search_available(session,frame),session.station_status()
        before=deepcopy(local.candidates)
        receipt,_=session.dispatch('search_row',{'basis_frame_id':frame,'direction':1},frame,state,0)
        result['receipt']=receipt
        assert receipt['status']=='executed',receipt
        assert receipt['observation_recovery']['kind']=='uncertain_viewpoint_search'
        assert receipt['observation_recovery']['original_selection']['candidates']==before
        assert not session.local_targets.surveyed and session.local_targets.active is None
        assert session.local_targets.allow_uncertain_deferral
        assert .17<receipt['actual_delta_y_m']<.185
        after,_,_=session.observe()
        result['fresh_frame']=after;result['fresh_proposals']=session.proposal_frames[after]
    result.update(trace=trace,warnings=[int(w.number) for w in env.data.warning],
                  attached_all_offline=bool(env.attached.all()),
                  source_sha256={n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest() for n in ('vista.py','local_targets.py','perception_recovery.py')})
    assert not any(result['warnings']);assert result['attached_all_offline']
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('prefix_frames','prefix_error','mode','new_next_action','warnings')},ensure_ascii=False),flush=True)


def main():
    from replay_source import replay_environment
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode',type=Path);p.add_argument('--prefix',type=int,required=True)
    p.add_argument('--call',type=int,required=True);p.add_argument('--mode',choices=('search','review'),required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    episode=args.episode.resolve();out=args.out.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    config=json.loads((episode/'run-config.json').read_text());report=json.loads((episode/'report.json').read_text())
    env,_=replay_environment(episode,config,report)
    error=0.
    try:
        for row in map(json.loads,(episode/'trajectory.jsonl').read_text().splitlines()):
            if env.frame>=args.prefix:break
            diff=max(float(np.max(abs(env.state()-row['state']))),float(np.max(abs(env.base_pose()-row['base_pose']))))
            error=max(error,diff);assert diff<1e-9,diff
            env.drive(*row['base_command']);env.step(row['requested_action'])
        assert env.frame==args.prefix
        validate(env,episode,out,args.prefix,args.call,args.mode,error)
    finally:env.close()

if __name__=='__main__':main()
