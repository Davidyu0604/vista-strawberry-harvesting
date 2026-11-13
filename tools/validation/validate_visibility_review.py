"""Replay saved semantic tool inputs/RGB-D, no simulation, model calls or video."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from vista import Session


def validate(episode,out):
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    observations={e['frame_id']:e for e in events if e['kind']=='observation'}
    batches={e['frame_id']:{k:v for k,v in e.items() if k not in ('kind','sim_time','frame_id')}
             for e in events if e['kind']=='body_proposals'}
    responses={e['call']:e['message']['tool_calls'][0]['function'] for e in events if e['kind']=='response'}
    out.mkdir(parents=True,exist_ok=True)
    s=Session(SimpleNamespace(meta={'tilted_stalk_prior':True}),SimpleNamespace(out=out),None,visual_proposals=True)
    results=[]
    def state_for(frame):
        return {k:observations[frame][k] for k in ('joints_rad','tcp_world_m','base_x_y_yaw')}
    for call in (9,10,11,13):
        tool=responses[call];args=json.loads(tool['arguments']);frame=args['frame_id'];state=state_for(frame)
        s.frames[frame]={'harvest_close':episode/'observations'/f'{frame}-harvest_close.jpg'}
        s.depth_frames[(frame,'harvest_close')]=episode/'observations'/f'{frame}-harvest_close-depth.npz'
        s.proposal_frames={frame:batches[frame]}
        if call==13:
            # Use the already executed original observation adjustment; this
            # harness does not command or pretend to simulate new movement.
            motion=next(e['result']['recovery'] for e in events if e['kind']=='receipt' and e.get('call')==12)
            assert motion['semantic_review_only'] and motion['ticks']==26
            local=s.local_targets
            local.survey_reference_depth=s.depth_frames[(local.survey_review['frame_id'],'harvest_close')]
            local.survey_state=state
            local.observation_adjustments=1;local.candidates[0]['observation_adjusted']=True
            s.forget_local_target()
        with patch('vista.robot_state',return_value=state):
            result,_=s.dispatch(tool['name'],args,frame,state,0)
        results.append({'original_call':call,'tool':tool['name'],'result':result})
        if call==11:
            assert s.local_targets.active is None and not s.local_targets.can_search
            assert s.local_targets.candidates[0]['visibility']=='occluded'
    active=s.local_targets.active
    assert active and active['proposal_id']=='P1' and active['maturity']=='ripe' and active['visibility']=='clear'
    assert active['status']=='pending' and active['observation_adjusted'] and active['failures']==0
    assert s.local_targets.proposal_review_attempts==2 and not s.target_confirmed and not s.grasp_attempted
    report={'source_episode':str(episode),'method':'Saved model arguments and RGB-D dispatch regression; recorded motion bookkeeping only',
            'model_calls':0,'simulation_steps':0,'new_online_harvest_validated':False,'results':results,
            'same_candidate_active':active['candidate_key'],'stalk_confirmation_still_required':True,
            'next_action':s.station_status()['next_action']}
    (out/'saved-review-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:report[k] for k in ('model_calls','simulation_steps','same_candidate_active','next_action')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('out',type=Path);a=p.parse_args();validate(a.episode,a.out)
