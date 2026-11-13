"""Saved two-camera seed2 identity regression; no model or physics execution."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from vista import Session, depth_point
from perception_recovery import current_body_proposals


def validate(episode,out):
    events=[json.loads(line) for line in (episode/'vista-events.jsonl').read_text().splitlines()]
    obs={e['frame_id']:e for e in events if e['kind']=='observation'}
    responses={e['call']:e['message']['tool_calls'][0]['function'] for e in events if e['kind']=='response'}
    batches={e['frame_id']:{k:v for k,v in e.items() if k not in ('kind','sim_time','frame_id')}
             for e in events if e['kind']=='body_proposals'}
    out.mkdir(parents=True,exist_ok=True)
    session=Session(SimpleNamespace(meta={'tilted_stalk_prior':True}),SimpleNamespace(out=out),None,visual_proposals=True)
    survey=json.loads(responses[1]['arguments']);original=json.loads(responses[6]['arguments'])
    for frame in (survey['frame_id'],original['frame_id']):
        session.frames[frame]={camera:episode/'observations'/f'{frame}-{camera}.jpg' for camera in ('harvest_close','wrist')}
        for camera in ('harvest_close','wrist'):
            session.depth_frames[(frame,camera)]=episode/'observations'/f'{frame}-{camera}-depth.npz'
    state_keys=('joints_rad','tcp_world_m','base_x_y_yaw')
    frame=survey['frame_id'];state={k:obs[frame][k] for k in state_keys}
    session.proposal_frames[frame]=batches[frame]
    with patch('vista.robot_state',return_value=state):session.dispatch('survey_candidates',survey,frame,state,0)
    frame=original['frame_id'];state={k:obs[frame][k] for k in state_keys}
    station=obs[frame]['station_search'];local=session.local_targets
    local.candidates=deepcopy(station['local_selection']['candidates'])
    local.observation_adjustments=station['local_selection']['observation_adjustments']
    local.survey_state=state  # Actual completed arm adjustment; close-camera calibration checked in dispatch.
    session.station_attempts=station['perception_calls']
    with Image.open(session.frames[frame]['harvest_close']) as rgb, np.load(session.depth_frames[(frame,'harvest_close')]) as sample:
        session.proposal_frames[frame]=current_body_proposals(local,rgb,sample['depth'],sample,state['base_x_y_yaw'],depth_point)
    before=deepcopy(local)
    with patch('vista.robot_state',return_value=state):
        try:session.dispatch('review_proposals',original,frame,state,0)
        except ValueError as exc:rejection=str(exc)
        else:raise AssertionError('Original not_fruit claim must remain rejected')
    assert local.candidates==before.candidates and not local.can_search
    session.local_targets=before
    revised=deepcopy(original)
    next(c for c in revised['candidates'] if c['proposal_id']=='P2')['maturity']='uncertain'
    revised['duplicate_links']=[{'fragment_id':'P2','parent_id':'P1',
                                'reason':next(c['reason'] for c in original['candidates'] if c['proposal_id']=='P2')}]
    with patch('vista.robot_state',return_value=state):
        result,_=session.dispatch('review_proposals',revised,frame,state,0)
    local=session.local_targets
    assert local.can_search and len(local.candidates)==1 and not session.target_confirmed
    assert local.candidates[0]['proposal_id']=='P1' and local.candidates[0]['failures']==2
    assert local.candidates[0]['status']=='unavailable' and local.observation_adjustments==1
    report={'episode':str(episode),'model_calls':0,'simulation_steps':0,
            'original_request_rejected':rejection,'revised_request':revised,
            'duplicate_checks':result['review']['duplicate_checks'],
            'local_selection':local.status(),'next_action':session.station_status()['next_action'],
            'limitation':'Offline adapted tool request: retain P2 uncertain and explicitly link it to P1 using the saved model explanation. No new online identity decision, driving or harvest verified.'}
    (out/'saved-duplicate-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'status':result['status'],'can_search':local.can_search,'checks':report['duplicate_checks']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('episode',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();validate(args.episode.resolve(),args.out.resolve())
