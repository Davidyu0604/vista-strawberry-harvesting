"""Saved seed2 RGB-D coverage correction regression, no simulation or model."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
from vista import Session,depth_point
from perception_recovery import current_body_proposals


def validate(episode,out):
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    obs={e['frame_id']:e for e in events if e['kind']=='observation'}
    batches={e['frame_id']:{k:v for k,v in e.items() if k not in ('kind','sim_time','frame_id')} for e in events if e['kind']=='body_proposals'}
    responses={e['call']:e['message']['tool_calls'][0]['function'] for e in events if e['kind']=='response'}
    out.mkdir(parents=True,exist_ok=True)
    s=Session(SimpleNamespace(meta={'tilted_stalk_prior':True}),SimpleNamespace(out=out),None,visual_proposals=True)
    results=[]
    for call in (9,10,11):
        t=responses[call];args=json.loads(t['arguments']);frame=args['frame_id']
        state={k:obs[frame][k] for k in ('joints_rad','tcp_world_m','base_x_y_yaw')}
        rgb_path=episode/'observations'/f'{frame}-harvest_close.jpg'
        depth_path=episode/'observations'/f'{frame}-harvest_close-depth.npz'
        s.frames[frame]={'harvest_close':rgb_path};s.depth_frames[(frame,'harvest_close')]=depth_path
        if call==9:batch=batches[frame]
        else:
            with Image.open(rgb_path) as rgb,np.load(depth_path) as sample:
                batch=current_body_proposals(s.local_targets,rgb,sample['depth'],sample,state['base_x_y_yaw'],depth_point)
        s.proposal_frames[frame]=batch
        row={'call':call,'old_labels_and_pixels':[(p['proposal_id'],p['pixel']) for p in batches[frame]['proposals']],
             'new_labels_and_pixels':[(p['proposal_id'],p['pixel']) for p in batch['proposals']]}
        with patch('vista.robot_state',return_value=state):
            try:r,_=s.dispatch(t['name'],args,frame,state,0);row['status']=r.get('status')
            except ValueError as exc:row['rejected']=str(exc)
        results.append(row)
    assert 'rejected' in results[1] and results[2]['status']=='corrected'
    assert s.local_targets.active and s.local_targets.active['candidate_key']=='view-candidate-1'
    assert not s.target_confirmed and not s.local_targets.can_search
    report={'episode':str(episode),'method':'Saved tool arguments with corrected original-region proposal display; no online semantic or harvest validation',
            'model_calls':0,'simulation_steps':0,'results':results,'next_action':s.station_status()['next_action'],
            'limitation':'Original call11 reason refers to old display pixel 138,90. New display keeps 145,84; online model must review the corrected display itself.'}
    (out/'saved-coverage-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();validate(a.episode.resolve(),a.out.resolve())
