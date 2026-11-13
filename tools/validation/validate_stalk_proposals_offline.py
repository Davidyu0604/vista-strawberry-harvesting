"""Validate numbered selection against archived RGB-D, without simulation or model calls."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from vista import Session


def validate(episode, frame):
    events=[json.loads(line) for line in (episode/'vista-events.jsonl').read_text().splitlines()]
    state=next(e for e in events if e['kind']=='observation' and e['frame_id']==frame)
    candidates=state['local_selection']['candidates']
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp);(out/'observations').mkdir()
        env=SimpleNamespace(meta={'harvest_mechanism':True,'tilted_stalk_prior':True,
                                  'basket_mount':'chassis','policy_cameras':['harvest_close','wrist']},
                            data=SimpleNamespace(time=state['simulation_time']))
        session=Session(env,SimpleNamespace(out=out),None,visual_proposals=True)
        session.local_targets.register(candidates,state['base_x_y_yaw'],'Archived visual survey')
        session.frames[frame]={camera:episode/'observations'/f'{frame}-{camera}.jpg'
                               for camera in session.policy_cameras}
        session.depth_frames={(frame,camera):episode/'observations'/f'{frame}-{camera}-depth.npz'
                              for camera in session.policy_cameras}
        session.plan_parking=lambda:{'evaluated':False,'reason':'Perception-only archive validation; no IK or motion'}
        batches={}
        for camera in session.policy_cameras:
            info,image=session.active_candidate_crop(frame,state,camera)
            batches[camera]=info
            if image is not None:image.save(Path('/tmp')/f'stalk-proposals-{camera}.png')
        results=[]
        for item in list(session.stalk_proposals.values()):
            session.forget_local_target()
            with patch('vista.robot_state',return_value=state):
                result,_=session._dispatch('locate_stalk_proposal',{
                    'frame_id':frame,'candidate_key':item['candidate_key'],'proposal_id':item['proposal_id'],
                    'confirmed_short_stalk':True,
                    'reason':'Offline interface validation, not a model semantic decision'},frame,state,0)
            results.append(result)
    return {'episode':str(episode.resolve()),'frame_id':frame,'batches':batches,'selections':results,
            'model_calls':0,'motion_validated':False,'videos_generated':0,
            'note':'Archived measured visual body and RGB-D only. Scripted confirmation is not an online success.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path);parser.add_argument('--frame',required=True)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    result=validate(args.episode,args.frame)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'out':str(args.out),'proposals':len(result['selections']),
                      'confirmed':sum(s.get('target_confirmed',False) for s in result['selections']),
                      'motion_validated':False},ensure_ascii=False))
