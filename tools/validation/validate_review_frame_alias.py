"""Recheck archived malformed review calls using saved images, without model/motion."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json,tempfile
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
from vista import Session


def validate(batch_root,out):
    results=[]
    for seed,survey_call,calls in ((7,4,(5,6,7,8,10,11)),(10,3,(4,5))):
        episode=batch_root/f'seed-{seed}'
        events=[json.loads(line) for line in (episode/'vista-events.jsonl').read_text().splitlines()]
        def args(call):
            e=next(e for e in events if e['kind']=='response' and e['call']==call)
            return json.loads(e['message']['tool_calls'][0]['function']['arguments'])
        def observation(frame):return next(e for e in events if e['kind']=='observation' and e['frame_id']==frame)
        def batch(frame):return next(e for e in events if e['kind']=='body_proposals' and e['frame_id']==frame)
        old=args(survey_call)
        for call in calls:
            submitted=args(call);original=deepcopy(submitted)
            frame=submitted.get('frame_id',submitted.get('basis_frame_id'));obs=observation(frame)
            with tempfile.TemporaryDirectory() as tmp:
                s=Session(SimpleNamespace(meta={'tilted_stalk_prior':True}),SimpleNamespace(out=Path(tmp)),None,visual_proposals=True)
                s.centered=object();s.calls=call;s.station_attempts=obs['station_search']['perception_calls']
                local=s.local_targets;local.surveyed=True
                status=obs['local_selection'];local.candidates=deepcopy(status['candidates'])
                local.selection_blocked_reason=status['selection_blocked_reason'];local.coverage=old['coverage']
                local.survey_review=deepcopy(old);local.survey_batch=batch(old['frame_id']);local.survey_state=observation(old['frame_id'])
                s.frames={frame:{cam:episode/path for cam,path in obs['images'].items()}}
                s.depth_frames={(f,'harvest_close'):episode/'observations'/f'{f}-harvest_close-depth.npz' for f in (old['frame_id'],frame)}
                s.proposal_frames={frame:batch(frame)}
                with patch('vista.robot_state',return_value=obs):
                    result,_=s.dispatch('review_proposals',submitted,frame,obs,0)
                assert submitted==original
                assert result['status'] in ('reviewed','unresolved') and 'frame_argument_normalized' in result
                assert result['review']['review']['candidates'][0]['maturity']=='turning'
                results.append({'seed':seed,'call':call,'result':result})
    out.write_text(json.dumps({'cases':results,'model_calls':0,'motion_executed':False,'online_success_validated':False},ensure_ascii=False,indent=2)+'\n')
    print(f'{len(results)} archived review requests accepted with current frame aliases; maturity checks retained.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();validate(a.batch.resolve(),a.out.resolve())
