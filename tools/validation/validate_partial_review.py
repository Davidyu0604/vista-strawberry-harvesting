"""Replay seed 5's saved semantic review through dispatch, without model or motion."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import json
import tempfile
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
from vista import Session
from perception_recovery import observation_adjustment_available


def validate(episode,out):
    events=[json.loads(line) for line in (episode/'vista-events.jsonl').read_text().splitlines()]
    def args(call):
        e=next(e for e in events if e['kind']=='response' and e['call']==call)
        return json.loads(e['message']['tool_calls'][0]['function']['arguments'])
    def observation(frame):
        return next(e for e in events if e['kind']=='observation' and e['frame_id']==frame)
    def batch(frame):
        return next(e for e in events if e['kind']=='body_proposals' and e['frame_id']==frame)
    old=args(5);review=args(7);original=deepcopy(review);frame=review['frame_id']
    obs=observation(frame);prior=observation(old['frame_id'])
    with tempfile.TemporaryDirectory() as tmp:
        s=Session(SimpleNamespace(meta={'tilted_stalk_prior':True}),SimpleNamespace(out=Path(tmp)),None,visual_proposals=True)
        # No physical environment: only dispatch routing and saved RGB-D validation.
        s.centered=object();s.calls=7;s.station_attempts=obs['station_search']['perception_calls']
        local=s.local_targets;local.surveyed=True
        local.candidates=deepcopy(obs['local_selection']['candidates'])
        local.selection_blocked_reason=obs['local_selection']['selection_blocked_reason']
        local.coverage=old['coverage'];local.survey_review=old;local.survey_batch=batch(old['frame_id'])
        local.survey_state=prior
        s.frames={frame:{camera:episode/path for camera,path in obs['images'].items()}}
        s.depth_frames={(f,'harvest_close'):episode/'observations'/f'{f}-harvest_close-depth.npz'
                        for f in (old['frame_id'],frame)}
        s.proposal_frames={frame:batch(frame)}
        with patch('vista.robot_state',return_value=obs):
            result,_=s.dispatch('review_proposals',review,frame,obs,0)
        assert review==original
        effective={c['proposal_id']:c['maturity'] for c in local.survey_review['candidates']}
        assert effective=={'P1':'turning','P2':'uncertain','P3':'not_fruit'},effective
        assert result['status']=='unresolved' and len(result['review']['deferred_corrections'])==1
        assert local.active is None and not local.can_search
        assert observation_adjustment_available(s,frame)
        assert 'adjust_observation' in result['next_action']
        s.calls=9
        finish,_=s.dispatch('finish',args(9),frame,obs,0)
        assert not finish['episode_ended'] and not s.finished
        assert finish['remaining_model_calls']==23
        output={'episode':str(episode),'review_result':result,'finish_result':finish,
                'effective_maturity':effective,'model_calls':0,'physical_motion_executed':False,
                'online_success_validated':False,
                'note':'Original saved RGB-D and original model arguments; runtime robot state supplied from the saved observation. Tests recovery eligibility, not successful adjustment or harvest.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print('P1 accepted as turning; P2 retained uncertain; observation recovery offered; early finish reminded once.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode',type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();validate(a.episode.resolve(),a.out.resolve())
