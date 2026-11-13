"""Saved physics + actual semantic review, followed by an offline search fixture."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse,json,sys,tempfile
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
from replay_source import replay_environment


def validate(episode,out):
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    rows=[json.loads(l) for l in (episode/'trajectory.jsonl').read_text().splitlines()]
    original=next(e for e in events if e['kind']=='observation' and e['frame_id']=='f0000751')
    completed=next(e['result']['recovery'] for e in events if e['kind']=='receipt' and e['call']==4)
    assert completed['semantic_review_only'] and completed['end_frame']==694
    env,source=replay_environment(episode,json.loads((episode/'run-config.json').read_text()),json.loads((episode/'report.json').read_text()))
    sys.path.insert(0,str(PROJECT_ROOT / 'src'))
    from vista import Session
    from perception_recovery import review_proposals,exhausted_observation_recovery
    from visual_proposals import reviewed_candidates
    error=0.;trace=[]
    try:
        for row in rows[:751]:
            drift=max(float(np.max(abs(env.state()-row['state']))),float(np.max(abs(env.base_pose()-row['base_pose']))))
            error=max(error,drift)
            if drift>1e-6:raise ValueError('Saved physics prefix diverged')
            env.drive(*row['base_command']);env.step(row.get('requested_action',row['action']))
        with tempfile.TemporaryDirectory(prefix='semantic-recovery-') as tmp:
            class Recorder:
                out=Path(tmp)
                def tick(self,q,phase):env.step(q);trace.append(phase)
            s=Session(env,Recorder(),None,centered_grasp=True,visual_proposals=True)
            s.calls=6;s.station_attempts=original['station_search']['perception_calls']
            # Reapply the recorded post-motion review through the production
            # function, so provenance is not granted just by setting a flag.
            prior=next(e for e in events if e['kind']=='observation' and e['frame_id']=='f0000704')['station_search']['local_selection']
            local=s.local_targets;local.surveyed=True
            local.candidates=deepcopy(prior['candidates']);local.coverage=prior['coverage']
            local.selection_blocked_reason=prior['selection_blocked_reason']
            local.observation_adjustments=prior['observation_adjustments']
            local.proposal_review_attempts=1
            def request(call):
                e=next(e for e in events if e['kind']=='response' and e['call']==call)
                args=json.loads(e['message']['tool_calls'][0]['function']['arguments'])
                return {k:v for k,v in args.items() if k not in ('correction_reason','duplicate_links')}
            local.survey_review=request(3)
            local.survey_batch=next(e for e in events if e['kind']=='body_proposals' and e['frame_id']=='f0000604')
            batch=next(e for e in events if e['kind']=='body_proposals' and e['frame_id']=='f0000704')
            review=request(5)
            with Image.open(episode/'observations/f0000704-harvest_close.jpg') as rgb:
                measured,blocked=reviewed_candidates(review,batch,'f0000704',rgb)
                review_proposals(local,review,batch,measured,blocked,'Recorded actual model review after completed adjustment',rgb,env.base_pose())
            assert local.active['semantic_recovery_review_frame']==704
            frame,state,_=s.observe();views=deepcopy(s.stalk_view_evidence)
            assert exhausted_observation_recovery(s,frame)
            base=env.base_pose().copy();before=deepcopy(local.status())
            result,_=s.dispatch('search_row',{'basis_frame_id':frame,'direction':1},frame,state,0)
            assert result['status']=='executed'
            assert result['candidate_transition']['failures_after']==0
            assert result['candidate_transition']['resolution']=='station_viewpoint_exhausted'
            assert s.search_steps==1 and not s.local_targets.surveyed
            assert not s.target_confirmed and s.locked_target is None
            assert env.attached.all() and not any(w.number for w in env.data.warning)
            delta=env.base_pose()-base;assert .17<delta[1]<.185 and abs(delta[0])<.005
            output=dict(source_episode=str(episode),physics_source=source,prefix_error=error,
                fresh_views=views,before=before,result=result,actual_base_delta=delta.tolist(),
                phases=sorted(set(trace)),all_attached_offline=bool(env.attached.all()),
                warnings=[int(w.number) for w in env.data.warning],model_calls=0,videos=0,
                online_success_validated=False,
                limitation='Recorded semantic review, then manually supplied offline search request. No new model decision or harvest tested.')
            out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(output,ensure_ascii=False,indent=2))
            print(json.dumps({k:output[k] for k in ('prefix_error','fresh_views','actual_base_delta','all_attached_offline','warnings')}))
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();validate(a.episode.resolve(),a.out.resolve())
