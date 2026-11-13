"""Replay failed seed169 prefix; validate measured unlisted-fruit observation search."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse,json,sys,tempfile
from copy import deepcopy
from pathlib import Path
import numpy as np
from replay_source import replay_environment


def validate(episode,out):
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    rows=[json.loads(l) for l in (episode/'trajectory.jsonl').read_text().splitlines()]
    original=next(e for e in events if e['kind']=='observation' and e['frame_id']=='f0000090')
    env,source=replay_environment(episode,json.loads((episode/'run-config.json').read_text()),json.loads((episode/'report.json').read_text()))
    sys.path.insert(0,str(PROJECT_ROOT))
    from vista import Session
    from unlisted_fruit_search import review_available
    error=0.;trace=[]
    try:
        for row in rows[:90]:
            drift=max(float(np.max(abs(env.state()-row['state']))),float(np.max(abs(env.base_pose()-row['base_pose']))))
            error=max(error,drift)
            if drift>1e-6:raise ValueError('Saved physics prefix diverged')
            env.drive(*row['base_command']);env.step(row.get('requested_action',row['action']))
        with tempfile.TemporaryDirectory(prefix='edge-viewpoint-') as tmp:
            class Recorder:
                out=Path(tmp)
                def tick(self,q,phase):env.step(q);trace.append(phase)
            s=Session(env,Recorder(),None,centered_grasp=True,visual_proposals=True)
            s.calls=3;s.station_attempts=original['station_search']['perception_calls']
            local=s.local_targets;local.surveyed=True
            old=original['station_search']['local_selection']
            local.candidates=deepcopy(old['candidates']);local.coverage=old['coverage']
            local.selection_blocked_reason=old['selection_blocked_reason']
            local.observation_adjustments=old['observation_adjustments']
            local.proposal_review_attempts=old['proposal_review_attempts']
            response=next(e for e in events if e['kind']=='response' and e['call']==2)
            args=json.loads(response['message']['tool_calls'][0]['function']['arguments'])
            local.survey_review={k:v for k,v in args.items() if k!='correction_reason'}
            local.survey_batch=next(e for e in events if e['kind']=='body_proposals' and e['frame_id']=='f0000045')
            local.survey_batch={k:v for k,v in local.survey_batch.items() if k not in ('kind','sim_time','frame_id')}
            local.survey_state=original
            local.survey_correction_attempts=1
            frame,state,_=s.observe()
            assert not local.can_search and local.active is None
            assert review_available(s)
            assert 'search_after_unlisted_review' in s.station_status()['next_action']
            before=deepcopy(local.status())
            base=env.base_pose().copy()
            result,_=s.dispatch('search_after_unlisted_review',{'basis_frame_id':frame,'all_missing_claims_reported':True,'bodies':[{'x':340,'y':95,'reason':'Saved model report: visible red body at approximately x340 y95'}]},frame,state,0)
            assert result['status']=='executed' and result['observation_recovery']
            assert s.search_steps==1 and not s.local_targets.surveyed
            assert not s.target_confirmed and s.locked_target is None
            assert env.attached.all() and not any(w.number for w in env.data.warning)
            delta=env.base_pose()-base;assert .17<delta[1]<.185 and abs(delta[0])<.005
            assert local.survey_review['missing_visible_ripe'] is True
            assert local.selection_blocked_reason==before['selection_blocked_reason']
            assert s.unlisted_search_evidence is None
            evidence=result['observation_recovery']['evidence']
            output=dict(source_episode=str(episode),physics_source=source,prefix_error=error,
                fresh_evidence=evidence,before=before,result=result,actual_base_delta=delta.tolist(),
                phases=sorted(set(trace)),all_attached_offline=bool(env.attached.all()),
                warnings=[int(w.number) for w in env.data.warning],model_calls=0,videos=0,
                online_success_validated=False,
                limitation='Saved failed model queue and reported pixel; offline supplied new tool request, no new model decision or harvest tested.')
            out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(output,ensure_ascii=False,indent=2))
            print(json.dumps({k:output[k] for k in ('prefix_error','fresh_evidence','actual_base_delta','all_attached_offline','warnings')}))
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();validate(a.episode.resolve(),a.out.resolve())
