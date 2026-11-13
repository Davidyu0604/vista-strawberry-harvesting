"""Replay exact saved pre-search physics, then exercise recovery and one move."""

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
    request=next(e for e in events if e['kind']=='request' and e['call']==6)
    original=next(e for e in events if e['kind']=='observation' and e['frame_id']==request['basis_frame_id'])
    completed=next(e['result']['recovery']['end_frame'] for e in events if e['kind']=='receipt' and e['call']==3)
    env,source=replay_environment(episode,json.loads((episode/'run-config.json').read_text()),json.loads((episode/'report.json').read_text()))
    sys.path.insert(0,str(PROJECT_ROOT))
    from vista import Session
    from perception_recovery import exhausted_observation_recovery
    error=0.;trace=[]
    try:
        for row in rows[:int(original['frame_id'][1:])]:
            drift=max(float(np.max(abs(env.state()-row['state']))),float(np.max(abs(env.base_pose()-row['base_pose']))))
            error=max(error,drift)
            if drift>1e-6:raise ValueError('Saved physics prefix diverged')
            env.drive(*row['base_command']);env.step(row.get('requested_action',row['action']))
        with tempfile.TemporaryDirectory(prefix='recovery-check-') as tmp:
            class Recorder:
                out=Path(tmp)
                def tick(self,q,phase):env.step(q);trace.append(phase)
            session=Session(env,Recorder(),None,centered_grasp=True,visual_proposals=True)
            session.calls=6;session.station_attempts=original['station_search']['perception_calls']
            local=session.local_targets;local.surveyed=True
            old=original['station_search']['local_selection']
            local.candidates=deepcopy(old['candidates']);local.coverage=old['coverage']
            local.selection_blocked_reason=old['selection_blocked_reason']
            local.observation_adjustments=old['observation_adjustments']
            # Older recording predates the marker; derive it ONLY from the
            # logged successful physical recovery receipt, not assumed state.
            local.active['observation_recovery_completed_frame']=completed
            frame,state,_=session.observe();views=deepcopy(session.stalk_view_evidence)
            assert exhausted_observation_recovery(session,frame)
            before=deepcopy(local.status());base=env.base_pose().copy()
            result,_=session.dispatch('search_row',{'basis_frame_id':frame,'direction':1},frame,state,0)
            assert result['status']=='executed' and result['candidate_transition']['failures_after']==2
            assert session.search_steps==1 and not session.local_targets.surveyed
            assert not session.target_confirmed and session.locked_target is None
            assert env.attached.all() and not any(w.number for w in env.data.warning)
            assert not session.frames and not session.depth_frames and not session.crops
            delta=env.base_pose()-base;assert .17<delta[1]<.185 and abs(delta[0])<.005
            output={'source_episode':str(episode),'physics_source':source,'prefix_error':error,
                    'successful_recovery_end_frame':completed,'fresh_views':views,'before':before,
                    'result':result,'actual_base_delta':delta.tolist(),'phases':sorted(set(trace)),
                    'all_attached_offline':bool(env.attached.all()),'warnings':[int(w.number) for w in env.data.warning],
                    'model_calls':0,'videos':0,'online_success_validated':False,
                    'limitation':'Mechanical continuation of saved model search request. No new online decision, relocalization or harvest validated.'}
            out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(output,ensure_ascii=False,indent=2))
            print(json.dumps({k:output[k] for k in ('prefix_error','successful_recovery_end_frame','fresh_views','actual_base_delta','all_attached_offline','warnings')},ensure_ascii=False))
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();validate(a.episode.resolve(),a.out.resolve())
