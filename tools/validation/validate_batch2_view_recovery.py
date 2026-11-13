"""Replay seed 8's archived localization and verify bounded view recovery; no API."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'src'))
import argparse,json,tempfile
from copy import deepcopy
import numpy as np
from replay_source import replay_environment


def validate(episode,out):
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    rows=[json.loads(l) for l in (episode/'trajectory.jsonl').read_text().splitlines()]
    original=next(x for x in events if x['kind']=='observation' and x['frame_id']=='f0000852')
    response=next(x for x in events if x['kind']=='response' and x['call']==16)
    tool=response['message']['tool_calls'][0]['function']
    args=json.loads(tool['arguments']);receipt_frame=873
    env,source=replay_environment(episode,json.loads((episode/'run-config.json').read_text()),json.loads((episode/'report.json').read_text()))
    sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'src'))
    from vista import Session
    from perception_recovery import out_of_view_after_recovery
    error=0.;trace=[]
    def replay(until):
        nonlocal error
        for row in rows[env.frame:until]:
            drift=max(float(np.max(abs(env.state()-row['state']))),float(np.max(abs(env.base_pose()-row['base_pose']))))
            error=max(error,drift);assert drift<1e-6
            env.drive(*row['base_command']);env.step(row.get('requested_action',row['action']))
    try:
        replay(852)
        with tempfile.TemporaryDirectory() as tmp:
            class Recorder:
                out=Path(tmp)
                def tick(self,q,phase):env.step(q);trace.append(phase)
            s=Session(env,Recorder(),None,centered_grasp=True,visual_proposals=True)
            s.calls=16;s.station_attempts=original['station_search']['perception_calls'];s.search_steps=1
            old=original['station_search']['local_selection'];local=s.local_targets
            local.surveyed=True;local.candidates=deepcopy(old['candidates']);local.coverage=old['coverage']
            local.selection_blocked_reason=old['selection_blocked_reason'];local.observation_adjustments=old['observation_adjustments']
            frame,state,_=s.observe()
            replay(receipt_frame)
            result,_=s.dispatch(tool['name'],args,frame,state,(receipt_frame-852)*.05)
            assert not result['target_confirmed'] and result['maturity_check']['passed']
            assert result['cross_view_checks']['wrist']['reason']=='outside_image'
            replay(912);s.calls=18
            frame,state,_=s.observe();views=deepcopy(s.stalk_view_evidence)
            assert out_of_view_after_recovery(s,frame)
            before=deepcopy(local.status());base=env.base_pose().copy()
            moved,_=s.dispatch('search_row',{'basis_frame_id':frame,'direction':1},frame,state,0)
            assert moved['status']=='executed',moved
            assert moved['candidate_transition']['failures_before']==moved['candidate_transition']['failures_after']==1
            assert not s.local_targets.surveyed and s.locked_target is None and not s.target_confirmed
            delta=env.base_pose()-base
            assert .17<delta[1]<.185 and abs(delta[0])<.005
            assert env.attached.all() and not any(w.number for w in env.data.warning)
            output={'episode':str(episode),'physics_source':source,'prefix_error':error,'localization':result,
                    'fresh_views':views,'before':before,'search':moved,'delta_m':delta.tolist(),
                    'all_attached_offline':bool(env.attached.all()),'model_calls':0,'online_success_validated':False}
            out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'prefix_error':error,'delta_m':delta.tolist(),'status':moved['status']}))
    finally:env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();validate(a.episode.resolve(),a.out.resolve())
