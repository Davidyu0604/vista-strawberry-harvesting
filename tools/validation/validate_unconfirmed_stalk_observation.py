"""Replay seed 5 before early finish, then test an explicit identity-review recovery."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'src'))
import argparse,json,tempfile
from copy import deepcopy
import numpy as np
from replay_source import replay_environment


def validate(episode,out):
    cfg=json.loads((episode/'run-config.json').read_text());report=json.loads((episode/'report.json').read_text())
    events=[json.loads(line) for line in (episode/'vista-events.jsonl').read_text().splitlines()]
    rows=[json.loads(line) for line in (episode/'trajectory.jsonl').read_text().splitlines()]
    def tool(call):
        e=next(e for e in events if e['kind']=='response' and e['call']==call)
        return json.loads(e['message']['tool_calls'][0]['function']['arguments'])
    old=tool(1);finish=tool(4)
    original=next(e for e in events if e['kind']=='observation' and e['frame_id']=='f0000094')
    env,source=replay_environment(episode,cfg,report)
    sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'src'))
    from vista import Session
    from perception_recovery import unconfirmed_stalk_options
    error=0.;trace=[]
    try:
        for row in rows[:94]:
            drift=max(float(np.max(abs(env.state()-row['state']))),float(np.max(abs(env.base_pose()-row['base_pose']))))
            error=max(error,drift);assert drift<1e-9
            env.drive(*row['base_command']);env.step(row.get('requested_action',row['action']))
        with tempfile.TemporaryDirectory() as tmp:
            class Recorder:
                out=Path(tmp)
                def tick(self,q,phase):env.step(q);trace.append({'frame':env.frame,'phase':phase})
            s=Session(env,Recorder(),None,centered_grasp=True,visual_proposals=True)
            s.calls=4;s.station_attempts=3;local=s.local_targets
            local.surveyed=True;local.candidates=deepcopy(original['local_selection']['candidates'])
            local.coverage=old['coverage'];local.survey_review=deepcopy(old)
            local.survey_batch=next(e for e in events if e['kind']=='body_proposals' and e['frame_id']==old['frame_id'])
            local.survey_state=next(e for e in events if e['kind']=='observation' and e['frame_id']==old['frame_id'])
            local.survey_reference_depth=episode/'observations'/f"{old['frame_id']}-harvest_close-depth.npz"
            frame,state,_=s.observe();before=deepcopy(s.stalk_view_evidence)
            options=unconfirmed_stalk_options(s,frame);assert options==[{'proposal_id':'W1','camera':'wrist'}],options
            end,_=s.dispatch('finish',finish,frame,state,0)
            assert not end['episode_ended'] and end['remaining_model_calls']==28
            # This request is an offline fixture derived from the recorded
            # model's uncertainty. No new model endorsement is claimed.
            args={'basis_frame_id':frame,'candidate_key':local.active['candidate_key'],
                  'reason':'Offline fixture: original finish could not confirm W1 stalk identity',
                  'unconfirmed_stalk_proposals':[{'proposal_id':'W1','reason':finish['reason'][:600]}]}
            moved,_=s.dispatch('adjust_observation',args,frame,state,0)
            assert moved['status']=='executed' and local.active['failures']==0
            assert not s.target_confirmed and not local.can_search
            after_frame,_,_=s.observe()
            assert after_frame!=frame and not unconfirmed_stalk_options(s,after_frame)
            assert env.attached.all() and not any(w.number for w in env.data.warning)
            result={'source_episode':str(episode),'physics_source':source,'prefix_frames':94,'prefix_error':error,
                    'before_views':before,'finish_result':end,'offline_adjustment_args':args,'adjustment':moved,
                    'after_views':s.stalk_view_evidence,'after_station_status':s.station_status(),
                    'trace':trace,'all_attached_offline':True,'warnings':0,'model_calls':0,'online_success_validated':False}
            out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            print('Original prefix exact; early finish redirected; explicit unconfirmed W1 review executes bounded adjustment; no skip/grasp.',flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();validate(a.episode.resolve(),a.out.resolve())
