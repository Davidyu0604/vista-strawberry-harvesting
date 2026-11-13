"""Replay recorded physics then audit delivery protection; no API calls or media."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from replay_source import replay_environment


def validate(episode,out):
    config=json.loads((episode/'run-config.json').read_text())
    report=json.loads((episode/'report.json').read_text())
    rows=[json.loads(line) for line in (episode/'trajectory.jsonl').read_text().splitlines()]
    first=next(i for i,row in enumerate(rows) if row['phase']=='delivery_carry')
    # Session.run adds ten settle ticks and delivery adds ten contact-hold ticks.
    prefix=first-20
    if prefix<0 or any(row['phase'] not in ('model_wait','decision_settle','grasp_contact_check')
                       for row in rows[prefix:first]):
        raise ValueError('Expected stationary pre-delivery prefix')
    env,source=replay_environment(episode,config,report)
    sys.path.insert(0,str(PROJECT_ROOT / 'src'))
    from vista import Session,MotionSafetyStop,robot_state
    from peduncle import BEND_THRESHOLD,EXTENSION_THRESHOLD,TENSION_THRESHOLD
    events=[];trace=[];metrics=[];max_error=0.;model_calls=[]
    class Recorder:
        def __init__(self):self.out=out.parent
        def tick(self,q,phase):
            env.step(q)
            trace.append({'frame':env.frame,'phase':phase,
                'bilateral_stalk_ids_offline':np.flatnonzero(env.bilateral_contacts()).tolist(),
                'detached_ids_offline':np.flatnonzero(~env.attached).tolist()})
    def client_call(messages):
        model_calls.append('scripted_archived_delivery_tool')
        return {'role':'assistant','tool_calls':[{'id':'offline','function':{
            'name':'deliver_to_basket','arguments':json.dumps({'basis_frame_id':'offline'})}}]},{}
    try:
        for row in rows[:prefix]:
            error=max(float(np.max(abs(env.state()-row['state']))),
                      float(np.max(abs(env.base_pose()-row['base_pose']))),
                      float(np.linalg.norm(env.data.site_xpos[env.tcp]-row['tcp_position'])))
            max_error=max(max_error,error)
            if error>1e-6:raise ValueError('Original prefix differs')
            env.drive(*row['base_command']);env.step(row.get('requested_action',row['action']))
            if row['phase']=='bend_pull':
                # Read-only acceptance data; never passed to Session/control.
                ids=np.flatnonzero(env.bilateral_contacts()).tolist()
                metrics.append({'frame':env.frame,'metrics':{str(i):dict(env.stalk_metrics[i]) for i in ids}})
        session=Session(env,Recorder(),SimpleNamespace(call=client_call),centered_grasp=True)
        session.grasp_attempted=True;session.target_stage='grasp'
        session.centered.carry_roll=float(env.last_action[4])
        session.observe=lambda:('offline',robot_state(env),{'role':'user','content':[]})
        session.log=lambda kind,**data:events.append({'kind':kind,**data})
        status='unexpected_return';failure=None
        try:session.run()
        except MotionSafetyStop as exc:status='motion_stopped';failure=str(exc)
        output={'episode':str(episode),'physics_source':source,'max_prefix_state_error':max_error,
                'status':status,'error':failure,'motion_stop_trace':session.motion_stop_trace,
                'final_frame':env.frame,'original_first_failure_frame':rows[first+9]['next_frame'],
                'network_model_calls':0,'scripted_tool_responses':len(model_calls),'new_media':0,
                'detachment_thresholds':{'bend_rad':BEND_THRESHOLD,'extension_m':EXTENSION_THRESHOLD,
                                         'elastic_force_n':TENSION_THRESHOLD,'tcp_pull_m':.012},
                'pull_metrics_offline':metrics,'trace':trace,'events':events,
                'detached_final_offline':np.flatnonzero(~env.attached).tolist(),
                'warnings':[int(w.number) for w in env.data.warning]}
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps({k:v for k,v in output.items() if k not in ('events','trace','pull_metrics_offline')},ensure_ascii=False),flush=True)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();validate(args.episode.resolve(),args.out.resolve())
