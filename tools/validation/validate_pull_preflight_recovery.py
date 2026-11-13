"""Offline continuation of a stopped recording using current bounded pull planning.

Replays all committed steps, repeats the tactile hold, then executes pull and
scripted mechanical delivery. Does not call a model or certify visual separation.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def continue_from_recorded_state(env,episode,output,prefix_error):
    from vista import Session,evaluate,jaw_contact_state
    from export_act_dataset import check_harvest
    report=json.loads((episode/'report.json').read_text())
    events=[json.loads(x) for x in (episode/'vista-events.jsonl').read_text().splitlines()]
    request=next(e for e in reversed(events) if e['kind']=='response')
    action=json.loads(request['message']['tool_calls'][0]['function']['arguments'])['actions'][0]
    trace=[];max_tracking=0.
    class Recorder:
        out=output.parent
        def tick(self,q,phase):
            env.step(q)
            trace.append(dict(frame=env.frame,phase=phase,base_pose=env.base_pose().tolist(),
                              base_command=env.drive_command.tolist(),
                              contacts_offline=np.flatnonzero(env.bilateral_contacts()).tolist(),
                              detached_offline=np.flatnonzero(~env.attached).tolist()))
    session=Session(env,Recorder(),None,centered_grasp=True,visual_proposals=True)
    session.grasp_attempted=True;session.target_stage='grasp'
    session.centered.carry_roll=float(env.last_action[4])
    session.centered.observation_roll=float(report['observation_roll_rad'])
    env.meta['observation_roll_rad']=session.centered.observation_roll
    check=session.centered.check_tracking
    def tracking(expected):
        nonlocal max_tracking
        value=check(expected);max_tracking=max(max_tracking,value);return value
    session.centered.check_tracking=tracking
    result=dict(episode=str(episode),prefix_frames=env.frame,prefix_error=prefix_error,
                model_calls=0,visual_detachment_review_validated=False,
                note='Exact committed prefix, then repeated 10-tick tactile hold. Scripted mechanical delivery; no online model or training approval.')
    try:
        result['pull']=session.execute([action])
        assert result['pull'][0]['status']=='executed',result['pull']
        assert not session.motion_stop_trace
        assert jaw_contact_state(env)['both']
        result['delivery']=session.deliver_to_basket()
        result['evaluation']=evaluate(env,trace)
        check_harvest(dict(result['evaluation'],basket_mount='chassis',events=env.events))
        result['independent_harvest_check_passed']=True
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        result['independent_harvest_check_passed']=False
    result.update(max_tracking_error_m=max_tracking,warnings=[int(w.number) for w in env.data.warning],
                  trace=trace,source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                         for n in ('centered_grasp.py','vista.py','collect_fresh50.py')})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('prefix_frames','prefix_error','independent_harvest_check_passed','max_tracking_error_m','warnings')}),flush=True)
    return result


def main():
    import argparse
    from replay_source import replay_environment
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();episode=args.episode.resolve()
    cfg=json.loads((episode/'run-config.json').read_text());report=json.loads((episode/'report.json').read_text())
    env,_=replay_environment(episode,cfg,report)
    error=0.
    try:
        for row in map(json.loads,(episode/'trajectory.jsonl').read_text().splitlines()):
            diff=max(float(np.max(abs(env.state()-row['state']))),float(np.max(abs(env.base_pose()-row['base_pose']))))
            error=max(error,diff);assert diff<1e-9
            env.drive(*row['base_command']);env.step(row['requested_action'])
        continue_from_recorded_state(env,episode,args.out.resolve(),error)
    finally:env.close()


if __name__=='__main__':main()
