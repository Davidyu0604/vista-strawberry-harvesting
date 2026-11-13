"""Audit blended pull and mechanical delivery from archived pre-pull contact hold; no model."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from replay_source import replay_environment


def validate(episode,out,preview_dir=None,pull_only=False):
    config=json.loads((episode/'run-config.json').read_text());report=json.loads((episode/'report.json').read_text())
    rows=[json.loads(line) for line in (episode/'trajectory.jsonl').read_text().splitlines()]
    events=[json.loads(line) for line in (episode/'vista-events.jsonl').read_text().splitlines()]
    prefix=next(i for i,row in enumerate(rows) if row['phase']=='grasp_contact_check')
    before=next(row for row in rows if row['phase']=='visual_pregrasp')
    lock=next(e['result'] for e in events if e['kind']=='receipt' and isinstance(e.get('result'),dict)
              and e['result'].get('target_confirmed') and e['result'].get('locked_target_m'))
    env,source=replay_environment(episode,config,report)
    sys.path.insert(0,str(PROJECT_ROOT / 'src'))
    from vista import Session,evaluate,jaw_contact_state
    trace=[];max_error=0.;phase=None;preview_count=0
    def preview(label):
        nonlocal preview_count
        if preview_dir is None:return
        from PIL import Image
        preview_dir.mkdir(parents=True,exist_ok=True)
        for camera in ('harvest_close','wrist'):
            Image.fromarray(env.render(camera,480,360)).save(preview_dir/f'{label}-{camera}.png')
            preview_count+=1
    class Recorder:
        def __init__(self):self.out=out.parent
        def tick(self,q,label):
            nonlocal phase
            env.step(q);phase=label
            trace.append({'frame':env.frame,'phase':label,'base_pose':env.base_pose().tolist(),
                'stalk_bilateral_ids_offline':np.flatnonzero(env.bilateral_contacts()).tolist(),
                'detached_ids_offline':np.flatnonzero(~env.attached).tolist(),
                'external_contact':jaw_contact_state(env)})
    try:
        for row in rows[:prefix]:
            error=max(float(np.max(abs(env.state()-row['state']))),float(np.max(abs(env.base_pose()-row['base_pose']))))
            max_error=max(error,max_error)
            if error>1e-6:raise ValueError('Original prefix differs')
            env.drive(*row['base_command']);env.step(row.get('requested_action',row['action']))
        # Offline target association never passed into controller.
        target=int(np.argmin(np.linalg.norm(env.data.site_xpos[env.grasp_sites]-lock['locked_target_m'],axis=1)))
        s=Session(env,Recorder(),None,centered_grasp=True);s.grasp_attempted=True;s.target_stage='grasp'
        s.locked_target=np.asarray(lock['locked_target_m']);s.maturity_evidence=lock['maturity_check']
        s.centered.carry_roll=float(env.last_action[4]);s.centered.observation_roll=float(before['state'][4])
        env.meta['observation_roll_rad']=s.centered.observation_roll
        result={};failure=None
        try:
            preview('before')
            result['pull']=s.execute([{'action':'bend_pull','dx':-.035,'dy':0,'dz':0,'pitch_deg':-37,'seconds':3}])
            if result['pull'][0]['status']!='executed':raise ValueError(str(result['pull']))
            for _ in range(20):s.tick(env.last_action,'decision_settle')
            preview('after')
            # Mechanical continuation only. This does not validate model visual review.
            if not pull_only:result['delivery']=s.deliver_to_basket()
        except (ValueError,RuntimeError) as exc:failure=str(exc)
        evaluation=evaluate(env,trace)
        until_release=[]
        for row in trace:
            if row['phase']=='delivery_release':break
            until_release.append(row)
        output={'episode':str(episode),'physics_source':source,'prefix_error':max_error,
                'failure':failure,'last_phase':phase,'receipts':result,'evaluation':evaluation,
                'associated_target_offline':target,'same_target_in_basket':evaluation['ripe_fruit_in_basket']==[target],
                'continuous_same_stalk_contact_until_release':bool(until_release and all(target in r['stalk_bilateral_ids_offline'] for r in until_release)),
                'warnings':[int(w.number) for w in env.data.warning],'trace':trace,
                'model_calls':0,'new_videos':0,'temporary_preview_images':preview_count,'visual_detachment_review_validated':False,'pull_only':pull_only,
                'detached_target_only':np.flatnonzero(~env.attached).tolist()==[target],
                'target_still_gripped':bool(env.bilateral_contacts()[target])}
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps({k:v for k,v in output.items() if k not in ('trace','receipts')},ensure_ascii=False),flush=True)
    finally:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('episode',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--preview-dir',type=Path)
    parser.add_argument('--pull-only',action='store_true')
    args=parser.parse_args();validate(args.episode.resolve(),args.out.resolve(),args.preview_dir,args.pull_only)
