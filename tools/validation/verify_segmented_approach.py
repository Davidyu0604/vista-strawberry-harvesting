"""Replay recorded controls up to pregrasp; test new approach without model calls."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse,json,hashlib
from pathlib import Path
import numpy as np
from peduncle import PeduncleEnv
from vista import Session


def verify(episode,out,legacy=False):
    cfg=json.loads((episode/'run-config.json').read_text())
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    target=next(r['result']['locked_target_m'] for r in events if r['kind']=='receipt' and isinstance(r.get('result'),dict) and r['result'].get('target_confirmed'))
    e=PeduncleEnv(seed=cfg['seed'],single=False,render_quality=cfg['quality'])
    rows=[json.loads(l) for l in (episode/'trajectory.jsonl').read_text().splitlines()]
    result={'controller_sha256':hashlib.sha256(project_source('vista.py').read_bytes()).hexdigest(),'legacy_approach':legacy,'source':str(episode),'max_replay_joint_error_rad':0.,'target_m':target}
    try:
        for row_index,row in enumerate(rows):
            result['max_replay_joint_error_rad']=max(result['max_replay_joint_error_rad'],float(np.max(np.abs(e.state()-row['state']))))
            if row['phase']=='visual_grasp':break
            e.drive(*row['base_command']);e.step(row['requested_action'])
        else:raise ValueError('No grasp approach in source')
        print(episode.name,'checkpoint',e.frame,'drift',result['max_replay_joint_error_rad'],flush=True)
        s=Session.__new__(Session);s.e=e;s.calls=0;s.max_calls=32
        s.locked_target=np.array(target);s.target_stage='pregrasp';s.grasp_attempted=False
        s.target_confirmed=True;s.maturity_evidence=None;s.approach_pitch=0
        pad=e.model.geom('peduncle_rubber_insert').id
        max_pen=0.;phases=[]
        def tick(q,phase):
            nonlocal max_pen
            e.step(q);phases.append(phase)
            if phase.startswith('visual_'):
                for c in e.data.contact:
                    pair=[int(c.geom1),int(c.geom2)]
                    if pad in pair:
                        g=pair[1] if pair[0]==pad else pair[0]
                        name=e.model.body(int(e.model.geom_bodyid[g])).name
                        if name.startswith(('stem_','branch_','leaf_')):max_pen=max(max_pen,float(-c.dist))
        s.tick=tick
        try:
            result['pregrasp']='Replayed exactly, including original observation/model-wait ticks'
            if legacy:e.meta['tilted_stalk_prior']=False
            result['grasp']=s.execute_approach('grasp')
            # Preserve ALL recorded waits and subsequent close/pull commands.
            # The source successful demonstration needed multiple pulls; a new
            # one-pull script is not a valid regression oracle for that episode.
            tail=row_index
            while tail<len(rows) and rows[tail]['phase']=='visual_grasp':tail+=1
            seen=set();last_phase=None
            for row in rows[tail:]:
                if row['phase'].startswith('delivery_') or row['phase']=='approach_retreat':break
                if row['phase']=='bend_pull' and last_phase!='bend_pull':
                    contact=bool(e.bilateral_contacts().any())
                    result.setdefault('pull_start_checks',[]).append(contact)
                    if not contact:raise ValueError('Recorded pull would be rejected: no bilateral contact')
                e.drive(*row['base_command']);tick(row['requested_action'],row['phase'])
                last_phase=row['phase']
                seen.update(np.flatnonzero(e.bilateral_contacts()).tolist())
            result['bilateral_fruit_ids']=sorted(seen)
            result['detached_and_held_ids']=[i for i in seen if not e.attached[i] and e.contacts(i)]
            result['status']='completed'
        except ValueError as exc:
            result.update(status='rejected',reason=str(exc),approach_trace=s.last_approach_trace,
                          target_cleared=s.locked_target is None,gripper_open=float(e.state()[5])>.9)
        result['max_pad_vegetation_penetration_m']=max_pen
        result['control_ticks']=len(phases)
        result['physics_warnings']=bool(any(w.number for w in e.data.warning))
        out.mkdir(parents=True,exist_ok=True)
        (out/(episode.name+'.json')).write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ('pregrasp','grasp','approach_trace','close','pull')},indent=2),flush=True)
    finally:e.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episodes',type=Path,nargs='+');p.add_argument('--out',type=Path,required=True);p.add_argument('--legacy',action='store_true');a=p.parse_args()
    for episode in a.episodes:verify(episode,a.out,a.legacy)
