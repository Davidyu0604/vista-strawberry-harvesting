"""Offline, hash-verified contact replay. Hidden geometry is diagnostic only."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from replay_source import replay_environment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();episode=a.episode.resolve();a.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((episode/'run-config.json').read_text())
    report=json.loads((episode/'report.json').read_text())
    rows=[json.loads(l) for l in (episode/'trajectory.jsonl').read_text().splitlines()]
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    confirmed=[e['result'] for e in events if e['kind']=='receipt' and isinstance(e.get('result'),dict)
               and e['result'].get('target_confirmed') and e['result'].get('locked_target_m')]
    first_approach=next(r['frame'] for r in rows if r['phase']=='visual_pregrasp')
    # Associate ONCE before approach; do not switch identity as a berry deflects.
    lock=next(r for r in reversed(confirmed) if int(r['frame_id'][1:])<=first_approach)
    target=np.asarray(lock['locked_target_m']);lock_frame=int(lock['frame_id'][1:])
    snapshots=set()
    for phase in ('visual_pregrasp','visual_grasp','gripper','terminal_settle'):
        frames=[r['frame'] for r in rows if r['phase']==phase]
        if frames:snapshots.update((frames[0],frames[-1]))
    env,source=replay_environment(episode,cfg,report)
    max_error=0.;samples=[];pairs=Counter();association=None;initial_site=None
    try:
        for row in rows:
            error=max(float(np.max(np.abs(env.state()-row['state']))),
                      float(np.max(np.abs(env.base_pose()-row['base_pose']))),
                      float(np.linalg.norm(env.data.site_xpos[env.tcp]-row['tcp_position'])))
            max_error=max(error,max_error)
            if error>1e-6:raise RuntimeError(f'Physics replay differs at frame {row["frame"]}: {error}')
            if row['frame']==lock_frame:
                distances=np.linalg.norm(env.data.site_xpos[env.grasp_sites]-target,axis=1)
                fruit=int(np.argmin(distances))
                if distances[fruit]>.015:raise ValueError('Visual lock has no nearby physical stalk; no diagnostic identity assigned')
                association={'fruit_id':fruit,'distance_to_grasp_site_m':float(distances[fruit]),
                             'frame':lock_frame,'visual_lock_m':target.tolist()}
                initial_site=env.data.site_xpos[env.grasp_sites[fruit]].copy()
            if association and row['frame']>=first_approach:
                contacts=[];touches={'fixed_stalk':False,'moving_stalk':False}
                robot_root=env.model.body_rootid[env.grip_body]
                for c in env.data.contact:
                    if c.dist>0:continue
                    geoms=list(map(int,c.geom));bodies=[int(env.model.geom_bodyid[g]) for g in geoms]
                    if not any(b in (env.grip_body,env.jaw_body) for b in bodies):continue
                    if all(env.model.body_rootid[b]==robot_root for b in bodies):continue
                    names=[env.model.geom(g).name or f'geom_{g}' for g in geoms]
                    contacts.append({'geoms':names,'bodies':[env.model.body(b).name for b in bodies],
                                     'penetration_mm':float(-c.dist*1000)})
                    pairs[' / '.join(names)]+=1
                    if int(env.stalk_geoms[fruit]) in geoms:
                        touches['fixed_stalk'] |= env.grip_body in bodies
                        touches['moving_stalk'] |= env.jaw_body in bodies
                rotation=env.data.xmat[env.fruit_bodies[fruit]].reshape(3,3)
                bottom=env.fruit_pos(fruit)+rotation@np.array([0,0,.022*env.meta['fruit_scales'][fruit]])
                local_target=rotation.T@(target-bottom)
                sample={'frame':row['frame'],'phase':row['phase'],
                        'tcp_to_visual_lock_mm':float(np.linalg.norm(env.data.site_xpos[env.tcp]-target)*1000),
                        'visual_lock_in_actual_stalk_frame_mm':(local_target*1000).tolist(),
                        'visual_lock_radial_error_mm':float(np.linalg.norm(local_target[:2])*1000),
                        'actual_stalk_site_displacement_mm':float(np.linalg.norm(env.data.site_xpos[env.grasp_sites[fruit]]-initial_site)*1000),
                        'gripper_joint_rad':float(env.state()[5]),'target_stalk_contacts':touches,'contacts':contacts}
                if row['frame'] in snapshots:
                    gr=env.data.xmat[env.grip_body].reshape(3,3)
                    sample['stalk_bottom_in_gripper_mm']=(gr.T@(bottom-env.data.xpos[env.grip_body])*1000).tolist()
                    sample['stalk_tip_in_gripper_mm']=(gr.T@(bottom+rotation[:,2]*.036-env.data.xpos[env.grip_body])*1000).tolist()
                    insert=env.model.geom('peduncle_rubber_insert')
                    sample['fixed_insert_local_bounds_mm']=[((insert.pos-insert.size)*1000).tolist(),((insert.pos+insert.size)*1000).tolist()]
                    sample['snapshot']=True
                samples.append(sample)
            env.drive(*row['base_command']);env.step(row.get('requested_action',row['action']))
        result={'source_episode':str(episode),'physics_source':source,'model_calls':0,
                'max_robot_replay_error':max_error,'offline_stalk_association':association,
                'contact_frame_counts':dict(pairs),'samples':samples,
                'bilateral_target_stalk_frames':sum(all(s['target_stalk_contacts'].values()) for s in samples),
                'note':'Hidden object geometry used only for this offline audit; contacts alone do not imply failure.'}
        (a.out/'contacts.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
        for sample in samples:
            if sample.get('snapshot'):print(json.dumps(sample))
    finally:env.close()


if __name__=='__main__':main()
