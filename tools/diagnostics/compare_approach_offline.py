"""Controlled offline geometry/pose screening from a recorded pregrasp state.

Uses logged RGB-D target. Privileged contact/detachment data are evaluation only.
Production scene and control settings are not modified by this experiment.
"""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse,copy,json
from collections import Counter
from pathlib import Path
import mujoco
import numpy as np
from peduncle import PeduncleEnv

FIELDS='attached contact_ticks grasp_origins events frame in_box_ticks last_action drive_command park_pose pull_directions stalk_metrics worst_obstacle_penetration worst_obstacle_pair worst_vegetation_penetration'.split()


def compare(episode,out):
    cfg=json.loads((episode/'run-config.json').read_text())
    rows=[json.loads(l) for l in (episode/'trajectory.jsonl').read_text().splitlines()]
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    target=np.array(next(r['result']['locked_target_m'] for r in events if r['kind']=='receipt' and r.get('call')==23))
    e=PeduncleEnv(seed=cfg['seed'],single=False,render_quality=cfg['quality'])
    out.mkdir(parents=True,exist_ok=True)
    try:
        drift=0.
        for row in rows:
            drift=max(drift,float(np.max(np.abs(e.state()-row['state']))))
            if row['phase']=='visual_pregrasp':break
            e.drive(*row['base_command']);e.step(row['requested_action'])
        snapshot=mujoco.MjData(e.model);mujoco.mj_copyData(snapshot,e.model,e.data)
        saved={k:copy.deepcopy(getattr(e,k)) for k in FIELDS}
        gid=e.model.geom('peduncle_rubber_insert').id
        original=e.model.geom_size[gid].copy()
        print(json.dumps({'checkpoint_frame':e.frame,'max_replay_joint_error_rad':drift}),flush=True)
        results=[]
        variants=[('baseline',0,.014),('pitch_down_10',-10,.014),('pitch_up_10',10,.014),('pad_height_20mm',0,.010),('pad_height_16mm',0,.008)]
        for name,degrees,halfheight in variants:
            mujoco.mj_copyData(e.data,e.model,snapshot)
            for k,v in saved.items():setattr(e,k,copy.deepcopy(v))
            e.model.geom_size[gid]=original;e.model.geom_size[gid,1]=halfheight
            mujoco.mj_forward(e.model,e.data)
            pitch=np.deg2rad(degrees);pairs=Counter();max_pen=0.;phases=[]
            def tick(q,n,phase):
                nonlocal max_pen
                for _ in range(n):
                    e.step(q)
                    if phase in ('pregrasp','grasp'):
                        for c in e.data.contact:
                            gs=[int(c.geom1),int(c.geom2)]
                            if gid in gs:
                                other=gs[1] if gs[0]==gid else gs[0]
                                body=e.model.body(int(e.model.geom_bodyid[other])).name
                                pairs[body+':'+(e.model.geom(other).name or str(other))]+=1
                                if body.startswith(('stem_','branch_','leaf_')):max_pen=max(max_pen,float(-c.dist))
                phases.append(phase)
            result={'variant':name,'pitch_deg':degrees,'pad_full_height_m':2*halfheight,
                    'target_m':target.tolist(),'geometry_only_mass_unchanged':True,'passed':False}
            try:
                for phase,p in [('pregrasp',target-[.055,0,0]),('grasp',target)]:
                    q,ik=e.ik(p,1,approach_pitch=pitch)
                    result[phase+'_ik_error_m']=ik
                    if ik>.006:raise ValueError(phase+' IK unreachable')
                    tick(q,55,phase)
                    error=float(np.linalg.norm(e.data.site_xpos[e.tcp]-p))
                    result[phase+'_tcp_error_m']=error
                    if error>.008:raise ValueError(phase+' tracking blocked')
                q=e.last_action.copy();q[5]=-.15;tick(q,50,'close');tick(q,20,'hold')
                bilateral=np.flatnonzero(e.bilateral_contacts()).tolist();result['bilateral_fruit_ids']=bilateral
                if not bilateral:raise ValueError('No bilateral stalk contact')
                q,ik=e.ik(e.data.site_xpos[e.tcp]+[-.035,0,0],-.15,approach_pitch=np.deg2rad(-37))
                if ik>.006:raise ValueError('Pull IK unreachable')
                tick(q,60,'pull');tick(q,40,'post_pull_hold')
                detached=[i for i in bilateral if not e.attached[i]]
                result['detached_fruit_ids']=detached
                result['held_after_pull_ids']=[i for i in detached if e.contacts(i)]
                result['passed']=bool(result['held_after_pull_ids'])
                if not result['passed']:result['reason']='No detached fruit retained after 2s hold'
            except ValueError as exc:result['reason']=str(exc)
            result.update(contact_samples=dict(pairs),max_pad_vegetation_penetration_m=max_pen,phases=phases)
            results.append(result)
            (out/'comparison.json').write_text(json.dumps({'episode':str(episode),'max_replay_joint_error_rad':drift,'results':results},indent=2))
            print(json.dumps(result),flush=True)
        e.model.geom_size[gid]=original
    finally:e.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    compare(a.episode,a.out)
