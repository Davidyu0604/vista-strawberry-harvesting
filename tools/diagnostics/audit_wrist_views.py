"""Offline camera-mount comparison on recorded visual targets in the full scene.

Replays original controls, checks robot-state fidelity, then temporarily changes
only camera extrinsics. No policy calls, fruit poses/IDs, or training exports.
Run one simulation process at a time (scene generation shares texture assets).
"""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw
from replay_source import replay_environment


VARIANTS = {
    'original': ([0,-.065,.015],[0,0,-.11],65),
    'angle-only': ([0,-.065,.015],[-.055,.045,-.10],65),
    'angle-wide': ([0,-.065,.015],[-.055,.045,-.10],80),
    'side-offset': ([-.04,-.085,.015],[-.055,.045,-.10],65),
    'side-wide': ([-.04,-.085,.015],[-.055,.045,-.10],80),
    'workspace-angle': ([-.04,-.085,.015],[-.10,.075,-.08],80),
}


def set_camera(env, spec):
    pos,target,fovy=spec
    z=np.asarray(pos)-target;z/=np.linalg.norm(z)
    x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    cid=env.model.camera('wrist').id
    env.model.cam_pos[cid]=pos
    mujoco.mju_mat2Quat(env.model.cam_quat[cid],np.column_stack([-x,-y,z]).ravel())
    env.model.cam_fovy[cid]=fovy
    mujoco.mj_camlight(env.model,env.data)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();episode=a.episode.resolve();a.out.mkdir(parents=True,exist_ok=True)
    config=json.loads((episode/'run-config.json').read_text())
    report=json.loads((episode/'report.json').read_text())
    targets={}
    for line in (episode/'vista-events.jsonl').read_text().splitlines():
        event=json.loads(line);r=event.get('result')
        if event['kind']=='receipt' and isinstance(r,dict) and r.get('cross_view_checks') and r.get('locked_target_m'):
            targets.setdefault(int(r['frame_id'][1:]),[]).append((event['call'],r['locked_target_m'],r['cross_view_checks'].get('wrist')))
    if not targets:raise ValueError('No recorded visual cross-view targets')
    env,source=replay_environment(episode,config,report)
    from vista import verify_projected_point
    rows=[json.loads(l) for l in (episode/'trajectory.jsonl').read_text().splitlines()]
    results=[];max_error=0.;cid=env.model.camera('wrist').id
    saved=(env.model.cam_pos[cid].copy(),env.model.cam_quat[cid].copy(),env.model.cam_fovy[cid])
    try:
        for row in rows:
            if row['frame']>max(targets):break
            error=max(float(np.max(np.abs(env.state()-row['state']))),
                      float(np.max(np.abs(env.base_pose()-row['base_pose']))),
                      float(np.linalg.norm(env.data.site_xpos[env.tcp]-row['tcp_position'])))
            max_error=max(error,max_error)
            if error>1e-6:raise RuntimeError(f'Replay differs at frame {row["frame"]}: {error}')
            for call,target,old_check in targets.get(row['frame'],[]):
                panel=Image.new('RGB',(480*3,390*2));checks={}
                for i,(name,spec) in enumerate(VARIANTS.items()):
                    set_camera(env,spec)
                    depth,calibration=env.depth_view('wrist',480,360)
                    check=verify_projected_point(np.asarray(target),depth,calibration)
                    tcp_check=verify_projected_point(env.data.site_xpos[env.tcp],depth,calibration)
                    checks[name]={'target_check':check,'tcp_in_frame':tcp_check.get('reason') not in ('outside_image','behind_camera')}
                    rgb=Image.fromarray(env.render('wrist',480,360));draw=ImageDraw.Draw(rgb)
                    if 'projected_pixel' in check:
                        x,y=check['projected_pixel'];draw.ellipse((x-6,y-6,x+6,y+6),outline='cyan',width=2)
                    panel.paste(rgb,((i%3)*480,(i//3)*390))
                    ImageDraw.Draw(panel).text(((i%3)*480+5,(i//3)*390+365),
                        f'{name}: {check.get("consistent")} {check.get("reason",round(check.get("surface_error_m",0),4))}',fill='white')
                panel.save(a.out/f'call-{call}-views.jpg')
                results.append({'call':call,'frame':row['frame'],'visual_target_m':target,
                                'recorded_wrist_check':old_check,'variants':checks})
            env.model.cam_pos[cid]=saved[0];env.model.cam_quat[cid]=saved[1];env.model.cam_fovy[cid]=saved[2]
            mujoco.mj_camlight(env.model,env.data)
            env.drive(*row['base_command']);env.step(row.get('requested_action',row['action']))
        summary={'source_episode':str(episode),'physics_source':source,'max_robot_state_error':max_error,
                 'model_calls':0,'camera_variants':VARIANTS,'checks':results,
                 'consistent_counts':{n:sum(r['variants'][n]['target_check']['consistent'] for r in results) for n in VARIANTS},
                 'limitation':'Archived measured targets, repeated points not independent. Camera audit only, not online harvest success; no scene truth used to aim cameras.'}
        (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({'seed':config['seed'],'checks':len(results),'max_robot_state_error':max_error,'counts':summary['consistent_counts']}),flush=True)
    finally:env.close()


if __name__=='__main__':main()
