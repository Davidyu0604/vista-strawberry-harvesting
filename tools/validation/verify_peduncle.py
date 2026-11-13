"""Single-fruit physical regression. Privileged fixture, never a VISTA demo."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse
import os
import sys
if '--viewer' in sys.argv:
    os.environ['MUJOCO_GL']='glfw'
    os.environ['PYOPENGL_PLATFORM']='glx'
import json
import time
import numpy as np
from PIL import Image
from pathlib import Path
from peduncle import PeduncleEnv
from run import Recorder
from scene import ROOT


def trial(seed, out, video=True, show=False):
    e=PeduncleEnv(seed,render_quality='fast')
    rec=Recorder(e,out,images=False,video=False)
    viewer=None
    if show:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(e.model,e.data)
        viewer.opt.geomgroup[3:5]=0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0
        viewer.cam.lookat[:]=[.15,.15,.23];viewer.cam.distance=1.1;viewer.cam.azimuth=145;viewer.cam.elevation=-25
    frames=[];checks={};phase_records=[]
    def tick(q,phase):
        start=time.monotonic()
        if video and e.frame%5==0:
            camera='harvest_close' if phase in ('idle','pregrasp','approach','close','bend_pull','lift') else 'detail'
            frames.append(Image.fromarray(e.render(camera,640,480)))
        if viewer:
            if not viewer.is_running():raise RuntimeError('Viewer closed')
            with viewer.lock():rec.tick(q,phase)
            viewer.sync();time.sleep(max(0,e.dt-(time.monotonic()-start)))
        else:rec.tick(q,phase)
    def move(p,grip,n,phase,pitch=0):
        q,error=e.ik(np.asarray(p),grip,approach_pitch=pitch)
        if error>.006:raise RuntimeError(f'IK {phase}: {error}')
        for _ in range(n):tick(q,phase)
        phase_records.append({'phase':phase,'attached':bool(e.attached[0]),'stalk_contact':e.contacts(0),
                              'metrics':dict(e.stalk_metrics.get(0,{}))})
    def drive_to(y,phase):
        q=e.last_action.copy()
        for _ in range(240):
            error=y-e.base_pose()[1]
            if abs(error)<.004:break
            e.drive(float(np.clip(1.2*error,-.06,.06)),0);tick(q,phase)
        else:raise RuntimeError('Drive timeout')
        e.drive(0,0)
        for _ in range(30):tick(q,phase+'_park')
    error=None
    try:
        for _ in range(40):tick(e.last_action,'idle')
        assert e.attached[0] and e.stalk_metrics[0]['extension_m']<.001
        checks['no_spontaneous_detachment']=True
        # Ground truth used solely to place this controller test at the known stalk.
        p=e.data.site_xpos[e.grasp_sites[0]].copy()
        move(p-[.055,0,0],1,55,'pregrasp')
        move(p,1,55,'approach')
        move(p,-.15,55,'close')
        assert e.contacts(0) and e.attached[0],phase_records[-1]
        checks['stalk_grasp_without_auto_detach']=True
        move(p-[.035,0,0],-.15,65,'bend_pull',-.65)
        assert not e.attached[0] and e.contacts(0),phase_records[-1]
        checks['bend_pull_detached_and_held']=True
        move(p-[.07,0,-.04],-.15,55,'lift',-.35)
        assert e.contacts(0),'Lost stalk during lift'
        checks['lift_held']=True
        drive_to(e.meta['collection_base_y'],'return_to_collection')
        basket=e.basket_position()
        move(basket+[0,0,.26],-.15,90,'transport')
        move(basket+[0,0,.14],-.15,55,'basket_approach')
        move(basket+[0,0,.115],-.15,40,'lower')
        move(basket+[0,0,.115],1,45,'release')
        move(basket+[0,0,.23],1,60,'clearance')
        for _ in range(30):tick(e.last_action,'settle')
        assert e.success,'Fruit did not settle in basket'
        checks['released_in_ground_basket']=True
        move([.19,e.base_pose()[1]-.04,.34],1,65,'stow')
        drive_to(0,'return_to_harvest')
        assert abs(e.base_pose()[1])<.012
        checks['returned_to_harvest']=True
    except Exception as exc:
        error=str(exc)
        raise
    finally:
        try:
            if viewer:viewer.close()
            e.drive(0,0)
            report=rec.finish()
            report.update(source='privileged single-fruit peduncle physics fixture; not model-directed',
                          checks=checks,phase_records=phase_records,error=error,
                          warnings=sum(w.number for w in e.data.warning),
                          physics_calibrated=False,gripper_insert=True)
            (out/'report.json').write_text(json.dumps(report,indent=2))
            (out/'scene.xml').write_text(e.xml)
            if frames:frames[0].save(out/'rollout.gif',save_all=True,append_images=frames[1:],duration=250,loop=0)
            print(json.dumps({'out':str(out),'checks':checks,'error':error,'events':e.events}),flush=True)
        finally:e.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--out',type=Path);parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--viewer',action='store_true')
    args=parser.parse_args()
    trial(args.seed,args.out or ROOT/'reports'/'01_控制验证'/f'peduncle-{time.time_ns()}',not args.no_video,args.viewer)
