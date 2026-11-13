"""Preview, native interactive viewer, and privileged scripted diagnostics."""
import argparse, json, time, queue, os
from recording_quality import atomic_json, phase_labels, audit_recording
from pathlib import Path
import numpy as np
from PIL import Image
from env import StrawberryEnv
from scene import ROOT,build

def preview(env,out):
    out.mkdir(parents=True,exist_ok=True)
    for name in ['overview','detail','top','wrist','plant_detail','robot_detail','chassis_side']:
        Image.fromarray(env.render(name,960,720)).save(out/f'{name}.png')
    xml,meta=env.xml,env.meta; (out/'scene.xml').write_text(xml)
    (out/'scene.json').write_text(json.dumps(meta,indent=2))

class Recorder:
    """Record observations BEFORE stepping, paired with actual command sent that tick."""
    def __init__(self,env,out,images=True,video=True):
        if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}')
        out.mkdir(parents=True); self.env=env; self.out=out; self.images=images
        self.rows=[]; self.frames=[]; self.video=video
        self.cameras=list(env.meta.get('policy_cameras', ['top','wrist','harvest_close'] if env.meta.get('harvest_mechanism') else ['top','wrist'])) if images else []
        self.image_size=(480,360)
        self.metadata={'schema_version':3 if env.meta.get('basket_mount')=='chassis' else 2,'status':'recording','control_dt':env.dt,
                       'cameras':self.cameras,'image_size':list(self.image_size),
                       'basket_mount':env.meta.get('basket_mount','ground'),
                       'wrist_camera_profile':env.meta.get('wrist_camera_profile','legacy-gripper-v1'),
                       'joint_control_limits':env.model.actuator_ctrlrange[:6].tolist(),
                       'alignment':'state/images at t, actual command applied over [t,t+dt)',
                       'seed':env.meta.get('seed'),'scene_version':env.meta.get('scene_version'),
                       'privileged_fields':['target','fruit_position','attached','bilateral_contact'],
                       'action_units':'six absolute joint targets (radians), gripper included; base_command m/s and rad/s'}
        atomic_json(out/'recording.json',self.metadata)
        (out/'trajectory.jsonl').touch()
    def tick(self,goal,phase):
        e=self.env; state=e.state(); t=float(e.data.time)
        row={'frame':e.frame,'timestamp':t,'phase':phase,'state':state.tolist(),'target':e.target,
             'fruit_position':e.fruit_pos(e.target).tolist() if e.target is not None else None,'tcp_position':e.data.site_xpos[e.tcp].tolist(),'bilateral_contact':bool(e.contacts(e.target)) if e.target is not None else False,'attached':bool(e.attached[e.target]) if e.target is not None else None,'images':{},'base_pose':e.base_pose().tolist(),'base_command':e.drive_command.tolist()}
        stage,waiting,recovery=phase_labels(phase)
        row.update(stage=stage,is_model_wait=waiting,is_recovery=recovery,
                   requested_action=np.asarray(goal,dtype=float).tolist(),image_timestamps={})
        if self.images:
            for camera in self.cameras:
                path=f'{camera}/{e.frame:06d}.jpg'; (self.out/camera).mkdir(exist_ok=True)
                image=Image.fromarray(e.render(camera,*self.image_size))
                temp=(self.out/path).with_suffix('.jpg.tmp')
                image.save(temp,format='JPEG',quality=92);temp.replace(self.out/path)
                row['image_timestamps'][camera]=t
                row['images'][camera]=path
        if self.video and e.frame%4==0: self.frames.append(Image.fromarray(e.render('detail',640,480)))
        if e.frame!=row['frame'] or float(e.data.time)!=t:
            raise RuntimeError('Simulation advanced during synchronized camera capture')
        atomic_json(self.out/'pending-step.json',row)
        row['action']=e.step(goal).tolist()
        row.update(next_timestamp=float(e.data.time),next_frame=e.frame)
        with (self.out/'trajectory.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        self.rows.append(row)
        (self.out/'pending-step.json').unlink()
    def finish(self):
        e=self.env
        # Committed steps are already on disk; never rewrite the recovery log.
        report={**e.meta,'target':e.target,'success':e.success,'events':e.events,'frames':len(self.rows),
                'source':'privileged scripted diagnostic; uses simulator target positions',
                'formal_training_demo':False,'no_grasp_weld':True,'max_sampled_arm_obstacle_penetration_m':e.worst_obstacle_penetration,'worst_obstacle_pair':e.worst_obstacle_pair,'max_sampled_arm_vegetation_penetration_m':e.worst_vegetation_penetration,'terminal_state':e.state().tolist(),
                'terminal_fruit_position':e.fruit_pos(e.target).tolist() if e.target is not None else None,'terminal_base_pose':e.base_pose().tolist(),'terminal_basket_position':e.basket_position().tolist()}
        atomic_json(self.out/'report.json',report)
        self.metadata['status']='finalized'
        atomic_json(self.out/'recording.json',self.metadata)
        audit_recording(self.out)
        Image.fromarray(e.render('detail',960,720)).save(self.out/'terminal.png')
        if self.frames: self.frames[0].save(self.out/'rollout.gif',save_all=True,append_images=self.frames[1:],duration=200,loop=0)
        return report

def scripted(env,rec):
    p=env.anchors[env.target].copy()
    def move(pos,grip,phase,ticks):
        q,err=env.ik(np.array(pos),grip)
        if err>.006: raise RuntimeError(f'Unreachable {phase}: {err}')
        for _ in range(ticks): rec.tick(q,phase)
    move(p-[.055,0,0],1,'pregrasp',55)
    move(p,1,'approach',55)
    move(p,-.15,'grasp',55)
    move(p-[.065,0,0],-.15,'pull',65)
    if env.attached[env.target]: return
    move(p-[.09,0,-.04],-.15,'lift',45)
    # Carry the fruit above the chassis before a short, straight return to the station.
    move(p-[.09,0,-.04],-.15,'carry_pose',25)
    def drive_to(y,phase):
        held=env.last_action.copy()
        for _ in range(240):
            error=y-env.base_pose()[1]
            if abs(error)<.004:break
            env.drive(float(np.clip(1.2*error,-.06,.06)),0)
            rec.tick(held,phase)
        else:raise RuntimeError(f'Failed to reach {phase}')
        env.drive(0,0)
        for _ in range(30):rec.tick(held,phase+'_park')
        if abs(y-env.base_pose()[1])>.012:raise RuntimeError('Parking drift')
    drive_to(env.meta['collection_base_y'],'return_to_collection')
    basket=env.basket_position()
    move(basket+[0,0,.26],-.15,'transport',90)
    move(basket+[0,0,.11],-.15,'basket_approach',40)
    move(basket+[0,0,.095],-.15,'lower',35)
    move(basket+[0,0,.095],1,'release',45)
    move(basket+[0,0,.20],1,'vertical_clearance',45)
    move(basket+[0,-.06,.20],1,'retreat',60)
    for _ in range(30): rec.tick(env.last_action,'settle')
    if not env.success:raise RuntimeError('Fruit did not settle inside ground basket')
    move([.19,env.base_pose()[1]-.04,.34],1,'return_stow',65)
    drive_to(env.meta['harvest_base_y'],'return_to_harvest')

def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['preview','viewer','diagnostic'])
    p.add_argument('--seed',type=int,default=0); p.add_argument('--out',type=Path); p.add_argument('--no-images',action='store_true'); p.add_argument('--no-video',action='store_true')
    p.add_argument('--quality',choices=['fast','full'],default=None)
    p.add_argument('--full-physics',action='store_true',help='Keep all plants awake for reference physics runs')
    a=p.parse_args();quality=a.quality or ('fast' if a.mode=='viewer' else 'full')
    e=StrawberryEnv(a.seed,full_physics=a.full_physics,render_quality=quality)
    try:
        if a.mode=='preview':
            out=a.out or ROOT/'reports'/'03_场景调试'/'preview'; preview(e,out); print(out)
        elif a.mode=='diagnostic':
            out=a.out or ROOT/'reports'/'01_控制验证'/f'diagnostic-{a.seed}-{time.time_ns()}'
            r=Recorder(e,out,not a.no_images,not a.no_video)
            try: scripted(e,r)
            except Exception as exc:
                e.meta['error']=repr(exc)
                raise
            finally:
                report=r.finish(); print(json.dumps({'out':str(out),'success':report['success'],'events':report['events']}))
        else:
            import mujoco.viewer
            selected=[e.target]; grip=[1.0]; goal=[e.last_action.copy()]
            keys=queue.SimpleQueue()
            def key(k): keys.put(k)
            def handle_key(k):
                if ord('1')<=k<=ord('8'): selected[0]=k-ord('1')
                if k in [ord('['),ord(']')]:
                    selected[0]=(selected[0]+(1 if k==ord(']') else -1))%len(e.fruit_bodies)
                    print('Selected fruit',selected[0],e.meta['maturity'][selected[0]])
                if k==ord('T'):
                    try:selected[0]=e.select_target();print('Nearest reachable ripe fruit',selected[0])
                    except RuntimeError as exc:print(exc)
                if k==ord('W'): e.drive(.08,0)
                if k==ord('S'): e.drive(-.08,0)
                if k==ord('Q'): e.drive(0,.25)
                if k==ord('E'): e.drive(0,-.25)
                if k==32: e.drive(0,0)
                if k in [ord(c) for c in 'AGCPOB']: e.drive(0,0)
                if e.target!=selected[0]:e.target=selected[0];e.in_box_ticks=0
                fruit=e.fruit_pos(selected[0])
                if k==ord('A'): goal[0]=e.ik(fruit+.055*e.pull_direction(selected[0]),1)[0]
                if k==ord('G'): goal[0]=e.ik(fruit,1)[0]
                if k==ord('C'): goal[0]=goal[0].copy(); goal[0][5]=-.15
                if k==ord('O'): goal[0]=goal[0].copy(); goal[0][5]=1
                if k==ord('P'): goal[0]=e.ik(e.data.site_xpos[e.tcp]+.065*e.pull_direction(selected[0]),-.15)[0]
                if k==ord('B'): goal[0]=e.ik(e.basket_position()+[0,0,.11],-.15)[0]
            print('1-8 quick select; [/] cycle all fruit; T nearest ripe; A pregrasp; G approach; C close; P pull; B basket; O open. W/S drive; Q/E turn; SPACE brake. Close window to exit.')
            with mujoco.viewer.launch_passive(e.model,e.data,key_callback=key) as viewer:
                viewer.opt.geomgroup[3]=0
                if quality=='fast':
                    viewer.opt.geomgroup[4]=0
                    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
                    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=0
                viewer.cam.lookat[:]=[.12,0,.20]; viewer.cam.distance=1.2; viewer.cam.azimuth=145; viewer.cam.elevation=-25
                while viewer.is_running():
                    start=time.monotonic()
                    with viewer.lock():
                        while not keys.empty(): handle_key(keys.get_nowait())
                        e.step(goal[0])
                    viewer.sync(); time.sleep(max(0,e.dt-(time.monotonic()-start)))
    finally: e.close()
if __name__=='__main__': main()
