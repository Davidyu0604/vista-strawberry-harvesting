"""Offline parking/entry screening. Not policy evaluation or training data.

All candidates share an exact replay checkpoint and the logged RGB-D target.
Moving candidates remeasure that visible target from the new close-camera RGB-D.
Hidden object identity is used only to evaluate contacts/detachment/delivery.
"""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse,copy,json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from peduncle import PeduncleEnv
from vista import Session,tilted_stalk_grasp_point,verify_projected_point,depth_point,evaluate
from visual_maturity import assess_maturity
from compare_approach_offline import FIELDS


def compare(episode,out,only=None,extended=False,contact_tolerant=True,retest_contact_rejections=False):
    cfg=json.loads((episode/'run-config.json').read_text())
    source=[json.loads(l) for l in (episode/'trajectory.jsonl').read_text().splitlines()]
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    target=np.array(next(r['result']['locked_target_m'] for r in events if r['kind']=='receipt' and isinstance(r.get('result'),dict) and r['result'].get('target_confirmed')))
    e=PeduncleEnv(seed=cfg['seed'],single=False,render_quality=cfg['quality']);out.mkdir(parents=True,exist_ok=True)
    results=[]
    try:
        drift=0.
        for r in source:
            drift=max(drift,float(np.max(np.abs(e.state()-r['state']))))
            if r['phase']=='visual_pregrasp':break
            e.drive(*r['base_command']);e.step(r['requested_action'])
        snapshot=mujoco.MjData(e.model);mujoco.mj_copyData(snapshot,e.model,e.data)
        saved={k:copy.deepcopy(getattr(e,k)) for k in FIELDS}
        print(json.dumps({'checkpoint_frame':e.frame,'max_replay_joint_error_rad':drift}),flush=True)
        arm=set()
        for b in range(e.model.nbody):
            parent=b
            while parent:
                if parent==e.model.body('base').id:arm.add(b);break
                parent=int(e.model.body_parentid[parent])
        variants=[('reference',0.,0.),('park_minus40',-.04,0.),('park_plus40',.04,0.),('entry_minus20',0.,-20.),('entry_plus20',0.,20.)]
        variants=[(*v,0.) for v in variants]
        extra=[('park_plus40_raise3',.04,0.,.003),('park_plus40_raise6',.04,0.,.006),('park_plus80',.08,0.,0.),('park_minus80',-.08,0.,0.)]
        variants=extra if extended else variants
        if only:variants=[v for v in variants+extra if v[0]==only][:1]
        if retest_contact_rejections:
            if only or extended or not contact_tolerant:
                raise ValueError('Contact retest requires --contact-tolerant without --only/--extended')
            variants=[v for v in variants+extra if v[0] in ('park_plus40','park_plus40_raise3','park_plus40_raise6','park_plus80')]
        if not variants:raise ValueError('Unknown variant')
        for name,offset,degrees,height in variants:
            mujoco.mj_copyData(e.data,e.model,snapshot)
            for k,v in saved.items():setattr(e,k,copy.deepcopy(v))
            rows=[];max_veg=0.;fruit_hits=set();phase='relocation';relocation_max=0.;relocation_fruit=set()
            s=Session.__new__(Session);s.e=e;s.calls=0;s.max_calls=32;s.grasp_attempted=False;s.target_stage=None
            s.target_confirmed=False;s.maturity_evidence=None;s.approach_pitch=0;s.locked_target=None
            def tick(q,label):
                nonlocal max_veg,relocation_max
                e.step(q);rows.append({'base_pose':e.base_pose().tolist()})
                if any(w.number for w in e.data.warning):raise ValueError('Physics warning; candidate stopped')
                if phase not in ('relocation','approach'):return
                for c in e.data.contact:
                    g1,g2=int(c.geom1),int(c.geom2);b1,b2=map(int,e.model.geom_bodyid[[g1,g2]])
                    if (b1 in arm)==(b2 in arm):continue
                    other=g2 if b1 in arm else g1;body=e.model.body(int(e.model.geom_bodyid[other])).name
                    if body.startswith(('stem_','leaf_','branch_')):
                        if phase=='approach':max_veg=max(max_veg,float(-c.dist))
                        else:relocation_max=max(relocation_max,float(-c.dist))
                    if (e.model.geom(other).name or '').startswith('fruit_collision_'):
                        (fruit_hits if phase=='approach' else relocation_fruit).add(body)
            s.tick=tick
            def move(point,label):
                q,error=e.ik(np.asarray(point),1,approach_pitch=0)
                if error>.006:raise ValueError(label+' IK unreachable')
                for _ in range(70):tick(q,label)
                if np.linalg.norm(e.data.site_xpos[e.tcp]-point)>.008:raise ValueError(label+' blocked')
            result={'variant':name,'base_offset_m':offset,'entry_angle_deg':degrees,'grasp_axis_offset_m':height,'passed':False,
                    'last_completed_stage':'checkpoint','damage_assessment':'unavailable: contact penetration is not a calibrated damage measure'}
            try:
                # Identical stow/restore protocol for reference and alternatives.
                start=e.base_pose().copy();move([.19,start[1]-.04,.34],'experiment_stow')
                goal=start[1]+offset
                for _ in range(160):
                    error=goal-e.base_pose()[1]
                    if abs(error)<.003:break
                    e.drive(float(np.clip(error,-.03,.03)),0);tick(e.last_action,'experiment_drive')
                else:raise ValueError('Parking timeout')
                e.drive(0,0)
                for _ in range(20):tick(e.last_action,'experiment_park')
                move([.19,e.base_pose()[1]-.15,.33],'experiment_observe')
                rgb=Image.fromarray(e.render('harvest_close',480,360));depth,cal=e.depth_view('harvest_close',480,360)
                check=verify_projected_point(target,depth,cal)
                if not check['consistent']:raise ValueError('Original visible target cannot be remeasured')
                x,y=check['sample_pixel'];point=depth_point(depth,cal,x,y)
                measured,fit=tilted_stalk_grasp_point(point,depth,cal,rgb)
                if np.linalg.norm(measured-target)>.008:raise ValueError('Remeasured stalk disagrees with logged target')
                measured=measured+height*np.asarray(fit['axis_world'])
                maturity,marked=assess_maturity(measured,fit['axis_world'],depth,cal,rgb)
                result['maturity']=maturity
                marked.save(out/(name+'-target.jpg'))
                if not maturity['passed']:raise ValueError('Visual maturity rejected')
                if not contact_tolerant and (relocation_max>.001 or relocation_fruit):raise ValueError('Relocation fails interference screen')
                s.locked_target=measured;s.target_confirmed=True;s.maturity_evidence=maturity
                angle=np.deg2rad(degrees);direction=np.array([np.cos(angle),np.sin(angle),0.])
                phase='approach';move(measured-.055*direction,'experiment_pregrasp');s.target_stage='pregrasp'
                result['approach']=s.execute_approach('grasp')
                result['grasp_tcp_error_m']=result['approach']['tcp_error_m']
                result['last_completed_stage']='approach'
                if not contact_tolerant and (max_veg>.001 or fruit_hits):raise ValueError('Approach fails interference screen')
                phase='grasp'
                for _ in range(20):tick(e.last_action,'experiment_observe_grasp')
                receipt=s.execute([{'action':'gripper','opening':-.15,'seconds':2.5}])
                if receipt[0]['status']!='executed':raise ValueError('Close rejected')
                result['last_completed_stage']='close'
                for _ in range(20):tick(e.last_action,'experiment_hold')
                ids=np.flatnonzero(e.bilateral_contacts()).tolist();result['bilateral_ids']=ids
                if not ids:raise ValueError('No bilateral stalk contact')
                result['last_completed_stage']='bilateral_stalk_hold'
                # Fixed bounded pull schedule from an existing successful demo.
                for dx,dz,pitch,seconds in [(-.035,0,-37,3),(-.045,.005,-37,2.5),(-.05,.02,-30,2.5)]:
                    receipt=s.execute([{'action':'bend_pull','dx':dx,'dy':0,'dz':dz,'pitch_deg':pitch,'seconds':seconds}])
                    if receipt[0]['status']!='executed':raise ValueError('Pull rejected: '+receipt[0].get('reason',''))
                    for _ in range(20):tick(e.last_action,'experiment_inspect_pull')
                held=[i for i in ids if not e.attached[i] and e.contacts(i)]
                result['detached_held_ids']=held
                if not held:raise ValueError('No detached fruit retained')
                result['last_completed_stage']='detached_and_held'
                # Independent mechanical eligibility, not an online oracle tool.
                result['delivery']=s.deliver_to_basket()
                for _ in range(40):tick(e.last_action,'terminal_settle')
                result['evaluation']=evaluate(e,rows)
                result['passed']=bool(result['evaluation']['success'])
                result['last_completed_stage']='delivery_evaluated'
            except ValueError as exc:result['reason']=str(exc)
            finally:e.drive(0,0)
            result.pop('approach',None)
            result.update(max_approach_vegetation_penetration_m=max_veg,approach_fruit_contact=sorted(fruit_hits),
                          max_relocation_vegetation_penetration_m=relocation_max,relocation_fruit_contact=sorted(relocation_fruit),
                          physics_warnings=bool(any(w.number for w in e.data.warning)))
            results.append(result)
            screen=({'allow_fruit_contact_before_close':True,'penetration_is_rejection_criterion':False,
                     'required':'Tracking, stable hold, ripe-fruit detachment, transport and basket evaluation; physics warnings still stop candidates',
                     'note':'Contact alone is allowed. Tissue damage is not assessed; object identities remain offline evaluation only.'}
                    if contact_tolerant else {'max_vegetation_penetration_m':.001,'allow_fruit_contact_before_close':False,'note':'Legacy conservative screen, not a calibrated damage threshold'})
            report={'episode':str(episode),'max_replay_joint_error_rad':drift,'screen':screen,'results':results}
            (out/'comparison.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(result),flush=True)
    finally:e.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--only');p.add_argument('--extended',action='store_true')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--contact-tolerant',dest='contact_tolerant',action='store_true',default=True,help='Default: allow contact and judge mechanical outcome')
    mode.add_argument('--legacy-interference-screen',dest='contact_tolerant',action='store_false',help='Reproduce the old conservative contact rejection screen; not a damage criterion')
    p.add_argument('--retest-contact-rejections',action='store_true',help='Retest four seed3 candidates previously stopped solely by interference screening')
    a=p.parse_args()
    compare(a.episode,a.out,a.only,a.extended,a.contact_tolerant,a.retest_contact_rejections)
