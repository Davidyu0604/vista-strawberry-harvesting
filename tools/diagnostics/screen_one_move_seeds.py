"""Offline seed screening. Forced mechanical search is NOT online success."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse,io,json,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from peduncle import PeduncleEnv
from vista import Session,depth_point,tilted_stalk_grasp_point,verify_projected_point
from visual_proposals import propose_bodies
from local_targets import body_pixel_evidence
from stalk_proposals import propose_stalk_points
from visual_maturity import assess_maturity


def inspect(e,s,preview,label):
    views={}
    for cam in ('harvest_close','wrist'):
        raw=Image.fromarray(e.render(cam,480,360));buf=io.BytesIO();raw.save(buf,format='JPEG')
        buf.seek(0);rgb=Image.open(buf).convert('RGB');d,cal=e.depth_view(cam,480,360)
        views[cam]=(rgb,d,cal)
        rgb.save(preview/f'{label}-{cam}.jpg')
    rgb,d,cal=views['harvest_close']
    batch=propose_bodies(rgb,d,cal,e.base_pose(),depth_point);checks=[]
    for p in batch['proposals']:
        if p['surface_m'] is None or not body_pixel_evidence(rgb,*p['pixel'])['passed']:continue
        fitted=propose_stalk_points(rgb,d,cal,p['surface_m'],tilted_stalk_grasp_point)
        row=dict(body=p,close_fits=len(fitted['proposals']),options=[])
        for fit in fitted['proposals']:
            maturity,_=assess_maturity(fit['target_m'],fit['axis_world'],d,cal,rgb)
            wrist=verify_projected_point(fit['target_m'],views['wrist'][1],views['wrist'][2])
            option=dict(target_m=fit['target_m'],axis_world=fit['axis_world'],maturity=maturity,wrist_projection=wrist)
            if maturity['passed']:
                try:
                    plan=s.centered.plan(fit['target_m'],fit['axis_world'])
                    option.update(pregrasp_reachable=True,max_ik_error_m=float(plan['max_ik_error_m']))
                except ValueError as exc:option.update(pregrasp_reachable=False,reason=str(exc))
            row['options'].append(option)
        checks.append(row)
    return dict(proposal_count=len(batch['proposals']),body_proposals=batch['proposals'],
                unresolved_depth_proposals=[p['proposal_id'] for p in batch['proposals'] if p['surface_m'] is None],
                red_body_checks=checks)


def screen(seeds,out,preview,check_recovery=False):
    out.mkdir(parents=True,exist_ok=True);preview.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        print(f'Screen seed {seed}: initialize full greenhouse',flush=True)
        e=PeduncleEnv(seed,single=False,mobile_layout=True,mobile_layout_level='moderate',render_quality='fast')
        try:
            with tempfile.TemporaryDirectory(prefix='one-move-screen-') as tmp:
                class Recorder:
                    out=Path(tmp)
                    def tick(self,q,phase):e.step(q)
                s=Session(e,Recorder(),None,centered_grasp=True,visual_proposals=True)
                for _ in range(10):s.tick(e.last_action,'idle')
                before=inspect(e,s,preview,f'seed-{seed}-before')
                print(f'Seed {seed}: initial vision checked; executing mechanical one-step fixture',flush=True)
                # Explicit empty-queue fixture: not a model permission or claim
                # that initial ripe candidates are absent.
                s.local_targets.register([],e.base_pose(),'OFFLINE MECHANICAL FIXTURE: empty queue, no model survey')
                motion=s.search_row(1)
                after=inspect(e,s,preview,f'seed-{seed}-after')
                recovery=None
                if check_recovery and after['red_body_checks']:
                    first=after['red_body_checks'][0]
                    if not first['close_fits']:
                        print(f'Seed {seed}: check bounded observation recovery of nearest red body',flush=True)
                        anchor={'tcp_world_m':e.data.site_xpos[e.tcp].copy(),'roll':float(e.state()[4])}
                        try:
                            plan=s.centered.observation_plan(anchor,0,body_point=first['body']['surface_m'])
                            adjusted=s.centered.adjust_observation(plan)
                            recovered=inspect(e,s,preview,f'seed-{seed}-recovered')
                            recovery=dict(body=first['body'],adjustment=adjusted,after=recovered)
                        except ValueError as exc:recovery=dict(body=first['body'],error=str(exc))
                result=dict(seed=seed,scene_version=e.meta['scene_version'],before=before,after=after,
                    initial_settle_ticks=10,
                    initial_proposal_queue_empty=not before['body_proposals'],
                    observation_recovery=recovery,
                    actual_move_m=motion['actual_delta_y_m'],warnings=[int(w.number) for w in e.data.warning],
                    all_attached_offline=bool(e.attached.all()),model_calls=0,videos=0,
                    online_one_move_success=False,
                    limitation='Seed shortlist only. Empty-queue fixture forces one search; RGB-D/color/IK checks do not certify semantic identity, online choice, grasp or harvest success.')
                results.append(result)
                (out/f'seed-{seed}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
                good=[opt for c in after['red_body_checks'] for opt in c['options'] if opt.get('pregrasp_reachable')]
                print(json.dumps(dict(seed=seed,after_reachable_red_fits=len(good),actual_move_m=result['actual_move_m'],warnings=result['warnings'])),flush=True)
        finally:e.close()
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--preview',type=Path,default=Path('/tmp/one-move-seed-screen'))
    p.add_argument('--check-nearest-recovery',action='store_true')
    a=p.parse_args();screen(a.seeds,a.out,a.preview,a.check_nearest_recovery)
