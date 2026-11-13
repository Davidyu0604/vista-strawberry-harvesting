"""Offline scene distribution and camera/motion checks; no model or harvest claim."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT / 'src'))


def distribution(out,count):
    from mobile_scene import sample_mobile_layout
    spec=importlib.util.spec_from_file_location('old_mobile',out/'source-before/mobile_scene.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rows=[]
    for seed in range(count):
        a=sample_mobile_layout(seed);p=np.asarray(a['positions']);dy=p[:,1]-a['initial_base_y'];ripe=np.array(a['maturity'])=='ripe'
        neg=np.flatnonzero(ripe&(dy>=-.38)&(dy<=-.26));pos=np.flatnonzero(ripe&(dy>=.26)&(dy<=.38))
        assert len(neg) and len(pos)
        assert ((~ripe)&(dy>=-.14)&(dy<=.10)).any()
        distance=np.linalg.norm(p[:,None]-p[None,:],axis=2);np.fill_diagonal(distance,np.inf)
        assert distance.min()>.08
        if a['start_region']=='unripe':assert not (ripe&(abs(dy)<.20)).any()
        if seed<100:
            original=old.sample_mobile_layout(seed,'stress');current=sample_mobile_layout(seed,'stress')
            np.testing.assert_array_equal(original['positions'],current['positions']);assert original['maturity']==current['maturity']
        rows.append(dict(seed=seed,attempts=a['sampling_attempts'],negative_ripe_dy_m=dy[neg].tolist(),positive_ripe_dy_m=dy[pos].tolist(),near_row_fruit_count=len(p)))
    result=dict(seed_count=count,all_passed=True,stress_profiles_unchanged_seeds=100,max_sampling_attempts=max(r['attempts'] for r in rows),rows=rows)
    (out/'distribution.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Distribution passed:',count,'seeds; max attempts',result['max_sampling_attempts'],flush=True)


def validate_scene(seed,out):
    from peduncle import PeduncleEnv
    from vista import Session
    import mujoco
    from PIL import Image
    env=PeduncleEnv(seed,single=False,mobile_layout=True,mobile_layout_level='moderate',render_quality='fast',initial_arm_pose='search_stow')
    dest=out/f'seed-{seed}';dest.mkdir(exist_ok=True)
    class Recorder:
        def __init__(self):self.out=dest
        def tick(self,q,phase):env.step(q)
    session=Session(env,Recorder(),None,centered_grasp=True,visual_proposals=True)
    start=float(env.meta['harvest_base_y']);points=np.asarray(env.meta['fruit_positions']);ripe=np.asarray(env.meta['ripe'],bool)
    directions={side:np.flatnonzero(ripe & (sign*(points[:,1]-start)>=.26) & (sign*(points[:,1]-start)<=.38)).tolist() for side,sign in [('negative',-1),('positive',1)]}
    result=dict(seed=seed,scene_version=env.meta['scene_version'],initial_base_y=start,opportunities_offline={k:[dict(fruit=i,position_m=points[i].tolist()) for i in ids] for k,ids in directions.items()},snapshots=[],moves=[],model_calls=0)
    def snapshot(label):
        frame,state,obs=session.observe()
        # Offline segmentation audit only; never passed to Session or model.
        counts={}
        for camera in ('harvest_close','wrist'):
            renderer=env._camera_renderer(camera,480,360);renderer.enable_segmentation_rendering()
            try:seg=renderer.render().copy()
            finally:renderer.disable_segmentation_rendering()
            counts[camera]={}
            for side,ids in directions.items():
                counts[camera][side]={}
                for i in ids:
                    gids=np.flatnonzero((env.model.geom_bodyid==env.fruit_bodies[i]) & (env.model.geom_type==mujoco.mjtGeom.mjGEOM_MESH))
                    pixels=int((np.isin(seg[:,:,0],gids)&(seg[:,:,1]==mujoco.mjtObj.mjOBJ_GEOM)).sum())
                    counts[camera][side][str(i)]=pixels
            Image.fromarray(env.render(camera,480,360)).save(dest/f'{label}-{camera}.jpg')
        result['snapshots'].append(dict(label=label,frame=frame,base_y=float(env.base_pose()[1]),visible_body_pixels_offline=counts))
    try:
        assert env.meta['scene_version']=='peduncle-v23-mobile-bilateral'
        for _ in range(20):session.tick(env.last_action,'settle')
        snapshot('start')
        # Empty queues are explicit mechanical fixtures, not claims of no fruit.
        for index,direction in enumerate((-1,1,1,1)):
            session.local_targets.register([],env.base_pose(),'Offline motion fixture; not visual evidence of fruit absence')
            result['moves'].append(session.search_row(direction))
            if index in (0,2,3):snapshot({0:'minus18cm',2:'plus18cm',3:'plus36cm'}[index])
        for side,labels in [('negative',('minus18cm',)),('positive',('plus18cm','plus36cm'))]:
            visible=max(pixels for s in result['snapshots'] if s['label'] in labels for c in s['visible_body_pixels_offline'].values() for pixels in c[side].values())
            result[side+'_max_visible_body_pixels']=visible
            assert visible>=50,(seed,side,visible)
        result['warnings']=[int(w.number) for w in env.data.warning];assert not any(result['warnings'])
        result['all_fruit_attached']=bool(env.attached.all());assert result['all_fruit_attached']
        result['source_sha256']={n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest() for n in ('mobile_scene.py','scene.py','peduncle.py')}
        (out/f'seed{seed}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('Scene passed:',seed,{k:result[k] for k in ('negative_max_visible_body_pixels','positive_max_visible_body_pixels','warnings')},flush=True)
    finally:env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=1000);args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    distribution(out,args.count)
    for seed in (36,38,42):validate_scene(seed,out)

if __name__=='__main__':main()
