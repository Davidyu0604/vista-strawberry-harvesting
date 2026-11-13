"""Offline layout and rendered visibility audit. Truth never enters policy."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse,importlib.util,json
from pathlib import Path
import numpy as np
from mobile_scene import sample_mobile_layout,MODERATE_FORWARD_OPPORTUNITY_M,MODERATE_NEAR_FRUIT_INTERVAL_M

def validate(out,preview):
    root=PROJECT_ROOT
    path=root/'reports/02_批量评估/duplicate-identity-seed2-20261111-130940/source/mobile_scene.py'
    spec=importlib.util.spec_from_file_location('old_layout',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    counts=[];attempts=[];first=set();modes=set()
    for seed in range(1000):
        a=sample_mobile_layout(seed);p=np.asarray(a['positions']);y=p[:,1]-a['initial_base_y'];ripe=np.array(a['maturity'])=='ripe'
        lo,hi=MODERATE_NEAR_FRUIT_INTERVAL_M;assert ((y>=lo)&(y<=hi)&~ripe).any()
        lo,hi=MODERATE_FORWARD_OPPORTUNITY_M;assert ((y>=lo)&(y<=hi)&ripe).any()
        assert a['start_region']!='gap'
        if a['start_region']=='unripe':assert not (ripe&(abs(y)<.20)).any()
        d=np.linalg.norm(p[:,None]-p[None,:],axis=2);np.fill_diagonal(d,np.inf);assert d.min()>.08
        if seed<100:
            before=old.sample_mobile_layout(seed,'stress');after=sample_mobile_layout(seed,'stress')
            for key in before:np.testing.assert_equal(before[key],after[key])
        counts.append(len(p));attempts.append(a['sampling_attempts']);first.add(a['maturity'][0]);modes.add(a['start_region'])
    assert first=={'ripe','turning','green'} and modes=={'unripe','mixed'}
    summary=dict(sampled_seeds=1000,fruit_count_range=[min(counts),max(counts)],max_attempts=max(attempts),stress_unchanged_seeds=100,model_calls=0,videos=0,online_success_validated=False,full_scenes=[])
    print(json.dumps(summary),flush=True)
    from peduncle import PeduncleEnv
    from PIL import Image,ImageDraw
    import mujoco
    preview.mkdir(parents=True,exist_ok=True)
    panels=[]
    for seed in range(6):
        e=PeduncleEnv(seed,single=False,mobile_layout=True,render_quality='fast')
        try:
            for _ in range(20):e.step(e.last_action)
            images=[e.render(c,480,360) for c in ('harvest_close','wrist')]
            assert e.meta['scene_version']=='peduncle-v22-mobile-maturity'
            assert e.meta['occlusion_profile'] in ('low','medium')
            assert e.attached.all() and not any(w.number for w in e.data.warning)
            # Segmentation is used only here, for offline visibility acceptance.
            renderer=e._camera_renderer('harvest_close',480,360)
            renderer.enable_segmentation_rendering()
            try:seg=renderer.render().copy()
            finally:renderer.disable_segmentation_rendering()
            valid=(seg[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&(seg[...,0]>=0)
            bodies=np.full(valid.shape,-1);bodies[valid]=e.model.geom_bodyid[seg[...,0][valid]]
            lo,hi=MODERATE_NEAR_FRUIT_INTERVAL_M;near=[];visible=[]
            for i in range(e.meta['mobile_layout_audit']['near_row_fruit_count']):
                y=float(e.meta['fruit_positions'][i][1]-e.meta['harvest_base_y'])
                px=int((bodies==e.fruit_bodies[i]).sum())
                row=dict(fruit_index_offline=i,maturity=e.meta['maturity'][i],relative_y_m=y,visible_pixels=px)
                if abs(y)<.45 and px>=60:visible.append(row)
                if lo<=y<=hi and row['maturity']!='ripe':near.append(row)
            assert any(r['visible_pixels']>=80 for r in near),(seed,near)
            check=dict(seed=seed,start_region=e.meta['mobile_layout_audit']['start_region'],occlusion=e.meta['occlusion_profile'],near_unripe_visibility=near,visible_local_fruit=visible,all_attached=True,warnings=[int(w.number) for w in e.data.warning],interactive_fruit_count=e.meta['interactive_fruit_count'])
            summary['full_scenes'].append(check);print(json.dumps(check),flush=True)
            panel=Image.fromarray(np.concatenate(images,axis=1));ImageDraw.Draw(panel).text((8,8),f'seed {seed} / {check["start_region"]} / {check["occlusion"]}',fill='white',stroke_width=2,stroke_fill='black')
            panel.save(preview/f'seed-{seed}.jpg');panels.append(panel.resize((480,180)))
        finally:e.close()
    sheet=Image.new('RGB',(960,540))
    for i,panel in enumerate(panels):sheet.paste(panel,((i%2)*480,(i//2)*180))
    sheet.save(preview/'contact-sheet.jpg')
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preview',type=Path,default=Path('/tmp/maturity-layout-preview'));a=p.parse_args();validate(a.out,a.preview)
