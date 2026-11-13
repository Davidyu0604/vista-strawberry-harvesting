"""Task randomization, separation, visibility and repeatability acceptance."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import json
import numpy as np
from env import StrawberryEnv
from scene import ROOT,build
import mujoco

def main():
    cases=[];layouts=[]
    for seed in [0,1,2,11,42]:
        e=StrawberryEnv(seed)
        try:
            meta=e.meta;layouts.append(meta['maturity'])
            assert meta['ripe'][0] and meta['protected_ripe_fruit']==0
            assert all(.86<=s<=1.08 for s in meta['fruit_scales'])
            assert len(meta['fruit_positions'])==8
            for _ in range(20):e.step(e.last_action)
            drift=max(np.linalg.norm(e.fruit_pos(i)-e.anchors[i]) for i in range(8))
            assert drift<.001,(seed,drift)
            errors=[e.ik(p)[1] for p in e.anchors];assert max(errors)<.006,(seed,errors)
            assert not e.events and not e.success
            # Count actually visible fruit pixels, including leaf/depth occlusion.
            with mujoco.Renderer(e.model,240,320) as renderer:
                renderer.enable_segmentation_rendering()
                renderer.update_scene(e.data,camera='top',scene_option=e.opt)
                seg=renderer.render();ids=seg[:,:,0];valid=(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&(ids>=0)
                body=np.full(ids.shape,-1);body[valid]=e.model.geom_bodyid[ids[valid]]
                visible=int(np.sum(body==e.fruit_bodies[0]))
                assert visible>=12,(seed,'protected fruit hidden',visible)
            spin_check=None
            if seed==0:
                jid=e.model.joint('fruit_joint_2').id; da=e.model.jnt_dofadr[jid]; bid=e.fruit_bodies[2]
                e.data.qvel[da+3:da+6]=[.5,.2,.3]
                before=.5*np.sum(e.model.body_inertia[bid]*e.data.qvel[da+3:da+6]**2)
                mujoco.mj_forward(e.model,e.data)
                for _ in range(40):e.step(e.last_action)
                after=.5*np.sum(e.model.body_inertia[bid]*e.data.qvel[da+3:da+6]**2)
                assert after<before*.01,(before,after)
                spin_check={'initial_rotational_energy':before,'final_rotational_energy':after}
            cases.append({'spin_damping':spin_check,'seed':seed,'target':e.target,'maturity':meta['maturity'],'scales':meta['fruit_scales'],'protected_fruit_visible_pixels_top_320x240':visible,'max_ik_error_m':max(errors),'stationary_drift_m':drift,'background_fruit_count':meta['background_fruit_count']})
        finally:e.close()
    assert len({tuple(x) for x in layouts})>1
    assert build(11)[1]==build(11)[1]
    report={'status':'passed','scene':'natural-ground-v3','cases':cases,'scope':'Visible reachable ripe target, geometry and stationary stability; no broad policy-success claim.'}
    (ROOT/'reports/03_场景调试/natural-v3/natural-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
