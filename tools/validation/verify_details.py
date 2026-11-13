"""Ensure visual fittings preserve dynamics and do not obstruct the task cameras."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import importlib.util,json
import numpy as np
from env import StrawberryEnv
from scene import ROOT
import mujoco

def main():
    spec=importlib.util.spec_from_file_location('scene_v3_reference',ROOT/'versions/natural-v3/scene.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    # The archived source resolves assets from its module directory; use current pinned assets.
    old.ROOT=ROOT
    xml,metadata=old.build(0); reference=mujoco.MjModel.from_xml_string(xml)
    results=[]
    for seed in [0,1,2]:
        e=StrawberryEnv(seed)
        try:
            if seed==0:
                for name in ['body_mass','body_inertia','body_ipos','body_iquat','jnt_range','actuator_ctrlrange','actuator_forcerange']:
                    np.testing.assert_allclose(getattr(e.model,name),getattr(reference,name),atol=1e-12,rtol=1e-12,err_msg=name)
                def collision_signature(m):
                    return sorted((m.body(int(m.geom_bodyid[g])).name,int(m.geom_type[g]),tuple(m.geom_pos[g]),tuple(m.geom_size[g])) for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g])
                assert collision_signature(e.model)==collision_signature(reference)
                assert e.meta['fruit_positions']==metadata['fruit_positions']
            for _ in range(20):e.step(e.last_action)
            assert not e.events and np.isfinite(e.data.qpos).all()
            visible={}
            with mujoco.Renderer(e.model,240,320) as renderer:
                for camera in ['top','wrist']:
                    renderer.enable_segmentation_rendering();renderer.update_scene(e.data,camera=camera,scene_option=e.opt)
                    seg=renderer.render();ids=seg[:,:,0];valid=(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&(ids>=0)
                    bodies=np.full(ids.shape,-1);bodies[valid]=e.model.geom_bodyid[ids[valid]]
                    count=int(np.sum(bodies==e.fruit_bodies[e.target]));assert count>=12,(seed,camera,count)
                    visible[camera]=count
            results.append({'seed':seed,'target':e.target,'visible_target_pixels_320x240':visible,'warnings':sum(w.number for w in e.data.warning)})
        finally:e.close()
    report={'status':'passed','scene':'photo-base-v6','dynamics_and_collision_unchanged_from_v3':True,'camera_visibility':results}
    (ROOT/'reports/03_场景调试/photo-v6/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
