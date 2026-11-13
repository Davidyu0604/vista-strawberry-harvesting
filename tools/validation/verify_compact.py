"""Check compact chassis envelope and geometric clearances at its home pose."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import json
import numpy as np
from env import StrawberryEnv
import mujoco

def main():
    e=StrawberryEnv(0);m=e.model;d=e.data
    try:
        lo=np.full(3,np.inf);hi=-lo
        for g in range(m.ngeom):
            name=m.geom(g).name or ''
            if name.endswith('_contact') and name.startswith('wheel_'):
                center=d.geom_xpos[g];size=m.geom_size[g]
                extent=np.array([size[1],size[0],size[0]])
                lo=np.minimum(lo,center-extent);hi=np.maximum(hi,center+extent)
        np.testing.assert_allclose(hi-lo,[.245,.300,.100],atol=1e-8)
        np.testing.assert_allclose(m.body('wheel_left_front').pos,[-.1,.04,.05])
        assert abs(m.geom('deck').size[0]*2-.15)<1e-9
        # Wheel cylinder inner edge x=.0775; chassis outer edge x=.075.
        clearance=.1-.0225-.075;assert clearance>.002
        base_low=np.inf
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=m.body('base').id or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            mesh=m.geom_dataid[g];verts=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
            points=verts@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            base_low=min(base_low,float(points[:,2].min()))
        top=d.geom_xpos[m.geom('arm_adapter').id,2]+m.geom('arm_adapter').size[2]
        assert base_low>=top-1e-7,(base_low,top)
        for _ in range(40):e.step(e.last_action)
        assert e.worst_obstacle_penetration<.001
        assert not any(w.number for w in d.warning)
        print(json.dumps({'status':'passed','chassis_length_width_height_m':(hi-lo)[[1,0,2]].tolist(),'wheel_sideplate_clearance_m':clearance,'arm_mesh_bottom_minus_adapter_top_m':base_low-top,'home_sampled_penetration_m':e.worst_obstacle_penetration,'electronics_mass_estimate_kg':.35,'limitations':'Assembly clearances and collision contacts, not a proof of no intersections at every possible arm pose'},indent=2))
    finally:e.close()
if __name__=='__main__':main()
