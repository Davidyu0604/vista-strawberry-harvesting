"""Reproducible CPU physics and offscreen rendering measurements; no API calls."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import json,time
import numpy as np
from env import StrawberryEnv
from scene import ROOT
from OpenGL.GL import glGetString,GL_RENDERER

def main():
    results=[]
    for full in [True,False]:
        e=StrawberryEnv(full_physics=full)
        try:
            t=time.perf_counter()
            for _ in range(80):e.step(e.last_action)
            wall=time.perf_counter()-t
            item={'profile':'full' if full else 'adaptive_sleep','sim_seconds':4,'wall_seconds':wall,'realtime_factor':4/wall,'awake_dofs':int(e.data.nv_awake),'total_dofs':e.model.nv,'fruit_count':len(e.fruit_bodies),'warnings':sum(w.number for w in e.data.warning)}
            assert not e.events
            if not full:
                render={}
                for quality in ['full','fast']:
                    e.render_quality=quality;e.render('detail',640,480)
                    t=time.perf_counter()
                    for _ in range(5):e.render('detail',640,480)
                    render[quality]=(time.perf_counter()-t)/5
                item['renderer']=glGetString(GL_RENDERER).decode();item['render_seconds_640x480']=render
                # Far fruit must wake when the robot is moved near it in a separate fixture.
                i=next(i for i,p in enumerate(e.anchors) if i>8 and p[0]>.28 and p[0]<.45 and p[1]>1.5)
                asleep_before=bool(e.data.tree_asleep[e.fruit_trees[i]]>=0)
                e.data.qpos[e.base_qa[1]]=e.anchors[i,1];e.park_pose=e.base_pose()
                import mujoco
                mujoco.mj_forward(e.model,e.data)
                e.step(e.last_action)
                awake_after=bool(e.data.tree_asleep[e.fruit_trees[i]]<0)
                assert asleep_before and awake_after,(i,asleep_before,awake_after)
                item['approach_wake_test']={'fruit':i,'asleep_before':asleep_before,'awake_after':awake_after}
            results.append(item)
        finally:e.close()
    report={'results':results,'physics_speedup':results[0]['wall_seconds']/results[1]['wall_seconds']}
    path=ROOT/'reports/03_场景调试/performance-v11/benchmark.json';path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
