"""Ground scene checks: reset stability, wheel traction, parking and camera outputs."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import json
import numpy as np
from env import StrawberryEnv
import mujoco
from scene import ROOT

def main():
    results=[]
    for seed in [0,1,2]:
        e=StrawberryEnv(seed)
        try:
            origin=e.base_pose()
            for _ in range(30):e.step(e.last_action)
            drift=max(np.linalg.norm(e.fruit_pos(i)-e.anchors[i]) for i in range(8))
            errors=[e.ik(p)[1] for p in e.anchors]
            assert drift<.001,(seed,drift)
            assert not e.events and not e.success
            assert max(errors)<.006,(seed,errors)
            assert np.linalg.norm(e.base_pose()-origin)<.003
            results.append({'seed':seed,'fruit_drift_m':drift,'max_ik_error_m':max(errors),'parked_base_pose':e.base_pose().tolist()})
        finally:e.close()
    e=StrawberryEnv(0)
    try:
        start=e.base_pose(); e.drive(.08,0)
        for _ in range(40):e.step(e.last_action)
        end=e.base_pose(); distance=end[1]-start[1]
        assert .08<distance<.22,(start,end)
        assert abs(end[0]-start[0])<.02
        wheel_speeds=e.data.qvel[[e.model.joint(n).dofadr[0] for n in ['wheel_left_front','wheel_left_rear','wheel_right_front','wheel_right_rear']]]
        assert np.all(wheel_speeds<-.2),wheel_speeds
        e.drive(0,0)
        for _ in range(30):e.step(e.last_action)
        parked=e.base_pose()
        assert np.linalg.norm(parked-end)<.006,(parked,end)
        # Independent turning fixture starts in the clear aisle. Reset placement is
        # not a driving maneuver: in-place turning beside the bed correctly collides.
        e.close(); e=StrawberryEnv(0)
        e.data.qpos[e.base_qa[0]]=-.25
        e.park_pose=e.base_pose()
        mujoco.mj_forward(e.model,e.data)
        turn_start=e.base_pose()
        e.drive(0,.25)
        for _ in range(40):e.step(e.last_action)
        turned=e.base_pose()
        assert .15<turned[2]-turn_start[2]<.65,(turned,parked)
        e.drive(0,0)
        for _ in range(20):e.step(e.last_action)
        for cam in ['top','wrist']:
            assert e.render(cam,320,240).shape==(240,320,3)
        assert not e.events
        mobile={'start':start.tolist(),'after_forward':end.tolist(),'wheel_speeds_rad_s':wheel_speeds.tolist(),'after_brake':parked.tolist(),'turn_fixture_start':turn_start.tolist(),'after_turn':turned.tolist(),'final_basket_world':e.basket_position().tolist(),'warnings':sum(w.number for w in e.data.warning)}
    finally:e.close()
    report={'scene':'natural-ground-v3','status':'passed','stationary':results,'drive':mobile,'note':'Wheel-ground traction with passive vertical travel; roll/pitch constrained, no rough-ground validation.'}
    out=ROOT/'reports/03_场景调试/natural-v3';out.mkdir(exist_ok=True);(out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
