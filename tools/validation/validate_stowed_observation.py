"""Serial mechanical posture/camera checks; no fruit truth used for control."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,json,tempfile
import numpy as np
from PIL import Image
from peduncle import PeduncleEnv
from vista import Session
from run import Recorder
from mounted_basket import ready_point


def validate(out,seeds):
    out.mkdir(parents=True,exist_ok=True);results=[]
    for seed in seeds:
        e=PeduncleEnv(seed,single=False,mobile_layout=True,render_quality='fast',initial_arm_pose='search_stow')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                s=Session(e,Recorder(e,Path(tmp)/'episode',False,False),None,centered_grasp=True,visual_proposals=True)
                for _ in range(10):s.tick(e.last_action,'decision_settle')
                initial_error=float(np.linalg.norm(e.data.site_xpos[e.tcp]-ready_point(e)));q=e.state().copy()
                for camera in ('harvest_close','wrist'):
                    Image.fromarray(e.render(camera,480,360)).save(out/f'seed-{seed}-initial-{camera}.jpg')
                s.local_targets.register([],e.base_pose(),'Offline mechanical fixture: empty queue is not an actual survey')
                result=s.search_row(1)
                error=float(np.linalg.norm(e.data.site_xpos[e.tcp]-ready_point(e)))
                assert initial_error<.004 and error<.008
                assert not any(r['phase']=='search_observe_pose' for r in s.rec.rows)
                assert np.max(abs(e.state()-q))<.01
                assert e.attached.all() and not any(w.number for w in e.data.warning)
                for camera in ('harvest_close','wrist'):
                    Image.fromarray(e.render(camera,480,360)).save(out/f'seed-{seed}-after-search-{camera}.jpg')
                results.append({'seed':seed,'initial_tcp_error_m':initial_error,'parked_tcp_error_m':error,
                    'delta_y_m':result['actual_delta_y_m'],'max_joint_drift_rad':float(np.max(abs(e.state()-q))),
                    'all_attached_offline':bool(e.attached.all()),'warning_count':sum(w.number for w in e.data.warning),
                    'phases':sorted(set(r['phase'] for r in s.rec.rows))})
                print(seed,initial_error,error,flush=True)
        finally:e.close()
    (out/'posture-validation.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[5,7,8,10])
    a=p.parse_args();validate(a.out.resolve(),a.seeds)
