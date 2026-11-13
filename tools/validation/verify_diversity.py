"""Offline generation/physics audit. Privileged checks never feed the policy."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import json
from pathlib import Path
import numpy as np
from PIL import Image
from peduncle import PeduncleEnv, build_peduncle


def main():
    out=PROJECT_ROOT/'reports/03_场景调试/diversity-v1-offline';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in range(10):
        e=PeduncleEnv(seed,single=False,render_quality='fast')
        try:
            initial=e.data.xpos[e.fruit_bodies].copy();peak=0.
            for _ in range(40):
                e.step(e.last_action)
                peak=max(peak,float(np.linalg.norm(e.data.xpos[e.fruit_bodies]-initial,axis=1).max()))
            assert peak<.0001,(seed,peak)
            assert e.attached.all() and not e.events
            assert not any(w.number for w in e.data.warning)
            assert np.max([m['bend_rad'] for m in e.stalk_metrics.values()])<.001
            np.testing.assert_allclose(e.base_pose(),e.meta['initial_base_pose'],atol=.0001)
            if seed<3:
                Image.fromarray(e.render('harvest_close',480,360)).save(out/f'seed-{seed}-close.png')
                Image.fromarray(e.render('overview',960,720)).save(out/f'seed-{seed}-overview.png')
            rows.append({'seed':seed,'fruit_count':len(e.fruit_bodies),'task_positions':e.meta['fruit_positions'][:8],
                         'task_maturity':e.meta['maturity'][:8],'task_quat':e.meta['fruit_quat'][:8],
                         'occlusion_profile':e.meta['occlusion_profile'],'initial_base_pose':e.meta['initial_base_pose'],
                         'initial_tcp_world':e.meta['initial_tcp_world'],'max_idle_drift_m':peak,
                         'reachable_ripe_diagnostic':e.target is not None})
            print(seed,rows[-1]['task_maturity'][0],len(e.fruit_bodies),peak,flush=True)
        finally:e.close()
    xml1,m1=build_peduncle(4,False);xml2,m2=build_peduncle(4,False)
    assert xml1==xml2 and m1==m2
    assert len(set(r['task_maturity'][0] for r in rows))==3
    assert len(set(r['occlusion_profile'] for r in rows))==3
    summary={'profile':'greenhouse-v1','seeds':10,'idle_seconds_per_seed':2,
             'deterministic_generation':True,'all_idle_checks_passed':True,
             'network_calls':0,'harvest_validation':False,
             'occlusion_note':'low/medium/high are leaf-retention settings, not measured visibility percentages',
             'results':rows}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
