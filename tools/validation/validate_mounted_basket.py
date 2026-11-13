"""Replay historical picking controls in the full new scene, then test rear delivery.

No model calls or hidden-state corrections. This is a changed-scene offline
diagnostic, not an exact historical replay or a new online success-rate sample.
"""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from peduncle import PeduncleEnv
from vista import Session, evaluate, jaw_contact_state


def validate(source, out):
    out.mkdir(parents=True, exist_ok=False)
    config=json.loads((source/'run-config.json').read_text())
    rows=[json.loads(l) for l in (source/'trajectory.jsonl').read_text().splitlines()]
    e=PeduncleEnv(config['seed'], single=False, fixed_base=config.get('fixed_base',False), render_quality='fast')
    trace=[]
    def tick(q,phase):
        trace.append({'phase':phase,'base_pose':e.base_pose().tolist(),
                      'jaw_contact':jaw_contact_state(e),'tcp':e.data.site_xpos[e.tcp].tolist()})
        e.step(q)
    session=Session.__new__(Session)
    session.e=e;session.grasp_attempted=True;session.tick=tick
    try:
        for row in rows:
            if row['phase']=='delivery_carry':break
            e.drive(*row['base_command']);e.step(row.get('requested_action',row['action']))
        else:raise ValueError('Source did not reach delivery')
        start={'frame':e.frame,'jaw_contact':jaw_contact_state(e),
               'detached_count':int(np.count_nonzero(~e.attached))}
        try:result=session.deliver_to_basket()
        except (ValueError,RuntimeError) as exc:result={'error':str(exc)}
        root=PROJECT_ROOT
        report={'source':str(source.resolve()),'model_calls':0,'formal_training_demo':False,
                'note':__doc__,'start':start,'delivery':result,'evaluation':evaluate(e,trace),
                'warnings':[int(w.number) for w in e.data.warning],'events':e.events,
                'source_sha256':{n:hashlib.sha256(project_source(n).read_bytes()).hexdigest()
                                 for n in ('mounted_basket.py','vista.py','peduncle.py','env.py')}}
        (out/'summary.json').write_text(json.dumps(report,indent=2))
        (out/'delivery-trace.json').write_text(json.dumps(trace))
        for camera in ('robot_detail','chassis_side'):
            Image.fromarray(e.render(camera,960,720)).save(out/(camera+'.png'))
        print(json.dumps(report,indent=2),flush=True)
    finally:e.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();validate(a.source,a.out)
