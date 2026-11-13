"""Replay logged robot commands, then validate delivery under current physics.

Offline regression only: no model requests, no object positions used for control.
"""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse
import hashlib
import json
from pathlib import Path
import time
from peduncle import PeduncleEnv
from run import Recorder
from vista import Session


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--wait-seconds',type=float,default=0)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    rows=[json.loads(s) for s in (args.source/'trajectory.jsonl').read_text().splitlines()]
    # This episode's first free TCP move is the model's post-pull transport.
    cutoff=next(i for i,r in enumerate(rows) if r['phase']=='move_tcp')
    metadata=json.loads((args.source/'report.json').read_text())
    out=args.out or PROJECT_ROOT/'reports'/'01_控制验证'/f'delivery-replay-{time.time_ns()}'
    e=PeduncleEnv(seed=metadata['seed'],render_quality='fast')
    rec=Recorder(e,out,False,False);s=Session(e,rec,None)
    result=None;error=None
    try:
        for row in rows[:cutoff]:s.tick(row['action'],'replay_'+row['phase'])
        assert not e.attached[0] and e.contacts(0),'Replay did not produce a detached held fruit'
        for _ in range(round(args.wait_seconds/e.dt)):s.tick(e.last_action,'extra_wait')
        s.grasp_attempted=True
        result=s.deliver_to_basket()
        assert e.success,'Fruit not settled in basket'
        assert abs(e.base_pose()[1])<.012,'Vehicle did not return'
        assert not any(w.number for w in e.data.warning),'Physics warning'
    except Exception as exc:
        error=str(exc)
        raise
    finally:
        try:
            report=rec.finish()
            report.update(source='offline logged-command replay under current physics; not a new live model episode',
                          replay_source=str(args.source.resolve()),replay_frames=cutoff,
                          extra_wait_seconds=args.wait_seconds,delivery=result,error=error,
                          source_hashes={name:hashlib.sha256(project_source(name).read_bytes()).hexdigest()
                                         for name in ('vista.py','peduncle.py','env.py')})
            (out/'report.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({'out':str(out),'success':bool(e.success),'error':error}),flush=True)
        finally:e.close()


if __name__=='__main__':main()
