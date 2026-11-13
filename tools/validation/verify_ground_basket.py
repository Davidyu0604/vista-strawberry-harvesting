"""Verify fixed basket placement and completed short shuttle diagnostic recordings."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse,json
from pathlib import Path
import numpy as np
from env import StrawberryEnv

def main():
    parser=argparse.ArgumentParser();parser.add_argument('episodes',nargs='+',type=Path);args=parser.parse_args()
    e=StrawberryEnv(0)
    try:
        basket=e.model.body('basket').id
        assert e.model.body_parentid[basket]==0
        position=e.basket_position()
        floor=e.model.geom('basket_floor').id
        assert abs(e.data.geom_xpos[floor,2]-e.model.geom_size[floor,2])<1e-9
        e.drive(.06,0)
        for _ in range(40):e.step(e.last_action)
        np.testing.assert_allclose(e.basket_position(),position,atol=1e-12)
        assert e.base_pose()[1]>.08
        assert not any(w.number for w in e.data.warning)
    finally:e.close()
    summaries=[]
    for path in args.episodes:
        report=json.loads((path/'report.json').read_text())
        assert report['success'] and 'error' not in report
        if 'max_sampled_arm_obstacle_penetration_m' in report:
            assert report['max_sampled_arm_obstacle_penetration_m']<.0005,(path,report['worst_obstacle_pair'])
        assert len(report['events'])==1 and report['events'][0]['fruit']==report['target']
        rows=[json.loads(line) for line in (path/'trajectory.jsonl').read_text().splitlines()]
        for phase in ['return_to_collection','return_to_collection_park','release','return_to_harvest','return_to_harvest_park']:
            assert any(r['phase']==phase for r in rows),(path,phase)
        last=rows[-1];fruit=np.array(last['fruit_position']);rel=fruit-position
        assert abs(rel[0])<.066 and abs(rel[1])<.048 and .012<rel[2]<.05,(path,rel)
        assert abs(last['base_pose'][1])<.012
        assert max(r['base_pose'][1] for r in rows)>.23
        assert not last['attached']
        summaries.append({'episode':str(path),'frames':len(rows),'final_base_pose':last['base_pose'],'final_fruit_position':fruit.tolist()})
    print(json.dumps({'status':'passed','basket_fixed_and_floor_on_ground':True,'completed_round_trips':summaries},indent=2))
if __name__=='__main__':main()
