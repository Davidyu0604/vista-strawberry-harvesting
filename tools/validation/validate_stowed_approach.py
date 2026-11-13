"""Check a recorded visual target's pregrasp path from the new stowed pose."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'src'))
import argparse,json,tempfile
import numpy as np
from peduncle import PeduncleEnv
from run import Recorder
from vista import Session
from mounted_basket import ready_point


def validate(episode,out):
    cfg=json.loads((episode/'run-config.json').read_text())
    events=[json.loads(line) for line in (episode/'vista-events.jsonl').read_text().splitlines()]
    target=next(e['result'] for e in events if e['kind']=='receipt' and e.get('call')==9)
    assert target['target_confirmed'] and target['maturity_check']['passed']
    e=PeduncleEnv(cfg['seed'],single=False,mobile_layout=True,render_quality='fast',initial_arm_pose='search_stow')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            s=Session(e,Recorder(e,Path(tmp)/'episode',False,False),None,centered_grasp=True,visual_proposals=True)
            for _ in range(10):s.tick(e.last_action,'decision_settle')
            s.local_targets.register([],e.base_pose(),'Offline mechanical fixture only')
            s.search_row(1)
            point=target['locked_target_m'];axis=target['stalk_fit']['axis_world']
            pregrasp=s.centered.approach('pregrasp',point,axis)
            s.centered.retreat();restored=s.centered.restore_observation()
            assert np.linalg.norm(e.data.site_xpos[e.tcp]-ready_point(e))<.008
            assert e.attached.all() and not any(w.number for w in e.data.warning)
            result={'seed':cfg['seed'],'source_episode':str(episode),'recorded_visual_target_m':point,
                    'pregrasp':pregrasp,'return':restored,'model_calls':0,'all_attached_offline':True,
                    'warnings':0,'online_success_validated':False,
                    'note':'Offline path check using a previously recorded RGB-D target; no new visual approval or grasp performed.'}
            out.write_text(json.dumps(result,indent=2)+'\n');print('Stowed pregrasp and retreat passed.',flush=True)
    finally:e.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();validate(a.episode.resolve(),a.out.resolve())
