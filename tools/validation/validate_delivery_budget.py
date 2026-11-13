"""Offline delivery regression; no model calls, images or dataset export."""
import sys,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
from replay_source import replay_environment
import argparse
parser=argparse.ArgumentParser(description='Replay recorded pre-delivery physics, then current delivery without API or images')
parser.add_argument('episode',type=Path);parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--stowed-return',action='store_true',help='Offline counterfactual: deliver from recorded held state, return to the new stowed observation pose')
args=parser.parse_args();root=Path(__file__).resolve().parents[2];b=args.episode.resolve()
seed=json.loads((b/'run-config.json').read_text())['seed']
r=json.loads((b/'report.json').read_text());cfg=json.loads((b/'run-config.json').read_text());rows=[json.loads(l) for l in (b/'trajectory.jsonl').read_text().splitlines()]
first=next(i for i,x in enumerate(rows) if x['phase']=='delivery_carry');prefix=first-10
assert all(x['phase']=='grasp_contact_check' for x in rows[prefix:first])
e,source=replay_environment(b,cfg,r)
stowed_home=None;stowed_roll=None
if args.stowed_return:
 stowed_home=np.array([.19,e.meta['initial_base_pose'][1]-.04,.34])
 q,ik_error=e.ik(stowed_home,1.,start=[0,-.3,.45,.25,0,1.])
 assert ik_error<.006
 stowed_roll=float(q[4])
sys.path.insert(0,str(root / 'src'))
from vista import Session,evaluate,jaw_contact_state
from centered_grasp import CenteredGrasp
trace=[];err=0
try:
 for row in rows[:prefix]:
  drift=max(float(np.max(abs(e.state()-row['state']))),float(np.max(abs(e.base_pose()-row['base_pose']))));err=max(err,drift);assert drift<1e-6
  e.drive(*row['base_command']);e.step(row.get('requested_action',row['action']))
 with tempfile.TemporaryDirectory() as tmp:
  class Recorder:
   out=Path(tmp)
   def tick(self,q,phase):
    e.step(q);trace.append({'frame':e.frame,'phase':phase,'base_pose':e.base_pose().tolist(),'base_command':e.drive_command.tolist(),'held_stalks_offline':np.flatnonzero(e.bilateral_contacts()).tolist(),'detached_ids_offline':np.flatnonzero(~e.attached).tolist()})
  s=Session(e,Recorder(),None,centered_grasp=True,visual_proposals=True)
  s.grasp_attempted=True;s.target_stage='grasp';s.centered.carry_roll=float(e.last_action[4]);s.centered.observation_roll=r['observation_roll_rad'];e.meta['observation_roll_rad']=r['observation_roll_rad']
  if args.stowed_return:
   e.meta['initial_tcp_world']=stowed_home.tolist();e.meta['observation_roll_rad']=stowed_roll
   s.centered.observation_roll=stowed_roll
  result=None;error=None
  try:result=s.deliver_to_basket()
  except Exception as exc:error=str(exc)
  print('DELIVERY_RESULT',seed,error,flush=True)
  out={'seed':seed,'prefix_error':err,'prefix_frames':prefix,'result':result,'error':error,'stop':s.motion_stop_trace,'evaluation':evaluate(e,trace),'trace':trace,'warnings':[int(w.number) for w in e.data.warning],'model_calls':0,
       'stowed_return_counterfactual':args.stowed_return,'online_success_validated':False}
  p=args.out;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2))
  print(json.dumps({k:v for k,v in out.items() if k not in ['trace','result']},ensure_ascii=False),flush=True)
  assert out['evaluation']['success'],out['evaluation']
  assert error is None and not any(out['warnings'])
  held=[x for x in trace if x['phase'] in ['grasp_contact_check','delivery_carry','delivery_aisle_turn','delivery_above','delivery_approach','delivery_lower']]
  target=out['evaluation']['ripe_fruit_in_basket'][0]
  assert all(target in x['held_stalks_offline'] for x in held)
  print('ALL_CHECKS_PASSED',seed,flush=True)
finally:e.close()
