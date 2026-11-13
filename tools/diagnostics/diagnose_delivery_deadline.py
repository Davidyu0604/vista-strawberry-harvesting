"""Replay original commands without changing motion or requesting a model."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
from replay_source import replay_environment
import argparse
parser=argparse.ArgumentParser(description='Read-only recorded delivery deadline diagnosis')
parser.add_argument('episode',type=Path);parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();b=args.episode.resolve()
r=json.loads((b/'report.json').read_text());cfg=json.loads((b/'run-config.json').read_text());rows=[json.loads(l) for l in (b/'trajectory.jsonl').read_text().splitlines()]
e,source=replay_environment(b,cfg,r)
from centered_grasp import CenteredGrasp
from mounted_basket import chassis_point
c=CenteredGrasp(e,None);errors=[];trace=[];side=None;q=None
try:
 for row in rows:
  err=max(np.max(abs(e.state()-row['state'])),np.max(abs(e.base_pose()-row['base_pose'])));errors.append(float(err));assert err<1e-6
  if row['phase']=='delivery_aisle_turn' and side is None:
   side=chassis_point(e,[.12,-.145,.33]);c.carry_roll=float(e.last_action[4]);q,_=c.ik(side,-.15,pitch=-.6);print('TURN_START',row['frame'],'goal',q,'start',e.last_action,flush=True)
  e.drive(*row['base_command']);e.step(row.get('requested_action',row['action']))
  if side is not None:
   trace.append({'frame':e.frame,'goal_joint_gap':float(max(abs(q-e.last_action))),'actual_goal_error_m':float(np.linalg.norm(e.data.site_xpos[e.tcp]-side)),'command_goal_error_m':float(np.linalg.norm(c.fk(e.last_action)-side)),'tracking_error_m':float(np.linalg.norm(e.data.site_xpos[e.tcp]-c.fk(e.last_action))),'held_stalks_offline':np.flatnonzero(e.bilateral_contacts()).tolist(),'detached_ids_offline':np.flatnonzero(~e.attached).tolist()})
 out={'max_prefix_state_error':max(errors),'trace':trace,'side':side.tolist(),'warnings':[int(w.number) for w in e.data.warning]}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2))
 print('FINAL',trace[-1],flush=True)
finally:e.close()
