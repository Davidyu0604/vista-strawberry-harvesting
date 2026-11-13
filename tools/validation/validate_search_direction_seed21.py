"""Recorded seed21 prefix and vision-derived alternative search; offline only."""
import sys,json,tempfile
from pathlib import Path
from copy import deepcopy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from replay_source import replay_environment
root=Path(__file__).resolve().parents[2];p=root/'reports/02_批量评估/fresh50-v1/seeds-16-25-20261112-134549/batch-2/seed-21'
read=lambda f:json.loads(f.read_text());report=read(p/'report.json');cfg=read(p/'run-config.json');ev=[json.loads(l) for l in (p/'vista-events.jsonl').read_text().splitlines()];rows=[json.loads(l) for l in (p/'trajectory.jsonl').read_text().splitlines()]
prefix=next(r['frame'] for r in rows if r['phase']=='search_drive');obs=[e for e in ev if e['kind']=='observation' and int(e['frame_id'][1:])<=prefix][-1];loc=next(e['result'] for e in ev if e['kind']=='receipt' and e['call']==2)
env,source=replay_environment(p,cfg,report)
from vista import Session,tilted_stalk_grasp_point,verify_projected_point
from stalk_proposals import propose_stalk_points
from PIL import Image
try:
 error=0
 for row in rows[:prefix]:
  drift=max(np.max(abs(env.state()-row['state'])),np.max(abs(env.base_pose()-row['base_pose'])));error=max(error,float(drift));assert drift<1e-9
  env.drive(*row['base_command']);env.step(row['requested_action'])
 with tempfile.TemporaryDirectory() as tmp:
  class Recorder:
   out=Path(tmp)
   def tick(self,q,phase):env.step(q)
  s=Session(env,Recorder(),None,centered_grasp=True,visual_proposals=True);s.calls=3;s.station_attempts=2
  local=s.local_targets;local.surveyed=True;local.candidates=deepcopy(obs['local_selection']['candidates']);local.coverage=obs['local_selection']['coverage']
  s.visual_search_hints=[dict(frame_id='f0000048',candidate_key='view-candidate-3',measured_stalk_m=loc['locked_target_m'],source='Original confirmed RGB-D stalk and maturity evidence, unreachable current parking')]
  frame,state,message=s.observe();guidance=json.loads(message['content'][0]['text'])['row_search_guidance']
  assert guidance['previously_confirmed_ripe_but_unreachable_here'][0]['toward_measured_target_direction']==1
  before=env.base_pose().copy();moved,_=s.dispatch('search_row',{'basis_frame_id':frame,'direction':1},frame,state,0)
  frame,state,_=s.observe();assert not s.local_targets.surveyed and not s.target_confirmed
  body=next(c['surface_m'] for c in obs['local_selection']['candidates'] if c['proposal_id']=='P3')
  fits=[]
  for camera in ['harvest_close','wrist']:
   with Image.open(s.frames[frame][camera]) as rgb,np.load(s.depth_frames[(frame,camera)]) as d:
    batch=propose_stalk_points(rgb.convert('RGB'),d['depth'],d,body,tilted_stalk_grasp_point)
   for fit in batch['proposals']:
    target=np.array(fit['target_m']);item={'camera':camera,'fit':fit}
    maturity,_=s.check_maturity(frame,target,np.array(fit['axis_world']));item['maturity']=maturity
    other='wrist' if camera=='harvest_close' else 'harvest_close'
    with np.load(s.depth_frames[(frame,other)]) as d:item['cross_view']=verify_projected_point(target,d['depth'],d)
    try:plan=s.centered.plan(target,np.array(fit['axis_world']));item['reachable']=True;item['max_ik_error_m']=plan['max_ik_error_m']
    except ValueError as exc:item['reachable']=False;item['reason']=str(exc)
    fits.append(item)
  output={'seed':21,'prefix_frames':prefix,'prefix_error':error,'original_search_directions':[-1]*4,'guidance':guidance,'offline_alternative_search':moved,'fresh_fit_checks':fits,'all_attached_offline':bool(env.attached.all()),'warnings':sum(w.number for w in env.data.warning),'model_calls':0,'online_success_validated':False,'note':'Scripted +Y alternative based only on original verified visual target. Fresh geometric fits are diagnostic hypotheses, not new model semantic decisions or harvest success.'}
  (root/'reports/01_控制验证/search-evidence-recovery-v1/direction-seed21.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps({'prefix_error':error,'guidance':guidance,'actual_move':moved['actual_delta_y_m'],'fits':[(f['camera'],f['maturity']['passed'],f['cross_view']['consistent'],f['reachable']) for f in fits]},ensure_ascii=False),flush=True)
finally:env.close()
