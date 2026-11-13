"""Replay one recorded identity-recovery prefix; seed argument 16/18/19/25. No model calls."""
import sys,json,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'src'))
from replay_source import replay_environment
root=Path(__file__).resolve().parents[2];seed=int(sys.argv[1]);p=next((root/'reports/02_批量评估/fresh50-v1/seeds-16-25-20261112-134549').glob(f'batch-*/seed-{seed}'))
ev=[json.loads(l) for l in (p/'vista-events.jsonl').read_text().splitlines()];r=json.loads((p/'report.json').read_text());cfg=json.loads((p/'run-config.json').read_text())
adjust=next(x for x in r['perception_recoveries'] if x['kind']=='observation_adjust');prefix=adjust['start_frame'];obs=[e for e in ev if e['kind']=='observation' and int(e['frame_id'][1:])<=prefix][-1]
response=next(e for e in ev if e['kind']=='response' and e['message']['tool_calls'][0]['function']['name']=='adjust_observation');args=json.loads(response['message']['tool_calls'][0]['function']['arguments'])
env,source=replay_environment(p,cfg,r)
from vista import Session
from copy import deepcopy
try:
 rows=[json.loads(l) for l in (p/'trajectory.jsonl').read_text().splitlines()];error=0
 for row in rows[:prefix]:
  err=max(np.max(abs(env.state()-row['state'])),np.max(abs(env.base_pose()-row['base_pose'])));error=max(error,float(err));assert err<1e-9
  env.drive(*row['base_command']);env.step(row['requested_action'])
 with tempfile.TemporaryDirectory() as tmp:
  class Recorder:
   out=Path(tmp)
   def tick(self,q,phase):env.step(q)
  s=Session(env,Recorder(),None,centered_grasp=True,visual_proposals=True);s.calls=response['call'];s.station_attempts=response['call']-1
  local=s.local_targets;local.surveyed=True;local.candidates=deepcopy(obs['local_selection']['candidates']);local.coverage=obs['local_selection']['coverage']
  first=next(e for e in ev if e['kind']=='observation');local.survey_state=first;local.survey_review={'frame_id':first['frame_id']};local.survey_reference_depth=p/f"observations/{first['frame_id']}-harvest_close-depth.npz"
  local.survey_batch={k:v for k,v in next(e for e in ev if e['kind']=='body_proposals').items() if k not in ('kind','frame_id','sim_time')}
  frame,state,_=s.observe();args['basis_frame_id']=frame
  try:
   moved,_=s.dispatch('adjust_observation',args,frame,state,0)
   frame,state,_=s.observe();fits=list(s.stalk_proposals.values())
   output={'seed':seed,'prefix_frames':prefix,'prefix_error':error,'adjustment':moved,'new_frame':frame,'view_evidence':s.stalk_view_evidence,'fits':[{k:v for k,v in f.items() if k!='state'} for f in fits],'attached':bool(env.attached.all()),'warnings':sum(w.number for w in env.data.warning)}
   if seed==18:
    if fits:
     chosen=fits[0]
     try:
      located,_=s.dispatch('locate_stalk_proposal',{'frame_id':frame,'candidate_key':local.active['candidate_key'],'proposal_id':chosen['proposal_id'],'confirmed_short_stalk':True,'reason':'Offline diagnostic fixture: inspect the new wrist fit, not an online model decision'},frame,state,0)
      output['offline_fit_dispatch']=located
     except Exception as exc:output['offline_fit_dispatch_error']=str(exc)

  except Exception as e:output={'seed':seed,'prefix_error':error,'error':str(e),'plan':getattr(s.centered,'observation_plan_diagnostics',None)}
  output.update(model_calls=0,online_success_validated=False,note='Original recorded prefix; scripted observation and optional fit hypothesis, no online semantic judgment or harvest success claim.')
  dest=root/f'reports/01_控制验证/search-evidence-recovery-v1/identity-framing-seed{seed}.json';dest.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
  print(json.dumps(output,ensure_ascii=False)[:3500],flush=True)
finally:env.close()
