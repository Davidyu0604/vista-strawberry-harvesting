"""Read original seed23 RGB-D and validate nonmoving missing-body feedback."""
import sys,json,tempfile
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'src'))
from vista import Session
from perception_recovery import MISSING_BLOCK
root=Path(__file__).resolve().parents[2]
p=root/'reports/02_批量评估/fresh50-v1/seeds-16-25-20261112-134549/batch-2/seed-23'
ev=[json.loads(l) for l in (p/'vista-events.jsonl').read_text().splitlines()]
frame='f0000565';obs=next(e for e in ev if e['kind']=='observation' and e['frame_id']==frame)
args=json.loads(next(e for e in ev if e['kind']=='response' and e['call']==9)['message']['tool_calls'][0]['function']['arguments'])
with tempfile.TemporaryDirectory() as tmp:
 env=SimpleNamespace(meta={'tilted_stalk_prior':True},frame=565,drive=Mock())
 s=Session(env,SimpleNamespace(out=Path(tmp)),None,visual_proposals=True)
 local=s.local_targets;local.surveyed=True;local.candidates=deepcopy(obs['local_selection']['candidates'])
 local.selection_blocked_reason=MISSING_BLOCK;local.survey_review={'missing_visible_ripe':True}
 local.survey_batch=next(e for e in reversed(ev) if e['kind']=='body_proposals' and int(e['frame_id'][1:])<=565)
 local.survey_correction_attempts=1;before=deepcopy(local.candidates)
 s.calls=9;s.station_attempts=3
 s.frames={frame:{'harvest_close':p/f'observations/{frame}-harvest_close.jpg'}}
 s.depth_frames={(frame,'harvest_close'):p/f'observations/{frame}-harvest_close-depth.npz'}
 with patch('vista.robot_state',return_value=obs):
  result,images=s.dispatch('inspect_unlisted',{'basis_frame_id':frame,'bodies':args['bodies']},frame,obs,0)
 assert not result['inspection']['checks'][0]['body_pixel_check']['passed']
 assert local.candidates==before and local.selection_blocked_reason==MISSING_BLOCK and not local.can_search
 assert local.survey_correction_attempts==1 and local.unlisted_inspections==1 and s.station_attempts==4
 env.drive.assert_not_called()
 output={'source_episode':str(p),'frame_id':frame,'original_request':args,'new_inspection':result,'feedback_image_count':sum(i['type']=='image_url' for i in images),'queue_preserved':True,'coverage_block_preserved':True,'motion_executed':False,'model_calls':0,'online_success_validated':False,'note':'Original reported (310,300) has RGB [19,24,18], not a coherent red body. This does not prove absence elsewhere; requires a new model review of marked/full images. No speculative correction applied.'}
 (root/'reports/01_控制验证/search-evidence-recovery-v1/inspection-seed23.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result['inspection']['checks'][0],ensure_ascii=False))
