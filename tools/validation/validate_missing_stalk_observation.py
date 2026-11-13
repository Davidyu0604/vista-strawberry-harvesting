"""Archived seed 2 evidence and full-scene observation recovery; no model/video."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse,json,sys,tempfile
from pathlib import Path
import numpy as np
from replay_source import replay_environment


def validate(episode,out,preview_dir=None):
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    rows=[json.loads(l) for l in (episode/'trajectory.jsonl').read_text().splitlines()]
    env,source=replay_environment(episode,json.loads((episode/'run-config.json').read_text()),json.loads((episode/'report.json').read_text()))
    sys.path.insert(0,str(PROJECT_ROOT / 'src'))
    from vista import Session
    from visual_proposals import reviewed_candidates
    from PIL import Image
    survey=json.loads(next(e['message']['tool_calls'][0]['function']['arguments'] for e in events if e['kind']=='response' and e.get('call')==3))
    frame=survey['frame_id'];observation=next(e for e in events if e['kind']=='observation' and e['frame_id']==frame)
    state={k:observation[k] for k in ('joints_rad','tcp_world_m','base_x_y_yaw')}
    batch={k:v for k,v in next(e for e in events if e['kind']=='body_proposals' and e['frame_id']==frame).items() if k not in ('kind','frame_id','sim_time')}
    max_error=0.;trace=[]
    try:
        for row in rows[:131]:
            error=max(float(np.max(abs(env.state()-row['state']))),float(np.max(abs(env.base_pose()-row['base_pose']))));max_error=max(max_error,error)
            if error>1e-6:raise ValueError('Original prefix differs')
            env.drive(*row['base_command']);env.step(row.get('requested_action',row['action']))
        with tempfile.TemporaryDirectory(prefix='missing-stalk-') as tmp:
            class Recorder:
                def __init__(self):self.out=Path(tmp)
                def tick(self,q,phase):env.step(q);trace.append({'phase':phase,'base_command':env.drive_command.tolist()})
            s=Session(env,Recorder(),None,centered_grasp=True,visual_proposals=True);s.calls=4
            with Image.open(episode/'observations'/f'{frame}-harvest_close.jpg') as rgb:
                measured,blocked=reviewed_candidates(survey,batch,frame,rgb)
            local=s.local_targets;local.register(measured,state['base_x_y_yaw'],survey['coverage'])
            local.selection_blocked_reason=blocked;local.survey_review=survey;local.survey_batch=batch;local.survey_state=state
            local.survey_reference_depth=episode/'observations'/f'{frame}-harvest_close-depth.npz'
            s.station_attempts=4;key=local.active['candidate_key'];base=env.base_pose().copy()
            f,current,message=s.observe();before=s.stalk_view_evidence
            assert 'adjust_observation' in json.loads(message['content'][0]['text'])['station_search']['next_action']
            result,_=s.dispatch('adjust_observation',{'basis_frame_id':f,'candidate_key':key,'reason':'Offline: both fresh measured views lack a fitted stalk; look again at the same candidate'},f,current,0)
            after_frame,after_state,_=s.observe();after=s.stalk_view_evidence
            proposals=list(s.stalk_proposals.values())
            if preview_dir is not None:
                preview_dir.mkdir(parents=True,exist_ok=True)
                for camera in ('harvest_close','wrist'):
                    info,image=s.active_candidate_crop(after_frame,after_state,camera)
                    if image is not None:image.save(preview_dir/f'{camera}.jpg')
            report={'source_episode':str(episode),'physics_source':source,'prefix_error':max_error,'before':before,'adjustment':result,'after':after,'after_stalk_proposals':proposals,'next_action':s.station_status()['next_action'],'base_displacement_m':float(np.linalg.norm(env.base_pose()[:2]-base[:2])),'all_attached_offline':bool(env.attached.all()),'warnings':[int(w.number) for w in env.data.warning],'model_calls':0,'new_videos':0,'online_success_claimed':False,'trace':trace}
            assert local.active['candidate_key']==key and local.active['failures']==0
            assert not s.target_confirmed and report['all_attached_offline'] and not any(report['warnings']) and report['base_displacement_m']<.005
            out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print(json.dumps({k:v for k,v in report.items() if k not in ('trace','adjustment','after_stalk_proposals')},ensure_ascii=False),flush=True)
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--preview-dir',type=Path);a=p.parse_args();validate(a.episode.resolve(),a.out.resolve(),a.preview_dir)
