"""Offline production Session integration check, no model calls or replay media."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from replay_source import replay_environment


def validate(episode,out):
    if out.exists():raise FileExistsError(out)
    config=json.loads((episode/'run-config.json').read_text());report=json.loads((episode/'report.json').read_text())
    rows=[json.loads(l) for l in (episode/'trajectory.jsonl').read_text().splitlines()]
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    pre=next(r['frame'] for r in rows if r['phase']=='visual_pregrasp')
    locks=[x['result'] for x in events if x['kind']=='receipt' and isinstance(x.get('result'),dict)
           and x['result'].get('target_confirmed') and x['result'].get('locked_target_m')
           and int(x['result']['frame_id'][1:])<=pre]
    lock=locks[-1];point=np.array(lock['locked_target_m']);axis=np.array(lock['stalk_fit']['axis_world'])
    e,source=replay_environment(episode,config,report)
    sys.path.insert(0,str(PROJECT_ROOT / 'src'))
    from vista import Session,evaluate,jaw_contact_state
    trace=[];max_error=0.
    class Recorder:
        def __init__(self):self.out=out.parent
        def tick(self,q,phase):
            old=e.last_action.copy();e.step(q)
            trace.append({'frame':e.frame,'phase':phase,'base_pose':e.base_pose().tolist(),
                'joints':e.state().tolist(),'command':e.last_action.tolist(),
                'command_step_rad':float(np.max(np.abs(e.last_action-old))),
                'tcp_m':e.data.site_xpos[e.tcp].tolist(),'external_contact':jaw_contact_state(e),
                'stalk_bilateral_ids_offline':np.flatnonzero(e.bilateral_contacts()).tolist(),
                'detached_ids_offline':np.flatnonzero(~e.attached).tolist()})
    try:
        for row in rows[:pre]:
            error=max(float(np.max(np.abs(e.state()-row['state']))),float(np.max(np.abs(e.base_pose()-row['base_pose']))),
                      float(np.linalg.norm(e.data.site_xpos[e.tcp]-row['tcp_position'])))
            max_error=max(max_error,error)
            if error>1e-6:raise ValueError('Original prefix differs')
            e.drive(*row['base_command']);e.step(row.get('requested_action',row['action']))
        cameras={name:{'position':e.data.cam_xpos[e.model.camera(name).id].copy(),
                       'rotation':e.data.cam_xmat[e.model.camera(name).id].copy()} for name in ('harvest_close','wrist')}
        # Read-only audit association, never passed to Session.
        target=int(np.argmin(np.linalg.norm(e.data.site_xpos[e.grasp_sites]-point,axis=1)))
        s=Session(e,Recorder(),None,centered_grasp=True);s.local_targets=None
        s.locked_target=point;s.target_confirmed=True
        s.maturity_evidence={'passed':True,'target_m':point.tolist(),'axis_world':axis.tolist()}
        result={};failure=None
        try:
            result['pregrasp']=s.execute_approach('pregrasp')
            for _ in range(20):s.tick(e.last_action,'decision_settle')
            result['grasp']=s.execute_approach('grasp')
            for _ in range(20):s.tick(e.last_action,'decision_settle')
            result['close']=s.execute([{'action':'gripper','opening':-.15,'seconds':3}])
            if result['close'][0]['status']!='executed':raise ValueError(str(result['close']))
            for _ in range(20):s.tick(e.last_action,'decision_settle')
            result['pull']=s.execute([{'action':'bend_pull','dx':-.035,'dy':0,'dz':0,'pitch_deg':-37,'seconds':3}])
            if result['pull'][0]['status']!='executed':raise ValueError(str(result['pull']))
            result['delivery']=s.deliver_to_basket()
        except (ValueError,RuntimeError) as exc:failure=str(exc)
        camera_check={}
        for name,original in cameras.items():
            idx=e.model.camera(name).id
            rotation=e.data.cam_xmat[idx].reshape(3,3)@original['rotation'].reshape(3,3).T
            rgb=e.render(name,480,360)
            camera_check[name]={'render_shape':list(rgb.shape),'rgb_std':float(np.std(rgb)),
                'position_change_mm':float(np.linalg.norm(e.data.cam_xpos[idx]-original['position'])*1000),
                'rotation_change_deg':float(np.rad2deg(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1))))}
        evaluation=evaluate(e,trace)
        output={'episode':str(episode),'physics_source':source,'max_prefix_state_error':max_error,
                'model_calls':0,'new_images':0,'new_videos':0,'error':failure,'receipts':result,
                'evaluation':evaluation,'offline_associated_target':target,
                'same_target_in_basket':evaluation['ripe_fruit_in_basket']==[target],
                'camera_check':camera_check,'warnings':[int(w.number) for w in e.data.warning],'trace':trace}
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps({k:v for k,v in output.items() if k not in ('trace','receipts')}),flush=True)
    finally:e.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();validate(a.episode.resolve(),a.out)
