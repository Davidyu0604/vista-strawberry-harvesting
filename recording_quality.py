"""Offline integrity checks for raw recordings, including interrupted episodes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image


def atomic_json(path, value):
    import os
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def phase_labels(phase):
    if phase.startswith('search_'):return 'search',False,False
    if phase=='model_wait':return 'waiting',True,False
    if phase in ('decision_settle','terminal_settle'):return 'settling',False,False
    if phase.startswith('approach_retreat'):return 'recovery',False,True
    if phase.startswith('observation_adjust'):return 'recovery',False,True
    if phase in ('visual_pregrasp','pregrasp'):return 'pregrasp',False,False
    if phase in ('visual_grasp','approach'):return 'approach',False,False
    if phase in ('gripper','grasp','grasp_contact_check'):return 'gripper',False,False
    if phase in ('bend_pull','pull') or phase.startswith('bend_pull_'):return 'detach',False,False
    if phase.startswith(('delivery_return','delivery_restore_observation')):return 'return',False,False
    if phase.startswith('delivery_release'):return 'release',False,False
    if phase.startswith(('delivery_above','delivery_approach','delivery_lower')):return 'place',False,False
    if phase.startswith(('delivery_clear','delivery_stow')):return 'post_release',False,False
    if phase=='delivery_carry':return 'withdraw',False,False
    if phase.startswith('delivery_'):return 'transport',False,False
    return 'other',False,False


def audit_recording(directory, *, write_report=True):
    directory=Path(directory);errors=[];rows=[];decoded=0
    try:meta=json.loads((directory/'recording.json').read_text())
    except (OSError,ValueError):meta={};errors.append('Missing or invalid recording.json')
    try:
        with (directory/'trajectory.jsonl').open() as f:
            for i,line in enumerate(f,1):
                try:
                    parsed=json.loads(line)
                    if not isinstance(parsed,dict):raise ValueError('Row must be an object')
                    rows.append(parsed)
                except ValueError:errors.append(f'Truncated/invalid trajectory line {i}')
    except OSError:errors.append('Missing trajectory.jsonl')
    expected=meta.get('cameras',[]);shape=tuple(meta.get('image_size',[]));dt=meta.get('control_dt',.05)
    previous=None
    for i,row in enumerate(rows):
        try:
            for key,n in [('state',6),('action',6),('requested_action',6),('base_pose',3),('base_command',2)]:
                arr=np.asarray(row[key],dtype=float)
                if arr.shape!=(n,) or not np.isfinite(arr).all():raise ValueError(f'invalid {key}')
            if not np.isfinite(row['timestamp']) or not np.isclose(row['next_timestamp']-row['timestamp'],dt,atol=1e-8):raise ValueError('invalid control interval')
            if row['next_frame']!=row['frame']+1:raise ValueError('invalid next_frame')
            if previous is not None:
                if row['frame']!=previous['frame']+1 or not np.isclose(row['timestamp']-previous['timestamp'],dt,atol=1e-8):raise ValueError('frame/time discontinuity')
                if np.max(np.abs(np.asarray(row['action'])-previous['action']))>.0400001:raise ValueError('joint command slew exceeds limit')
            limits=np.asarray(meta['joint_control_limits'])
            if np.any(np.asarray(row['action'])<limits[:,0]-1e-8) or np.any(np.asarray(row['action'])>limits[:,1]+1e-8):raise ValueError('joint command outside limits')
            if abs(row['base_command'][0])>.15 or abs(row['base_command'][1])>.5:raise ValueError('base command outside limits')
            stage,wait,recovery=phase_labels(row['phase'])
            if (row['stage'],row['is_model_wait'],row['is_recovery'])!=(stage,wait,recovery):raise ValueError('incorrect phase labels')
            if set(row['images'])!=set(expected):raise ValueError('missing/extra camera')
            for camera,relative in row['images'].items():
                path=(directory/relative).resolve()
                if not path.is_relative_to(directory.resolve()):raise ValueError('image path outside recording')
                if row['image_timestamps'][camera]!=row['timestamp']:raise ValueError('camera timestamp mismatch')
                with Image.open(path) as im:
                    im.load()
                    if im.size!=shape or im.mode!='RGB':raise ValueError('image size/mode mismatch')
                decoded+=1
        except (KeyError,ValueError,TypeError,OSError,IndexError) as exc:errors.append(f'row {i}: {exc}')
        previous=row
    if not rows:errors.append('Empty recording')
    if (directory/'pending-step.json').exists():errors.append('Uncommitted step; inspect pending-step.json')
    try:report=json.loads((directory/'report.json').read_text())
    except (OSError,ValueError):report={}
    finalized=meta.get('status')=='finalized'
    if finalized and report.get('frames')!=len(rows):errors.append('Report frame count mismatch')
    phases=Counter(row.get('stage','missing') for row in rows)
    complete=finalized and report.get('episode_completed',False) and report.get('stop_reason')=='model_finish' and not report.get('error')
    three_cameras=set(expected)=={'top','wrist','harvest_close'}
    two_cameras=set(expected)=={'harvest_close','wrist'}
    camera_layout_valid=(two_cameras if meta.get('basket_mount')=='chassis' and meta.get('schema_version')==3 else three_cameras)
    behaviour_review={}
    review_path=directory/'behaviour-review.json'
    if review_path.exists():
        try:
            behaviour_review=json.loads(review_path.read_text())
            if not isinstance(behaviour_review,dict):raise ValueError('Expected review object')
        except (OSError,ValueError):
            behaviour_review={'training_approved':False,'reason':'Invalid behaviour-review.json'}
    behaviour_blocked=behaviour_review.get('training_approved') is False
    result={'integrity_passed':not errors,'errors':errors[:100],'error_count':len(errors),
            'committed_frames':len(rows),'decoded_images':decoded,'recording_finalized':finalized,
            'episode_completed':bool(complete),'harvest_success':bool(report.get('success',False)),
            'three_camera_recording':three_cameras,'two_camera_recording':two_cameras,
            'camera_layout_valid':camera_layout_valid,'basket_mount':meta.get('basket_mount','ground'),'stage_counts':dict(phases),
            'behaviour_review_blocks_training':behaviour_blocked,
            'model_wait_frames':sum(bool(r.get('is_model_wait')) for r in rows),
            'recovery_frames':sum(bool(r.get('is_recovery')) for r in rows),
            'eligible_for_training_export_review':bool(not errors and complete and camera_layout_valid and report.get('success') and not behaviour_blocked),
            'note':'Raw integrity only; not ACT format/training certification. Preserve raw waits; curate exports separately. Privileged diagnostic fields must not enter policy inputs.'}
    if write_report:atomic_json(directory/'quality.json',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    print(json.dumps(audit_recording(parser.parse_args().directory),indent=2,ensure_ascii=False))
