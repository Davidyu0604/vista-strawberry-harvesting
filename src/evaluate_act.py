"""Execute a trained ACT in MuJoCo; isolated scene, no vision-model service."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

if '--viewer' in sys.argv:
    os.environ['MUJOCO_GL'] = 'glfw'
    os.environ['PYOPENGL_PLATFORM'] = 'glx'

import numpy as np
from PIL import Image

from project_paths import PROJECT_ROOT as ROOT, project_source, source_files
CAMERAS = ('harvest_close', 'wrist')


def harvest_outcome(env, initial_roll):
    """Physical-only evaluator; never passes simulator fruit truth to the policy."""
    from mounted_basket import ready_point
    inside = []
    detached = np.flatnonzero(~env.attached).tolist()
    half = env.meta['basket_acceptance_half_xy_m']
    for i in detached:
        rel = env.data.body('basket').xmat.reshape(3, 3).T @ (env.fruit_pos(i) - env.basket_position())
        dof = env.fruit_dofs[i]
        if (env.meta['ripe'][i] and abs(rel[0]) < half[0] and abs(rel[1]) < half[1]
                and .012 < rel[2] < .05 and np.linalg.norm(env.data.qvel[dof:dof+3]) < .05
                and not env.contacts(i)):
            inside.append(i)
    returned = bool(np.linalg.norm(env.data.site_xpos[env.tcp] - ready_point(env)) < .012
                    and abs(env.state()[4] - initial_roll) < .015)
    stopped = bool(np.max(np.abs(env.data.qvel[env.base_da])) < .02)
    single = len(inside) == len(detached) == 1
    return {'ripe_fruit_in_basket': inside, 'detached_fruit_count': len(detached),
            'single_fruit_delivered': single, 'returned_to_ready': returned,
            'base_stopped': stopped, 'physical_success_now': bool(single and returned and stopped)}


def bounded_action(action, ranges):
    action = np.asarray(action, dtype=float)
    if action.shape != (8,) or not np.isfinite(action).all():
        raise ValueError('Expected eight finite action values')
    low = np.r_[ranges[:, 0], -.15, -.5]
    high = np.r_[ranges[:, 1], .15, .5]
    return np.clip(action, low, high)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'training/act-fresh60-v1/best.pt')
    parser.add_argument('--seed', type=int, default=61)
    parser.add_argument('--max-steps', type=int, default=3000)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--record-images', action='store_true')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--allow-diagnostic', action='store_true', help='Explicitly evaluate an experimental checkpoint')
    parser.add_argument('--n-action-steps', type=int, help='Execute this many predicted actions before replanning')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--policy-python', type=Path, default=ROOT.parent / '.venv-act-train/bin/python')
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error('--max-steps must be positive')
    if args.n_action_steps is not None and args.n_action_steps < 1:
        parser.error('--n-action-steps must be positive')
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file() or not args.policy_python.is_file():
        parser.error('Checkpoint or ACT Python not found')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from recording_quality import atomic_json
    worker = env = viewer = None
    completed_steps = consecutive_success = clipped_steps = 0
    outcome, error, stop = {}, None, 'max_steps'
    started = time.monotonic()
    worker_env = dict(os.environ, PYTHONPATH=str(ROOT / '.act-deps'), PYTHONNOUSERSITE='1',
                      HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      HF_HOME=str(ROOT / '.hf-cache'), TORCH_HOME=str(ROOT / '.torch-cache'))
    header = {}
    with tempfile.TemporaryDirectory(prefix='act-rollout-') as scratch, (out / 'inference.log').open('w') as log:
        try:
            worker_command = [str(args.policy_python), '-u', str(project_source('act_inference_worker.py')),
                              str(checkpoint), '--device', args.device]
            if args.allow_diagnostic:
                worker_command.append('--allow-diagnostic')
            if args.n_action_steps is not None:
                worker_command.extend(['--n-action-steps', str(args.n_action_steps)])
            worker = subprocess.Popen(worker_command,
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                      text=True, bufsize=1, env=worker_env)
            def response():
                line = worker.stdout.readline()
                if not line:
                    raise RuntimeError('ACT worker exited; see inference.log')
                result = json.loads(line)
                if 'error' in result:
                    raise RuntimeError(result['error'])
                return result
            header = response()
            if not header.get('ready'):
                raise RuntimeError('ACT worker did not initialize')
            print(f'[ACT] Loaded step {header["checkpoint_step"]}; device={args.device}; seed={args.seed}; '
                  f'n_action_steps={header["n_action_steps"]}; diagnostic={header["diagnostic"]}', flush=True)
            # Generate assets outside the working project; do not change training inputs.
            import scene
            import natural_assets
            scratch = Path(scratch)
            (scratch / 'assets/natural').mkdir(parents=True)
            (scratch / 'assets/so101').symlink_to(ROOT / 'assets/so101', target_is_directory=True)
            scene.ROOT = scratch
            natural_assets.ASSETS = scratch / 'assets/natural'
            from peduncle import PeduncleEnv
            env = PeduncleEnv(args.seed, single=False, fixed_base=False, mobile_layout=True,
                              mobile_layout_level='moderate', render_quality='fast', initial_arm_pose='search_stow')
            initial_roll = float(env.state()[4])
            initial_arm = env.state().copy()
            initial_base = env.base_pose().copy()
            run_config = {**header, 'seed': args.seed, 'max_steps': args.max_steps,
                          'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          'scene_version': env.meta['scene_version'], 'control_hz': 20,
                          'policy_inputs': ['6 joint positions + base x/y/yaw', *CAMERAS],
                          'image_preprocessing': 'JPEG quality 92; RGB /255; bilinear 240x320; ImageNet mean/std',
                          'render_backend': os.environ.get('MUJOCO_GL'),
                          'opengl_platform': os.environ.get('PYOPENGL_PLATFORM'),
                          'action_semantics': '6 absolute joint targets + base linear/angular velocity',
                          'actuator_bounds_clipped': True, 'joint_rate_limit_rad_per_tick': .04,
                          'success_rule': 'One ripe detached fruit settled in basket; return to ready and stop for 10 ticks; no scripted phase labels',
                          'no_model_service_calls': True, 'formal_training_demo': False}
            atomic_json(out / 'run-config.json', run_config)
            if args.viewer:
                import mujoco.viewer
                viewer = mujoco.viewer.launch_passive(env.model, env.data)
                viewer.opt.geomgroup[3] = 0
                viewer.cam.lookat[:] = [.06, initial_base[1], .27]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.55, 145, -28
                viewer.sync()
            with (out / 'trajectory.jsonl').open('w') as trajectory:
                for step in range(args.max_steps):
                    tick_started = time.monotonic()
                    if viewer is not None and not viewer.is_running():
                        stop = 'viewer_closed'
                        break
                    state = np.r_[env.state(), env.base_pose()].tolist()
                    tcp_position = env.data.site_xpos[env.tcp].tolist()
                    images, image_paths, image_hashes = {}, {}, {}
                    for camera in CAMERAS:
                        buffer = io.BytesIO()
                        Image.fromarray(env.render(camera, 480, 360)).save(buffer, format='JPEG', quality=92)
                        encoded = buffer.getvalue()
                        image_hashes[camera] = hashlib.sha256(encoded).hexdigest()
                        images[camera] = base64.b64encode(encoded).decode('ascii')
                        if args.record_images:
                            path = out / camera / f'{step:06d}.jpg'
                            path.parent.mkdir(exist_ok=True)
                            path.write_bytes(encoded)
                            image_paths[camera] = str(path.relative_to(out))
                    worker.stdin.write(json.dumps({'state': state, 'images': images}) + '\n')
                    worker.stdin.flush()
                    predicted = response()['action']
                    action = bounded_action(predicted, env.model.actuator_ctrlrange[:6])
                    clipped = not np.array_equal(action, predicted)
                    clipped_steps += int(clipped)
                    timestamp = float(env.data.time)
                    env.drive(*action[6:])
                    executed = env.step(action[:6])
                    completed_steps += 1
                    outcome = harvest_outcome(env, initial_roll)
                    # Evaluation only: contacts never enter the inference request.
                    bilateral_fruits = np.flatnonzero(env.bilateral_contacts()).tolist()
                    consecutive_success = consecutive_success + 1 if outcome['physical_success_now'] else 0
                    trajectory.write(json.dumps({'frame': step, 'timestamp': timestamp, 'state': state,
                        'tcp_position': tcp_position,
                        'predicted_action': predicted, 'action': np.r_[executed, action[6:]].tolist(),
                        'bounds_clipped': clipped, 'images': image_paths, 'evaluation': outcome,
                        'image_sha256': image_hashes,
                        'bilateral_contact_fruits_after_step': bilateral_fruits}) + '\n')
                    trajectory.flush()
                    if step % 50 == 0 or consecutive_success >= 10:
                        travel = np.linalg.norm(env.base_pose()[:2] - initial_base[:2])
                        arm_change = np.max(np.abs(env.state() - initial_arm))
                        target_delta = np.max(np.abs(np.asarray(predicted[:6]) - state[:6]))
                        print(f'[ACT] {step+1}/{args.max_steps} sim={env.data.time:.2f}s '
                              f'base_offset={travel:.6f}m arm_change={arm_change:.4f}rad '
                              f'cmd_v={action[6]:.3e}m/s cmd_w={action[7]:.3e}rad/s '
                              f'joint_target_delta={target_delta:.3e}rad '
                              f'detached={outcome["detached_fruit_count"]} '
                              f'in_basket={len(outcome["ripe_fruit_in_basket"])}', flush=True)
                    if viewer is not None:
                        viewer.sync()
                        time.sleep(max(0., env.dt - (time.monotonic() - tick_started)))
                    if consecutive_success >= 10:
                        stop = 'harvest_success'
                        break
        except KeyboardInterrupt:
            stop = 'interrupted'
        except Exception as exc:
            stop, error = 'runtime_error', f'{type(exc).__name__}: {exc}'
        finally:
            report = {'seed': args.seed, 'checkpoint_step': header.get('checkpoint_step'),
                      'diagnostic': header.get('diagnostic'), 'diagnostic_kind': header.get('diagnostic_kind'),
                      'training_episode_seed': header.get('training_episode_seed'),
                      'total_single_episode_steps': header.get('total_single_episode_steps'),
                      'n_action_steps': header.get('n_action_steps'),
                      'steps': completed_steps, 'stop_reason': stop, 'error': error,
                      'harvest_success': stop == 'harvest_success', 'evaluation': outcome,
                      'consecutive_success_ticks': consecutive_success, 'bounds_clipped_steps': clipped_steps,
                      'wall_seconds': time.monotonic() - started, 'model_service_calls': 0,
                      'policy': 'ACT closed-loop', 'formal_training_demo': False}
            if env is not None:
                report['events'] = env.events
                report['max_sampled_obstacle_penetration_m'] = env.worst_obstacle_penetration
            atomic_json(out / 'report.json', report)
            if viewer is not None:
                viewer.close()
            if env is not None:
                env.close()
            if worker is not None:
                if worker.stdin and not worker.stdin.closed:
                    worker.stdin.close()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    worker.wait(timeout=5)
            print(json.dumps({'out': str(out), **report}, ensure_ascii=False), flush=True)
    if error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
