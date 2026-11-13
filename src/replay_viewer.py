"""Replay recorded controls in a GUI, without calling the visual policy."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

from replay_source import replay_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--skip-initial-wait', action='store_true',
                        help='Fast-forward initial settling/waiting; preserve physics steps')
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be positive and finite')
    episode = args.episode.resolve()
    config = json.loads((episode / 'run-config.json').read_text())
    report = json.loads((episode / 'report.json').read_text())
    rows = [json.loads(line) for line in (episode / 'trajectory.jsonl').read_text().splitlines()]
    start_frame = next((r['frame'] for r in rows if r['stage'] not in ('settling', 'waiting')), 0) if args.skip_initial_wait else 0
    env, physics_source = replay_environment(episode, config, report)
    max_joint_error = max_base_error = max_tcp_error = 0.
    played = 0
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.opt.geomgroup[3] = 0
            viewer.cam.lookat[:] = [.18, 0., .23]
            viewer.cam.distance = .95
            viewer.cam.azimuth = 145
            viewer.cam.elevation = -25
            viewer.sync()
            print('REPLAY started; no model calls. Close window to stop. Do not press Reset/Backspace.', flush=True)
            phase = None
            for row in rows:
                if not viewer.is_running():
                    break
                started = time.monotonic()
                joint_error = float(np.max(np.abs(env.state() - row['state'])))
                base_error = float(np.max(np.abs(env.base_pose() - row['base_pose'])))
                tcp_error = float(np.linalg.norm(env.data.site_xpos[env.tcp] - row['tcp_position']))
                max_joint_error = max(max_joint_error, joint_error)
                max_base_error = max(max_base_error, base_error)
                max_tcp_error = max(max_tcp_error, tcp_error)
                if max(joint_error, base_error, tcp_error) > 1e-6:
                    raise RuntimeError(f'Replay state differs at frame {row["frame"]}; stopped')
                if row['stage'] != phase:
                    phase = row['stage']
                    print(f'{row["timestamp"]:6.2f}s | {phase}', flush=True)
                env.drive(*row['base_command'])
                env.step(row.get('requested_action', row['action']))
                played += 1
                if row['frame'] >= start_frame or played % 20 == 0:
                    viewer.sync()
                if row['frame'] >= start_frame:
                    time.sleep(max(0., env.dt / args.speed - (time.monotonic() - started)))
            result = dict(source_episode=str(episode), model_calls=0, played_frames=played,
                          completed=played == len(rows), max_joint_error_rad=max_joint_error,
                          initial_wait_fast_forwarded=args.skip_initial_wait,
                          max_base_pose_error=max_base_error, max_tcp_error_m=max_tcp_error,
                          physics_source_hashes_verified=True, physics_source=physics_source, events=env.events,
                          recorded_success=report['success'])
            output = episode / f'viewer-replay-{time.time_ns()}.json'
            output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result), flush=True)
            print('REPLAY finished. Window remains open; close it when done.', flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(.05)
    finally:
        env.close()


if __name__ == '__main__':
    main()
