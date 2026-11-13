"""Render presentation cameras in an isolated, hash-checked episode replay.

Uses CPU Mesa by default, no model API or training changes. Requires the original
recording; the two policy cameras are not changed. Run in a fresh Python process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
os.environ.setdefault('LP_NUM_THREADS', '2')
mesa = Path('/usr/share/glvnd/egl_vendor.d/50_mesa.json')
if mesa.exists():
    os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES', str(mesa))

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--verify-existing-video', action='store_true',
                        help='Replay without rendering and validate an existing derived video')
    args = parser.parse_args()
    episode = args.episode.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((episode / 'run-config.json').read_text())
    report = json.loads((episode / 'report.json').read_text())
    rows = [json.loads(s) for s in (episode / 'trajectory.jsonl').read_text().splitlines()]
    public_demo = json.loads((root / 'docs/evidence/demo.json').read_text())
    if sha(episode / 'trajectory.jsonl') != public_demo['source_sha256']['trajectory.jsonl']:
        raise ValueError('Episode must match the published dual-camera demo')
    sample_indices = public_demo['preview']['selected_source_row_indices']
    with tempfile.TemporaryDirectory(prefix='vista-presentation-') as scratch:
        scratch = Path(scratch)
        for path in (root / 'src').glob('*.py'):
            shutil.copy2(path, scratch / path.name)
        (scratch / 'assets/natural').mkdir(parents=True)
        (scratch / 'assets/so101').symlink_to(root / 'assets/so101', target_is_directory=True)
        # All imported scene modules and generated assets live in this scratch tree.
        sys.path.insert(0, str(scratch))
        from replay_source import replay_environment
        env, _ = replay_environment(episode, config, report, asset_root=scratch)
        if config.get('centered_grasp'):
            env.meta['centered_grasp'] = True
            env.meta['observation_roll_rad'] = float(env.last_action[4])
        writer = None
        try:
            base = env.base_pose()
            cameras = {
                'greenhouse-overview': camera([-.30, .85, 1.05], 5.7, 135, -24),
                'robot-third-person': camera([.06, base[1], .27], 1.25, 145, -25),
            }
            # Shadows/visible foliage only; physics and policy camera parameters unchanged.
            env.render_quality = 'full'
            if not args.verify_existing_video:
                for name, cam in cameras.items():
                    Image.fromarray(env.render(cam, 1280, 720)).save(out / (name + '.png'))
                print('Third-person stills rendered with software OpenGL.', flush=True)
            if not (args.video or args.verify_existing_video):
                return
            path = out / 'seed-060-third-person.mp4'
            if path.exists() and not args.verify_existing_video:
                raise FileExistsError(path)
            if args.verify_existing_video:
                if not path.exists():
                    raise FileNotFoundError(path)
            else:
                writer = imageio_ffmpeg.write_frames(str(path), (960, 588), fps=10,
                    codec='libx264', quality=None, macro_block_size=2,
                    output_params=['-threads', '2', '-preset', 'fast', '-crf', '22',
                                   '-movflags', '+faststart'])
                writer.send(None)
            references = {}
            if args.verify_existing_video:
                reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
                video_meta = next(reader)
                if tuple(video_meta['size']) != (960, 588):
                    raise ValueError('Existing video has incorrect dimensions')
                count = 0
                for output_index, pixels in enumerate(reader):
                    if output_index in (0, 100, 200, len(sample_indices) - 1):
                        references[output_index] = np.frombuffer(pixels, dtype=np.uint8).reshape(588, 960, 3)
                    count += 1
                if count != len(sample_indices):
                    raise ValueError('Existing video has incorrect frame count')
            pixel_checks = []
            try:
                font = ImageFont.truetype('DejaVuSans.ttf', 17)
            except OSError:
                font = ImageFont.load_default(size=17)
            selected = set(sample_indices)
            errors = {'joint_rad': 0., 'base_pose': 0., 'tcp_m': 0., 'fruit_m': 0.}
            env.render_quality = 'fast'
            rendered = 0
            fruit_samples = 0
            for index, row in enumerate(rows):
                errors['joint_rad'] = max(errors['joint_rad'], float(np.max(np.abs(env.state() - row['state']))))
                errors['base_pose'] = max(errors['base_pose'], float(np.max(np.abs(env.base_pose() - row['base_pose']))))
                errors['tcp_m'] = max(errors['tcp_m'], float(np.linalg.norm(env.data.site_xpos[env.tcp] - row['tcp_position'])))
                if row.get('fruit_position') is not None:
                    fruit_samples += 1
                    errors['fruit_m'] = max(errors['fruit_m'], float(np.linalg.norm(env.fruit_pos(row['target']) - row['fruit_position'])))
                if max(errors.values()) > 1e-6:
                    raise RuntimeError(f'Physics replay diverged at source row {index}: {errors}')
                if index in selected:
                    rendered += 1
                if index in selected and (not args.verify_existing_video or rendered - 1 in references):
                    cam = cameras['robot-third-person']
                    cam.lookat[1] = env.base_pose()[1]
                    view = Image.fromarray(env.render(cam, 960, 540))
                    canvas = Image.new('RGB', (960, 588), '#101b28')
                    canvas.paste(view, (0, 24))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((12, 3), 'THIRD-PERSON / PHYSICS REPLAY / SEED 60', font=font, fill='#85d9b0')
                    draw.text((12, 567), f'{row["phase"]} | source {row["timestamp"]:.2f}s | 2x / stationary waits trimmed', font=font, fill='white')
                    if args.verify_existing_video:
                        difference = float(np.abs(np.asarray(canvas).astype(np.int16) -
                                           references[rendered - 1].astype(np.int16)).mean())
                        if difference > 8:
                            raise ValueError(f'Existing video disagrees with re-render: mean pixel difference {difference}')
                        pixel_checks.append({'output_frame': rendered - 1, 'mean_absolute_rgb_error': difference})
                    else:
                        writer.send(np.asarray(canvas))
                    if row['phase'] in ('bend_pull_hold', 'delivery_release'):
                        snapshot = out / ('third-person-' + row['phase'] + '.png')
                        if not snapshot.exists():
                            canvas.save(snapshot)
                env.drive(*row['base_command'])
                env.step(row.get('requested_action', row['action']))
                if index % 250 == 0:
                    print(f'Replayed {index}/{len(rows)} source frames; max deviation {max(errors.values()):.3g}', flush=True)
            if writer is not None:
                writer.close()
                writer = None
            from vista import evaluate
            acceptance = evaluate(env, rows)
            if not acceptance['success'] or acceptance['ripe_fruit_in_basket'] != report['ripe_fruit_in_basket']:
                raise RuntimeError(f'Replay did not reproduce independent harvest acceptance: {acceptance}')
            if env.events != report['events']:
                raise RuntimeError('Replayed detachment events differ from the original')
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-y',
                '-i', str(path), '-filter_complex',
                '[0:v]scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];'
                '[b][p]paletteuse=dither=bayer:bayer_scale=3', '-threads', '2',
                '-filter_complex_threads', '1', '-loop', '0',
                str(out / 'seed-060-third-person.gif')], check=True)
            metadata = {
                'seed': config['seed'], 'scene_version': config['scene_version'],
                'method': 'Physics replay of recorded commands, rendered from a new third-person camera',
                'original_camera_recording': False, 'state_teleportation': False,
                'physics_sources_hash_checked': True, 'source_sha256': public_demo['source_sha256'],
                'max_replay_deviation': {**errors, 'fruit_m': errors['fruit_m'] if fruit_samples else None},
                'fruit_position_samples_checked': fruit_samples,
                'independent_harvest_acceptance': acceptance, 'detachment_events_matched': True,
                'basket_success_reproduced': acceptance['success'],
                'all_physics_steps_replayed': True, 'source_rows': len(rows),
                'output_frames': rendered, 'fps': 10, 'duration_seconds': rendered / 10,
                'selected_source_row_indices': sample_indices,
                'presentation_camera_tracks': 'Base Y only; not fruit or target coordinates',
                'playback': 'Same verified stationary-wait trimming and 2x timing as dual-camera preview',
                'renderer': 'Mesa software OpenGL; fast video rendering, full-quality initial stills',
                'no_model_calls': True,
                'existing_video_rerender_spot_checks': pixel_checks,
                'media_sha256': {p.name: sha(p) for p in sorted(out.iterdir()) if p.suffix in ('.png', '.mp4', '.gif')},
            }
            (out / 'third-person-replay.json').write_text(json.dumps(metadata, indent=2) + '\n')
            print(json.dumps({'success': True, 'frames': rendered, 'max_deviation': errors}), flush=True)
        finally:
            if writer is not None:
                writer.close()
            env.close()


if __name__ == '__main__':
    main()
