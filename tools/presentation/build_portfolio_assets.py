"""Build public demo media and aggregate evidence from local, verified recordings.

No simulation, model API, GPU, training, or source-recording writes.
Requires Pillow, numpy and imageio-ffmpeg. Raw data is not shipped in Git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=root)
    parser.add_argument('--seed', type=int, default=60)
    args = parser.parse_args()
    root = args.root.resolve()
    prepared_path = root / 'datasets/fresh60-trimmed-v1/manifest.json'
    export_root = root / 'datasets/strawberry-act-60-v1'
    gpu_path = root / 'reports/04_训练验证/fresh60-training-v1/gpu-training-verification.json'
    prepared = read(prepared_path)
    exported = read(export_root / 'export_manifest.json')
    verification = read(export_root / 'export_verification.json')
    gpu = read(gpu_path)
    assert prepared['status'] == 'verified' and verification['passed']
    assert exported['formal_training_dataset'] is True
    candidates = [e for e in prepared['episodes'] if e['seed'] == args.seed]
    if len(candidates) != 1:
        raise ValueError('Expected exactly one accepted recording for this seed')
    episode = candidates[0]
    source = Path(episode['source'])
    for name, expected in episode['source_files_sha256'].items():
        if sha(source / name) != expected:
            raise ValueError(f'Source checksum changed: {name}')
    report, quality, config = [read(source / name) for name in
                              ('report.json', 'quality.json', 'run-config.json')]
    assert report['success'] and quality['integrity_passed']
    rows = [json.loads(line) for line in (source / 'trajectory.jsonl').read_text().splitlines()]
    retained = episode['source_row_indices']
    assert len(rows) == episode['raw_frames'] and len(retained) == episode['frames']
    assert retained == sorted(set(retained)) and retained[-1] < len(rows)
    media, evidence = root / 'docs/media', root / 'docs/evidence'
    media.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    stem = f'seed-{args.seed:03d}'
    full, preview = media / f'{stem}-full.mp4', media / f'{stem}-preview.mp4'
    for path in (full, preview):
        if path.exists():
            raise FileExistsError(f'Remove the previous derived output explicitly: {path}')
    try:
        title_font = ImageFont.truetype('DejaVuSans-Bold.ttf', 22)
        label_font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError:
        title_font = ImageFont.load_default(size=22)
        label_font = ImageFont.load_default(size=16)

    def frame(row, preview_mode):
        canvas = Image.new('RGB', (960, 464), '#101b28')
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 10), 'VISTA / STRAWBERRY HARVESTING', font=title_font, fill='#f1f5f9')
        draw.text((18, 42), 'HARVEST CAMERA', font=label_font, fill='#85d9b0')
        draw.text((498, 42), 'WRIST CAMERA', font=label_font, fill='#85d9b0')
        for x, camera in ((0, 'harvest_close'), (480, 'wrist')):
            with Image.open(source / row['images'][camera]) as image:
                if image.size != (480, 360):
                    raise ValueError('Unexpected camera dimensions')
                canvas.paste(image.convert('RGB'), (x, 68))
        mode = '2x / stationary waits trimmed' if preview_mode else '1x / all recorded frames'
        label = f'Seed {args.seed}  |  {row["phase"]}  |  source {row["timestamp"]:.2f}s  |  {mode}'
        draw.text((14, 440), label, font=label_font, fill='#e2e8f0')
        return np.asarray(canvas)

    writers = []
    selected = retained[::4]  # 20 Hz edited timeline, shown at 10 fps: 2x playback.
    selected_set = set(selected)
    try:
        for path, fps in ((full, 20), (preview, 10)):
            writer = imageio_ffmpeg.write_frames(str(path), (960, 464), fps=fps,
                codec='libx264', quality=None, macro_block_size=2,
                output_params=['-threads', '2', '-preset', 'fast', '-crf', '23',
                               '-movflags', '+faststart'])
            writer.send(None)
            writers.append(writer)
        for index, row in enumerate(rows):
            writers[0].send(frame(row, False))
            if index in selected_set:
                writers[1].send(frame(row, True))
    finally:
        for writer in writers:
            writer.close()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, '-v', 'error', '-i', str(preview), '-filter_complex',
        '[0:v]scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];'
        '[b][p]paletteuse=dither=bayer:bayer_scale=3', '-threads', '2',
        '-filter_complex_threads', '1', '-loop', '0', str(media / f'{stem}-preview.gif')], check=True)
    # A lossless snapshot from the source-recording composite, not a generated scene.
    poster_row = next(r for r in rows if r['phase'] == 'bend_pull_hold')
    Image.fromarray(frame(poster_row, False)).save(media / f'{stem}-poster.png')
    metadata = {
        'seed': args.seed, 'scene_version': config['scene_version'],
        'controller': 'VISTA-style visual/tool loop; not ACT', 'model': config['model'],
        'source_episode_id': source.relative_to(root).as_posix(),
        'source_sha256': episode['source_files_sha256'],
        'controller_source_sha256': config['source_sha256'],
        'cameras': ['harvest_close', 'wrist'], 'model_calls': report['model_calls'],
        'harvest_success': report['success'], 'integrity_passed': quality['integrity_passed'],
        'single_fruit_delivered': report['single_fruit_delivered'],
        'returned_to_ready': report['returned_to_ready'],
        'full': {'frames': len(rows), 'fps': 20, 'duration_seconds': len(rows) / 20,
                 'all_recorded_frames_in_order': True},
        'preview': {'prepared_frames': len(retained), 'output_frames': len(selected),
                    'fps': 10, 'duration_seconds': len(selected) / 10,
                    'playback_speed_after_wait_trimming': 2,
                    'removed_stationary_wait_frames': len(rows) - len(retained),
                    'selected_source_row_indices': selected,
                    'rule': 'Verified stationary-wait trimming, then every fourth retained frame at 10 fps.'},
        'media_sha256': {p.name: sha(p) for p in sorted(media.glob(stem + '-*'))},
        'scope': 'One selected successful simulation example; not a success-rate estimate.'}
    write(evidence / 'demo.json', metadata)
    train_indices, val_indices = exported['split']['train'], exported['split']['validation']
    train_seeds = sorted({e['seed'] for e in exported['sources'] if e['episode_index'] in train_indices})
    val_seeds = sorted({e['seed'] for e in exported['sources'] if e['episode_index'] in val_indices})
    assert not set(train_seeds) & set(val_seeds)
    total = prepared['totals']
    write(evidence / 'data-summary.json', {
        'snapshot_date': '2026-11-13', 'scope': 'Local dataset preparation and diagnostic checks',
        'raw_data_and_checkpoints_in_repository': False,
        'episodes': len(prepared['episodes']), 'unique_seeds': len(set(train_seeds + val_seeds)),
        'raw_frames': total['raw_frames'], 'retained_frames': total['frames'],
        'removed_frames': total['removed_frames'],
        'removed_percent': 100 * total['removed_frames'] / total['raw_frames'],
        'fps': 20, 'stationary_wait_policy': prepared['policy'],
        'all_non_wait_frames_retained': all(e['all_non_wait_frames_retained'] for e in prepared['episodes']),
        'split': {'train_episodes': len(train_indices), 'validation_episodes': len(val_indices),
                  'train_frames': gpu['train_frames'], 'validation_frames': gpu['validation_frames'],
                  'train_seeds': train_seeds, 'validation_seeds': val_seeds, 'seed_overlap': []},
        'scene_counts': {v: sum(e['scene_version'] == v for e in prepared['episodes'])
                         for v in sorted({e['scene_version'] for e in prepared['episodes']})},
        'export_verification': verification,
        'gpu_diagnostic': {k: gpu[k] for k in ('passed', 'initial_training_steps', 'resumed_steps',
                          'optimizer_step_restored', 'parameters_updated_after_resume',
                          'normalization_train_only', 'chunk_size', 'batch_size', 'gpu')},
        'closed_loop_act_evaluated': False,
        'source_sha256': {'prepared_manifest': sha(prepared_path),
                          'export_manifest': sha(export_root / 'export_manifest.json'),
                          'export_verification': sha(export_root / 'export_verification.json'),
                          'gpu_diagnostic': sha(gpu_path)}})
    print(json.dumps({'full_seconds': len(rows) / 20, 'preview_seconds': len(selected) / 10,
                      'media_bytes': sum(p.stat().st_size for p in media.glob(stem + '-*'))}))


if __name__ == '__main__':
    main()
