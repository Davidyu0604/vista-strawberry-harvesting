"""Create a trimmed synchronized-camera viewing copy, preserving the raw episode."""
import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--format', choices=('webm', 'mp4'), default='webm',
                        help='WebM/VP8 works with the installed desktop decoder; MP4 requires H.264')
    args = parser.parse_args()
    episode = args.episode.resolve()
    rows = [json.loads(line) for line in (episode / 'trajectory.jsonl').read_text().splitlines()]
    start = next((i for i, r in enumerate(rows) if r['stage'] not in ('settling', 'waiting')), None)
    if start is None:
        raise ValueError('No task motion found; cannot trim to a picking sequence')
    out = episode / ('replay-trimmed.' + args.format)
    if out.exists():
        raise FileExistsError(out)
    recorded=json.loads((episode/'recording.json').read_text())['cameras']
    cameras=tuple(c for c in ('harvest_close','wrist','top') if c in recorded)
    if len(cameras) not in (2,3) or set(cameras)!=set(recorded):
        raise ValueError('Unsupported recorded camera layout')
    width=480*len(cameras)
    fps = json.loads((episode / 'report.json').read_text())['fps']
    codec = 'libvpx' if args.format == 'webm' else 'libx264'
    options = (['-crf', '10', '-b:v', '3M', '-deadline', 'good', '-cpu-used', '4']
               if args.format == 'webm' else ['-crf', '20', '-movflags', '+faststart'])
    writer = imageio_ffmpeg.write_frames(str(out), (width, 390), fps=fps,
        codec=codec, quality=None, pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
        macro_block_size=2, output_params=options)
    writer.send(None)
    try:
        for row in rows[start:]:
            canvas = Image.new('RGB', (width, 390), 'black')
            draw = ImageDraw.Draw(canvas)
            for i, camera in enumerate(cameras):
                with Image.open(episode / row['images'][camera]) as source:
                    if source.size != (480, 360):
                        raise ValueError('Unexpected recorded image size')
                    canvas.paste(source.convert('RGB'), (i * 480, 30))
                draw.text((i * 480 + 8, 8),
                    f'{camera} | source {row["timestamp"]:.2f}s | {row["stage"]}', fill='white')
            writer.send(np.asarray(canvas))
    finally:
        writer.close()
    info = dict(source_episode=str(episode), video=str(out), codec=codec,
        raw_episode_modified=False, training_dataset_modified=False,
        removed_initial_frames=start, removed_initial_seconds=start / fps,
        source_start_frame=rows[start]['frame'], source_end_frame=rows[-1]['frame'],
        exported_frames=len(rows)-start, fps=fps, duration_seconds=(len(rows)-start)/fps,
        cameras=list(cameras), rule='Trim only initial settling/waiting before first task motion; preserve all later frames.')
    (episode / 'replay-trimmed.json').write_text(json.dumps(info, indent=2) + '\n')
    from record_catalog import register_episode
    register_episode(episode)
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
