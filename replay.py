"""Render recorded robot/base commands without contacting the model API."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from replay_source import replay_environment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('episode',type=Path)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    report=json.loads((args.episode/'report.json').read_text())
    config=json.loads((args.episode/'run-config.json').read_text())
    out=args.out or args.episode/'replay'
    out.mkdir(exist_ok=False)
    rows=[json.loads(line) for line in (args.episode/'trajectory.jsonl').read_text().splitlines()]
    e,physics_source=replay_environment(args.episode.resolve(),config,report)
    frames=[];tcp_error=0.;fruit_error=0.
    try:
        for i,row in enumerate(rows):
            tcp_error=max(tcp_error,float(np.linalg.norm(e.data.site_xpos[e.tcp]-row['tcp_position'])))
            if row.get('fruit_position') is not None:
                fruit_error=max(fruit_error,float(np.linalg.norm(e.fruit_pos(row['target'])-row['fruit_position'])))
            if tcp_error>1e-6 or fruit_error>1e-6:raise RuntimeError('Recorded physics state differs; replay stopped')
            if i%8==0 or i==len(rows)-1:
                # 0.4s simulated time per image, played for 0.2s: 2x replay.
                camera='detail' if row['phase'].startswith('delivery_') or not e.attached[0] else 'harvest_close'
                im=Image.fromarray(e.render(camera,640,480))
                draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,30),fill='black')
                draw.text((8,9),f"REPLAY 2x | t={row['timestamp']:.1f}s | {row['phase']} | {camera}",fill='white')
                frames.append(im)
            e.drive(*row['base_command']);e.step(row.get('requested_action',row['action']))
        result={'source_episode':str(args.episode.resolve()),'no_model_calls':True,
                'method':'physics replay of recorded joint and base commands; no state teleportation',
                'playback_speed':2,'max_tcp_deviation_m':tcp_error,'max_fruit_deviation_m':fruit_error,
                'replay_in_basket':bool(e.success),'base_pose':e.base_pose().tolist(),
                'recorded_success':report['success'],'source_hashes_verified':True,'physics_source':physics_source}
        (out/'replay-report.json').write_text(json.dumps(result,indent=2))
        frames[0].save(out/'replay.gif',save_all=True,append_images=frames[1:],duration=200,loop=0)
        frames[-1].save(out/'final.png')
        print(json.dumps({'out':str(out),**result}),flush=True)
    finally:e.close()


if __name__=='__main__':main()
