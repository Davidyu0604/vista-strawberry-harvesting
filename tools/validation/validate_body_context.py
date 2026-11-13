"""Saved RGB-D regression only: no model, simulation truth or video generation."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from local_targets import body_pixel_evidence
from visual_proposals import propose_bodies
from vista import depth_point


def validate(out):
    root=PROJECT_ROOT
    batch=root/'reports/02_批量评估/maturity-layout-seeds2-4-20261111-134748'
    spec=importlib.util.spec_from_file_location('old_proposals',batch/'seed-4/source/visual_proposals.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    results=[]
    for seed in (2,4):
        run=batch/f'seed-{seed}'
        events=[json.loads(line) for line in (run/'vista-events.jsonl').read_text().splitlines()]
        frames=[]
        for event in events:
            if event['kind']!='observation':continue
            frame=event['frame_id'];path=run/'observations'/f'{frame}-harvest_close.jpg'
            with Image.open(path) as source,np.load(path.with_name(f'{frame}-harvest_close-depth.npz')) as d:
                rgb=source.convert('RGB');base=event['base_x_y_yaw']
                before=old.propose_bodies(rgb,d['depth'],d,base,depth_point)
                after=propose_bodies(rgb,d['depth'],d,base,depth_point)
            # The orange foreground fruit must remain represented and measured,
            # not silently discarded to let the robot move.
            anchor=(335,314) if seed==2 else (178,176)
            def contains(p):
                x0,y0,x1,y1=p['bbox_xyxy'];x,y=anchor
                return x0<=x<=x1 and y0<=y<=y1
            foreground=[p for p in after['proposals'] if contains(p)]
            assert len(foreground)==1,(seed,frame,foreground)
            p=foreground[0]
            assert p['depth_verified'] and p['body_area_pixels']>500
            assert not body_pixel_evidence(rgb,*p['pixel'])['passed']
            if seed==4:
                ripe=[p for p in after['proposals'] if p['bbox_xyxy'][0]<=281<=p['bbox_xyxy'][2]
                      and p['bbox_xyxy'][1]<=132<=p['bbox_xyxy'][3]]
                assert len(ripe)==1 and ripe[0]['depth_verified']
                assert body_pixel_evidence(rgb,*ripe[0]['pixel'])['passed']
            frames.append(dict(frame=frame,
                old_count=len(before['proposals']),new_count=len(after['proposals']),
                old_unknown_depth=sum(p['surface_m'] is None for p in before['proposals']),
                new_unknown_depth=sum(p['surface_m'] is None for p in after['proposals']),
                foreground=foreground[0],proposals=after['proposals']))
        results.append(dict(seed=seed,frames=frames))
    summary=dict(model_calls=0,videos=0,online_success_validated=False,
                 scene_changed=False,saved_rgbd_regressions=results,
                 source_sha256={name:hashlib.sha256(project_source(name).read_bytes()).hexdigest()
                                for name in ('visual_proposals.py','vista.py','test_visual_proposals.py','validate_body_context.py')})
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    for row in results:
        print(json.dumps(dict(seed=row['seed'],frames=len(row['frames']),
            initial_counts={key:row['frames'][0][key] for key in
                ('old_count','new_count','old_unknown_depth','new_unknown_depth')})))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    validate(parser.parse_args().out)
