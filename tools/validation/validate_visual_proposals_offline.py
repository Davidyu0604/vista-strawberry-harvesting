"""Check numbered proposals on archived camera observations, without model/simulation."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from local_targets import body_pixel_evidence
from visual_proposals import propose_bodies
from vista import depth_point


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    results=[]
    for episode in args.episodes:
        with (episode/'vista-events.jsonl').open() as stream:
            observation=next(json.loads(line) for line in stream if json.loads(line)['kind']=='observation')
        frame=observation['frame_id']
        path=episode/observation['images']['harvest_close']
        depth_path=path.with_name(f'{frame}-harvest_close-depth.npz')
        with Image.open(path) as source:rgb=source.convert('RGB')
        with np.load(depth_path) as sample:
            batch=propose_bodies(rgb,sample['depth'],sample,observation['base_x_y_yaw'],depth_point)
        for proposal in batch['proposals']:
            proposal['body_pixel_check']=body_pixel_evidence(rgb,*proposal['pixel'])
        results.append({'episode':str(episode.resolve()),'frame_id':frame,
                        'base_x_y_yaw':observation['base_x_y_yaw'],
                        'input_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,depth_path)},
                        **batch})
    root=PROJECT_ROOT
    output={'model_calls':0,'simulation_runs':0,'videos_generated':0,'training_exported':False,
            'limitation':'Only archived RGB-D proposal sampling. No new model semantic review, online harvest or success rate.',
            'source_sha256':{name:hashlib.sha256(project_source(name).read_bytes()).hexdigest()
                             for name in ('visual_proposals.py','vista.py','local_targets.py',Path(__file__).name)},
            'results':results}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:json.dump(output,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'output':str(args.out),'observations':len(results),
                      'proposal_counts':[len(r['proposals']) for r in results]}))


if __name__=='__main__':main()
