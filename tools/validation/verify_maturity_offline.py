"""Replay saved successful localizations, without physics, network or paid calls."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from visual_maturity import assess_maturity


def replay(folder, out):
    results=[]
    for line in (folder/'vista-events.jsonl').read_text().splitlines():
        r=json.loads(line).get('result',{})
        if not isinstance(r,dict) or not r.get('stalk_fit') or not r.get('locked_target_m'):continue
        frame=r['frame_id'];axis=r['stalk_fit'].get('axis_world',[0,0,1])
        with np.load(folder/'observations'/f'{frame}-harvest_close-depth.npz') as sample, Image.open(folder/'observations'/f'{frame}-harvest_close.jpg') as image:
            result,marked=assess_maturity(r['locked_target_m'],axis,sample['depth'],sample,image)
        result.update(source=str(folder),frame_id=frame)
        marked.save(out/f'{folder.name}-{frame}-maturity.jpg')
        results.append(result)
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('episodes',type=Path,nargs='+')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    results=[r for p in args.episodes for r in replay(p,args.out)]
    (args.out/'report.json').write_text(json.dumps(results,indent=2))
    for r in results:print(Path(r['source']).name,r['frame_id'],'PASS' if r['passed'] else 'REJECT',f"red={r['red_fraction']:.3f}",f"samples={r['samples']}")
