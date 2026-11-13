"""Audit recorded selected RGB-D pixels and candidate attribution without simulation/API."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import argparse,json
from pathlib import Path
import numpy as np
from local_targets import LocalTargets,CandidateCorrespondenceError
from vista import depth_point


def audit(source,out):
    active=None;records=[]
    for line in (source/'vista-events.jsonl').read_text().splitlines():
        event=json.loads(line)
        if event['kind']!='receipt' or not isinstance(event.get('result'),dict):continue
        result=event['result']
        if 'selected_pixel' in result and active:
            frame=result['frame_id'];camera=result['camera'];pixel=result['selected_pixel']
            with np.load(source/f'observations/{frame}-{camera}-depth.npz') as z:
                point=depth_point(z['depth'],z,*pixel)
            queue=LocalTargets();queue.register([active],[0,0,0],'Recorded visual candidate only')
            try:queue.check_target(point);corresponds=True
            except CandidateCorrespondenceError:corresponds=False
            records.append({'call':event['call'],'frame':frame,'selected_pixel':pixel,'selected_rgb':result['selected_rgb'],
                            'active_candidate_key':active['candidate_key'],'selected_surface_m':point.tolist(),
                            'delta_from_active_body_m':(point-np.array(active['surface_m'])).tolist(),
                            'surface_in_active_neighbourhood':corresponds,'old_fit_diagnostic':result.get('stalk_fit'),
                            'new_rule_counts_candidate_failure':corresponds})
        local=result.get('local_selection') or result.get('station_search',{}).get('local_selection')
        if local:active=local['active']
    report={'source':str(source.resolve()),'model_calls':0,'method':'Recorded RGB-D surfaces and model-proposed body measurements; no scene IDs or rerun',
            'selected_pixels':records,'limitation':'Spatial correspondence only; not semantic stalk identification or an online trial.'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();print(json.dumps(audit(a.source,a.out),ensure_ascii=False,indent=2))
