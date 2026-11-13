"""Offline saved-frame semantic-review fixture; no new model/physics rollout."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from local_targets import LocalTargets
from perception_recovery import body_semantics_review_required, review_proposals
from visual_proposals import refresh_proposals, reviewed_candidates
from vista import depth_point


def validate(out):
    root=PROJECT_ROOT
    run=root/'reports/02_批量评估/body-context-seeds2-4-20261111-135905/seed-2'
    events=[json.loads(l) for l in (run/'vista-events.jsonl').read_text().splitlines()]
    initial=next(e for e in events if e['kind']=='body_proposals')
    request=next(e for e in events if e['kind']=='response')['message']['tool_calls'][0]['function']
    original=json.loads(request['arguments']);frame=original['frame_id']
    base=next(e for e in events if e['kind']=='observation')['base_x_y_yaw']
    with Image.open(run/'observations'/f'{frame}-harvest_close.jpg') as rgb:
        measured,blocked=reviewed_candidates(original,initial,frame,rgb)
    local=LocalTargets();local.register(measured,base,original['coverage'])
    local.survey_review=deepcopy(original);local.survey_batch=deepcopy(initial)
    conflicts=[c['proposal_id'] for c in local.candidates if body_semantics_review_required(local,c)]
    assert conflicts==['P1','P2']
    local.candidates[0]['correction_attempts']=2
    try:local.require_active()
    except ValueError as error:
        instruction=str(error);assert 'review_proposals' in instruction
    else:raise AssertionError('Unverified ripe claim must not authorize approach')
    review=deepcopy(original);review['frame_id']='f0000047'
    # Explicitly adapted review, NOT a new model decision or online success.
    for item in review['candidates']:
        if item['proposal_id'] in conflicts:
            item.update(maturity='turning',reason='Offline fixture: whole body is orange-red, not fully red')
    with Image.open(run/'observations/f0000047-harvest_close.jpg') as rgb, np.load(run/'observations/f0000047-harvest_close-depth.npz') as sample:
        batch=refresh_proposals(initial,rgb,sample['depth'],sample,base,depth_point)
        measured,blocked=reviewed_candidates(review,batch,review['frame_id'],rgb)
        result=review_proposals(local,review,batch,measured,blocked,'Offline adapted review of orange-red body context',rgb,base)
    assert all(c['maturity']=='turning' and c['status']=='visually_unavailable'
               for c in local.candidates if c['proposal_id'] in conflicts)
    assert local.candidates[0]['correction_attempts']==2
    assert local.active is None and not local.can_search  # Other uncertain claims remain.
    summary=dict(model_calls=0,videos=0,online_success_validated=False,
        fixture_note='Adapted P1/P2 maturity review, not an actual model response',
        original_source=str(run),conflicting_claims=conflicts,instruction=instruction,
        review_result=result,remaining_selection=local.status(),
        source_sha256={name:hashlib.sha256(project_source(name).read_bytes()).hexdigest() for name in
            ('perception_recovery.py','local_targets.py','vista.py','test_body_semantics_review.py','validate_body_semantics_review.py')})
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(dict(conflicts=conflicts,review_accepted=True,
                         other_uncertain_claims_still_block_search=not local.can_search)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);validate(p.parse_args().out)
