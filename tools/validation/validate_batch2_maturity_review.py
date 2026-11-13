"""Saved RGB-D review evidence only; corrected claims are an offline fixture."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2] / 'src'))
import argparse,json
from copy import deepcopy
from PIL import Image
from local_targets import LocalTargets
from visual_proposals import reviewed_candidates
from perception_recovery import review_proposals,body_color_conflict


def validate(episode,out):
    events=[json.loads(l) for l in (episode/'vista-events.jsonl').read_text().splitlines()]
    def tool(call):
        e=next(e for e in events if e['kind']=='response' and e['call']==call)
        return json.loads(e['message']['tool_calls'][0]['function']['arguments'])
    def batch(frame):return next(e for e in events if e['kind']=='body_proposals' and e['frame_id']==frame)
    old=tool(1);review=tool(3);reason=review.pop('correction_reason');old_batch=batch(old['frame_id']);current=batch(review['frame_id'])
    base=next(e['base_x_y_yaw'] for e in events if e['kind']=='observation' and e['frame_id']==old['frame_id'])
    with Image.open(episode/'observations'/f"{old['frame_id']}-harvest_close.jpg") as im:
        measured,blocked=reviewed_candidates(old,old_batch,old['frame_id'],im.convert('RGB'))
    local=LocalTargets();local.register(measured,base,old['coverage']);local.selection_blocked_reason=blocked
    local.survey_review=deepcopy(old);local.survey_batch=deepcopy(old_batch)
    with Image.open(episode/'observations'/f"{review['frame_id']}-harvest_close.jpg") as im:
        rgb=im.convert('RGB');measured,blocked=reviewed_candidates(review,current,review['frame_id'],rgb)
        entry=review_proposals(local,review,current,measured,blocked,reason,rgb,base)
    assert entry['unresolved_claims'][0]['ripe_color_conflict'] and not local.can_search
    # Offline counterfactual: explicit same-body orange/turning review on next frame.
    corrected=deepcopy(review);corrected['frame_id']='f0000164'
    corrected['candidates'][0].update(maturity='turning',reason='Offline fixture: orange-dominant whole body')
    with Image.open(episode/'observations'/'f0000164-harvest_close.jpg') as im:
        rgb=im.convert('RGB');fresh=batch(corrected['frame_id'])
        measured,blocked=reviewed_candidates(corrected,fresh,corrected['frame_id'],rgb)
        correction=review_proposals(local,corrected,fresh,measured,blocked,'Offline corrected visual claim',rgb,base)
    assert local.can_search and local.proposal_review_attempts==2
    output={'original_review_feedback':entry,'counterfactual_correction':correction,'queue_search_allowed':local.can_search,
            'model_calls':0,'online_success_validated':False,'note':'Tests truthful feedback and existing color gate, not a new model decision or successful harvest.'}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print('Saved review stayed blocked; corrected orange claim passed the original gate.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();validate(a.episode.resolve(),a.out.resolve())
