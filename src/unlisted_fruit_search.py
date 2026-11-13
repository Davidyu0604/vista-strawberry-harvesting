"""Measured out-of-envelope fruit permits observation search, never grasp."""
import numpy as np
from scipy import ndimage
from visual_proposals import body_color_masks, body_regions, propose_bodies
from perception_recovery import MISSING_BLOCK


def inspection_available(session):
    local = getattr(session, 'local_targets', None)
    return bool(local and local.surveyed and local.selection_blocked_reason == MISSING_BLOCK
                and getattr(local, 'unlisted_inspections', 0) < 2
                and getattr(session, 'visual_proposals', False)
                and session.target_stage is None and not session.grasp_attempted
                and not getattr(session, 'motion_stop_trace', None))


def inspect_claim_pixels(rgb, depth, calibration, base, claims, unproject, proposals):
    """Return feedback on claimed pixels, without certifying absence or clearing a block."""
    from local_targets import body_pixel_evidence
    if not isinstance(claims, list) or not 1 <= len(claims) <= 8:
        raise ValueError('Report 1 to 8 claimed bodies in ORIGINAL 480x360 coordinates')
    checked = []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {'x', 'y', 'reason'}:
            raise ValueError('Each claimed body requires x, y, reason')
        x, y = claim['x'], claim['y']
        if (type(x) is not int or type(y) is not int or not 1 <= x < rgb.width - 1
                or not 1 <= y < rgb.height - 1 or not isinstance(claim['reason'], str)
                or not 1 <= len(claim['reason'].strip()) <= 1200):
            raise ValueError('Use an interior original-image pixel and visual reason')
        evidence = body_pixel_evidence(rgb, x, y)
        item = {'pixel': [x, y], 'reason': claim['reason'], 'body_pixel_check': evidence,
                'numbered_boxes_containing_pixel': [p['proposal_id'] for p in proposals
                    if p['bbox_xyxy'][0] <= x <= p['bbox_xyxy'][2]
                    and p['bbox_xyxy'][1] <= y <= p['bbox_xyxy'][3]]}
        try:
            point = np.asarray(unproject(depth, calibration, x, y))
            item['selected_surface_m'] = point.tolist()
            item['selected_surface_distance_xy_m'] = float(np.linalg.norm(point[:2] - np.asarray(base)[:2]))
        except ValueError as exc:
            item['depth_error'] = str(exc)
        item['instruction'] = ('Inspect the marked ORIGINAL point and full frame. If the claimed fruit is elsewhere, '
                               'correct its coordinates; do not infer absence from a bad point. If the missing-fruit '
                               'claim was genuinely mistaken, correct_survey with visual evidence. Otherwise retain '
                               'the block. This check authorizes no movement, candidate deletion or grasp.')
        checked.append(item)
    return checked


def review_available(session):
    local=getattr(session,'local_targets',None)
    return bool(local and local.surveyed and not local.candidates
        and local.selection_blocked_reason==MISSING_BLOCK
        and getattr(local,'survey_review',{}).get('missing_visible_ripe') is True
        and getattr(session,'visual_proposals',False)
        and not getattr(session,'fixed_base',False)
        and session.target_stage is None and not session.grasp_attempted
        and not session.target_confirmed and not getattr(session,'motion_stop_trace',None)
        and getattr(session,'preclose_recoveries',0)<2
        and not getattr(session,'preclose_recovery_failed',False))


def measure_unlisted(rgb,depth,calibration,base,claims,unproject):
    """Check fresh image evidence, including possible nearer/unknown regions."""
    fresh=propose_bodies(rgb,depth,calibration,base,unproject)
    if fresh['proposals'] or fresh['omitted_by_limit']:
        raise ValueError('Fresh local or unresolved RGB-D regions remain; no unlisted-fruit search')
    if not isinstance(claims,list) or not 1<=len(claims)<=8:
        raise ValueError('Report 1 to 8 unnumbered fruit bodies from the original close image')
    red,mask=body_color_masks(rgb);regions=list(body_regions(mask));checked=[];used=set()
    for claim in claims:
        if not isinstance(claim,dict) or set(claim)!={'x','y','reason'}:
            raise ValueError('Each body requires x, y and visual reason')
        x,y=claim['x'],claim['y']
        if (type(x) is not int or type(y) is not int or not 1<=x<rgb.width-1 or not 1<=y<rgb.height-1
                or not isinstance(claim['reason'],str) or not 1<=len(claim['reason'].strip())<=1200):
            raise ValueError('Use an interior body pixel in original image coordinates and visual evidence')
        if red[y-1:y+2,x-1:x+2].sum()<5:
            raise ValueError('Selected pixel lacks coherent red body evidence; inspect the image')
        selected=next(((i,b,r) for i,(b,r) in enumerate(regions)
                       if b[0].start<=y<b[0].stop and b[1].start<=x<b[1].stop
                       and r[y-b[0].start,x-b[1].start]),None)
        if selected is None:raise ValueError('No image body region at selected pixel')
        i,box,region=selected
        if i in used:raise ValueError('Duplicate body region; report each visible unnumbered fruit once')
        used.add(i)
        inside=(ndimage.distance_transform_edt(np.pad(region,1))[1:-1,1:-1]>=1.4)&red[box]
        yy,xx=np.where(inside)
        if len(yy)<9 or int((red[box]&region).sum())<30:
            raise ValueError('Body too small for reliable distance measurement')
        samples=depth[box][inside]
        coherent=np.abs(samples-depth[y,x])<.02
        patch=depth[y-1:y+2,x-1:x+2]
        if (not np.all(np.isfinite(samples)&(samples>.01)&(samples<2))
                or coherent.mean()<.75 or np.count_nonzero(np.isfinite(patch)&(np.abs(patch-depth[y,x])<.01))<5):
            raise ValueError('Body depth missing or discontinuous; no search authorized')
        points=np.array([unproject(depth,calibration,int(rx+box[1].start),int(ry+box[0].start)) for ry,rx in zip(yy,xx)])
        distances=np.linalg.norm(points[:,:2]-np.asarray(base)[:2],axis=1)
        if not np.isfinite(points).all() or distances.min()<=.58 or distances.max()>1.2:
            raise ValueError('Body is near, on the workspace boundary, or too far to verify; no search authorized')
        if not np.all((points[:,2]>=.12)&(points[:,2]<=.46)):
            raise ValueError('Body height outside row observation range')
        point=np.median(points[coherent],axis=0)
        checked.append(dict(pixel=[x,y],reason=claim['reason'],surface_m=point.tolist(),
            distance_xy_m=float(np.linalg.norm(point[:2]-np.asarray(base)[:2])),
            minimum_distance_xy_m=float(distances.min()),interior_samples=len(points)))
    nearest=min(checked,key=lambda p:p['distance_xy_m'])
    dy=nearest['surface_m'][1]-base[1]
    if abs(dy)<.05:raise ValueError('No clear along-row observation direction')
    return dict(bodies=checked,direction=1 if dy>0 else -1,
                note='Outside current 0.56m candidate envelope with 0.02m margin. Not a maturity, identity or reachability certificate. Preserve missing-fruit claim; re-survey after a bounded observation move.')
