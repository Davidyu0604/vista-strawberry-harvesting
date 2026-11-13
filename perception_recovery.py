"""Bounded correction of visual evidence; no scene identity or task truth."""
from copy import deepcopy
import numpy as np


MISSING_BLOCK = 'Model reports an unnumbered visible ripe fruit; coverage unresolved, no skipping or driving'


def current_body_proposals(local, rgb, depth, calibration, base_pose, unproject):
    """Keep survey identities for either review path; new detection is a survey."""
    from visual_proposals import propose_bodies, refresh_proposals
    if (can_correct_survey(local) or can_review_proposals(local)
            or (getattr(local,'allow_uncertain_deferral',False) and getattr(local,'survey_batch',None))):
        return refresh_proposals(local.survey_batch, rgb, depth, calibration, base_pose, unproject)
    return propose_bodies(rgb, depth, calibration, base_pose, unproject)


def post_observation_review_available(local):
    """One extra review earned by a completed adjustment since the last review."""
    if local is None or getattr(local,'proposal_review_attempts',0)!=2:return False
    frame=(getattr(local,'survey_review',None) or {}).get('frame_id','')
    if not isinstance(frame,str) or not frame.startswith('f') or not frame[1:].isdigit():return False
    return any(c.get('observation_adjusted')
               and isinstance(c.get('observation_recovery_completed_frame'),int)
               and c['observation_recovery_completed_frame']>int(frame[1:])
               for c in local.candidates)


def can_review_proposals(local):
    return bool(local is not None and getattr(local, 'survey_review', None)
                and (getattr(local, 'proposal_review_attempts', 0) < 2
                     or post_observation_review_available(local))
                and ('unresolved body depth' in (local.selection_blocked_reason or '')
                     or any(c['status'] in ('unverified_measurement', 'unverified_semantics') for c in local.candidates)))


def measured_edge_fragments(local, rgb, depth, calibration, base_pose, unproject):
    """Refresh clipped, narrow survey regions; never classify them as absent."""
    from visual_proposals import refresh_proposals
    if not getattr(local,'survey_batch',None):return []
    fresh=refresh_proposals(local.survey_batch,rgb,depth,calibration,base_pose,unproject)
    original={p['proposal_id']:p for p in local.survey_batch['proposals']}
    unresolved={c.get('proposal_id'):c for c in local.candidates
                if c['status']=='unverified_semantics' and c.get('maturity')=='uncertain'}
    checks=[]
    for p in fresh['proposals']:
        if p['proposal_id'] not in unresolved:continue
        a=original[p['proposal_id']];x0,y0,x1,y1=p['bbox_xyxy']
        clipped=((x0==0 or x1==rgb.width-1) and x1-x0+1<=6
                 or (y0==0 or y1==rgb.height-1) and y1-y0+1<=6)
        if not clipped or not 0<p.get('body_area_pixels',p.get('red_area_pixels',0))<=96:continue
        if a['surface_m'] is None or p['surface_m'] is None:continue
        drift=float(np.linalg.norm(np.asarray(a['surface_m'])-p['surface_m']))
        if not np.isfinite(drift) or drift>.004:continue
        checks.append({'candidate_key':unresolved[p['proposal_id']]['candidate_key'],
                       'proposal_id':p['proposal_id'],'bbox_xyxy':p['bbox_xyxy'],
                       'surface_m':p['surface_m'],'depth_drift_m':drift,
                       'maturity':'uncertain','identity_resolved':False})
    return checks


def edge_viewpoint_search_available(session, frame=None):
    local=getattr(session,'local_targets',None)
    evidence=getattr(session,'edge_viewpoint_evidence',None)
    if not (local and local.surveyed and not local.selection_blocked_reason
            and local.active is None and getattr(local,'proposal_review_attempts',0)>=2
            and getattr(session,'centered',None) is not None
            and not getattr(session,'fixed_base',False)
            and session.target_stage is None and not session.grasp_attempted
            and not session.target_confirmed and not getattr(session,'motion_stop_trace',None)
            and not getattr(session,'preclose_recovery_failed',False)
            and evidence and evidence['frame_id'] in getattr(session,'frames',{})
            and (frame is None or evidence['frame_id']==frame)):
        return False
    unresolved=[c for c in local.candidates if c['status'] in ('unverified_semantics','unverified_measurement')]
    if any(c['status'] in ('pending','ready') for c in local.candidates):return False
    checked={c['candidate_key'] for c in evidence['fragments']}
    return bool(unresolved and all(c['status']=='unverified_semantics' and c.get('maturity')=='uncertain'
                                  and c['candidate_key'] in checked for c in unresolved))


def uncertain_viewpoint_search_available(session, frame=None):
    """A reviewed unknown region may prompt a new view, never certify absence."""
    local=getattr(session,'local_targets',None)
    if not (local and getattr(local,'allow_uncertain_deferral',False) and local.surveyed
            and not local.effective_selection_blocked_reason and local.active is None
            and getattr(local,'proposal_review_attempts',0)>=1
            and getattr(session,'centered',None) is not None
            and not getattr(session,'fixed_base',False)
            and getattr(session,'search_steps',0)<4
            and getattr(session,'max_calls',32)-getattr(session,'calls',0)>=10
            and session.target_stage is None and not session.grasp_attempted
            and not session.target_confirmed and not getattr(session,'motion_stop_trace',None)
            and not getattr(session,'preclose_recovery_failed',False)):
        return False
    batches=getattr(session,'proposal_frames',{})
    frame=frame or next(reversed(batches),None)
    if frame not in getattr(session,'frames',{}) or frame not in batches:return False
    reviewed=(getattr(local,'survey_review',None) or {}).get('frame_id')
    if not reviewed or not frame.startswith('f') or not reviewed.startswith('f'):return False
    if not frame[1:].isdigit() or not reviewed[1:].isdigit() or int(frame[1:])<int(reviewed[1:]):return False
    unresolved=any(c['status'] in ('unverified_semantics','unverified_measurement') for c in local.candidates)
    missing=any(p.get('surface_m') is None for p in batches[frame]['proposals'])
    return bool(unresolved or missing)


def body_color_conflict(local, candidate):
    """A mostly orange measured region needs semantic review, not red-pixel hunting.

    This only routes the next tool. It never changes maturity or releases a
    queue blocker; the existing fresh-image review and color checks do that.
    """
    if not (local is not None and getattr(local,'survey_batch',None) and candidate
            and candidate['status']=='unverified_measurement'
            and candidate.get('maturity')=='ripe'):
        return False
    proposal=next((p for p in local.survey_batch['proposals']
                   if p['proposal_id']==candidate.get('proposal_id')),None)
    return bool(proposal and proposal.get('depth_verified') is True
                and proposal.get('candidate_color_support')=='red_or_orange'
                and proposal.get('body_area_pixels',0)>=24
                and proposal.get('red_area_pixels',0)/proposal['body_area_pixels']<.35)


def body_semantics_review_required(local, candidate):
    return can_review_proposals(local) and body_color_conflict(local, candidate)


def body_context_colors(rgb, box):
    """Same local-color evidence used by the existing maturity correction gate."""
    hsv=np.asarray(rgb.convert('HSV'));x0,y0,x1,y1=box
    patch=hsv[max(0,y0-12):min(hsv.shape[0],y1+13),max(0,x0-12):min(hsv.shape[1],x1+13)]
    h,s,v=patch[...,0],patch[...,1],patch[...,2]
    colored=(s>=115)&(v>=30);n=int(colored.sum())
    return {'colored_pixels':n,
            'red_fraction':float((((h<8)|(h>244))&colored).sum()/max(1,n)),
            'orange_green_fraction':float((((h>=8)&(h<=65))&colored).sum()/max(1,n))}


def review_proposals(local, review, batch, measured, blocked, reason, rgb, base_pose,
                     duplicate_links=None, wrist_view=None):
    """Refresh the same image regions, preserving retries and all unresolved claims."""
    from local_targets import LocalTargets
    if not can_review_proposals(local):raise ValueError('No unresolved proposals or review budget exhausted')
    local.proposal_review_attempts = getattr(local, 'proposal_review_attempts', 0) + 1
    submitted_review=deepcopy(review)
    review=deepcopy(review)
    old = local.survey_review
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 1200:
        raise ValueError('Explain the previous visual error and new evidence')
    if review['frame_id'] == old['frame_id']:raise ValueError('A newer full image is required')
    if review['missing_visible_ripe'] != old['missing_visible_ripe']:
        raise ValueError('Proposal review cannot clear missing-fruit coverage; use correct_survey separately')
    before = {p['proposal_id']:p for p in old['candidates']}
    after = {p['proposal_id']:p for p in review['candidates']}
    previous = {p['proposal_id']:p for p in local.survey_batch['proposals']}
    current = {p['proposal_id']:p for p in batch['proposals']}
    if not (before.keys() == after.keys() == previous.keys() == current.keys()):
        raise ValueError('Cannot add, omit or renumber proposals during review')
    if batch['omitted_by_limit'] != local.survey_batch['omitted_by_limit']:
        raise ValueError('Cannot clear truncated coverage by re-reviewing')
    colors = {}; deferred=[]; changed_depth=[]
    for key, p in previous.items():
        q = current[key]
        if np.max(np.abs(np.asarray(p['bbox_xyxy'])-q['bbox_xyxy'])) > 2:
            raise ValueError('Proposal region changed; cannot relabel another fruit')
        x,y = q['pixel'];x0,y0,x1,y1 = p['bbox_xyxy']
        if not x0 <= x <= x1 or not y0 <= y <= y1:raise ValueError('Remeasurement left original image region')
        a,b = p['surface_m'], q['surface_m']
        if a is not None and (b is None or np.linalg.norm(np.asarray(a)-b) > .004):
            if not getattr(local,'allow_uncertain_deferral',False):
                raise ValueError('Previously measured body moved by more than 4 mm')
            changed_depth.append({'proposal_id':key,'previous_surface_m':a,'current_surface_m':b,
                                  'resolution':'retain uncertain; fresh body/stalk confirmation required'})
            after[key]['maturity']='uncertain'
        old_stage,new_stage = before[key]['maturity'], after[key]['maturity']
        colors[key]=body_context_colors(rgb,p['bbox_xyxy'])
        if any(item['proposal_id']==key for item in changed_depth):continue
        if old_stage != new_stage:
            if old_stage not in ('ripe','uncertain') or new_stage not in ('ripe','uncertain','turning','green'):
                raise ValueError('Cannot relabel a fruit claim as not_fruit. For a same-fruit fragment retain its maturity and submit duplicate_links with both-view evidence; otherwise retain uncertain')
            if new_stage in ('green','turning'):
                # Inspect the local body context, not the isolated red seed/edge.
                if colors[key]['colored_pixels']<24 or colors[key]['red_fraction']>=.35 or colors[key]['orange_green_fraction']<.5:
                    # Keep supported corrections to other fruit. This is an
                    # explicit unresolved fallback, never acceptance as unripe.
                    deferred.append({'proposal_id':key,'requested_maturity':new_stage,
                                     'effective_maturity':'uncertain',
                                     'reason':'Insufficient local orange/green evidence; retain uncertain claim'})
                    after[key]['maturity']='uncertain'
    if deferred or changed_depth:
        from visual_proposals import reviewed_candidates
        measured,blocked=reviewed_candidates(review,batch,review['frame_id'],rgb)
    from proposal_identity import check_duplicate_links
    links=getattr(local,'duplicate_links',[]) if duplicate_links is None else duplicate_links
    # Associations cannot disappear on a later semantic review and resurrect
    # an already failed target as a new candidate.
    previous_links=getattr(local,'duplicate_links',[])
    if not isinstance(links,list):raise ValueError('duplicate_links must be a list')
    if any(not any(isinstance(q,dict) and q.get('fragment_id')==p['fragment_id']
                   and q.get('parent_id')==p['parent_id'] for q in links) for p in previous_links):
        raise ValueError('Retain previously accepted duplicate associations at this station')
    duplicate_checks=check_duplicate_links(links,review['candidates'],batch,wrist_view)
    aliases={p['fragment_id']:p['parent_id'] for p in duplicate_checks}
    for src,dst in aliases.items():
        if before[src]['maturity'] not in ('uncertain','ripe') or before[dst]['maturity']!='ripe':
            raise ValueError('Duplicate association cannot promote an originally unripe or unresolved parent')
    # Visibility is new evidence and may improve after moving the arm. Losing
    # sight of a ripe claim must retain an unresolved candidate, not turn it
    # into "visually_unavailable" and silently authorize skipping/driving.
    measured=[deepcopy(p) for p in measured if p['proposal_id'] not in aliases]
    for item in measured:
        if item['maturity']=='ripe' and item['visibility']=='occluded':
            item['semantic_unverified']=True
    refreshed=LocalTargets();refreshed.register(measured,base_pose,review['coverage'])
    # A point lost after an arm/view change stays explicitly unresolved. Keep
    # its last known location for audit only; it cannot become a grasp target.
    for change in changed_depth:
        if change['current_surface_m'] is not None:continue
        prior=next((c for c in local.candidates if c.get('proposal_id')==change['proposal_id']),None)
        if prior is not None:
            retained=deepcopy(prior)
            retained.update(maturity='uncertain',status='unverified_semantics',
                            semantic_unverified=True,body_pixel_verified=False,
                            retained_last_known_measurement=True)
            refreshed.candidates.append(retained)
    refreshed.candidates.sort(key=lambda c:c['distance_xy_m'])
    # Transfer identity and every retry/terminal outcome; no fresh attempts on a
    # failed fruit, including duplicate image fragments merged by registration.
    used=set();next_key=max((int(c['candidate_key'].rsplit('-',1)[1]) for c in local.candidates),default=0)+1
    for item in refreshed.candidates:
        matches=[c for c in local.candidates if c.get('proposal_id')==item.get('proposal_id')
                 or aliases.get(c.get('proposal_id'))==item.get('proposal_id')
                 or np.linalg.norm(np.asarray(c['surface_m'])-item['surface_m'])<.035]
        if matches:
            prior=max(matches,key=lambda c:c['failures'])
            item['failures']=prior['failures']
            item['correction_attempts']=max(c.get('correction_attempts',0) for c in matches)
            item['observation_adjusted']=any(c.get('observation_adjusted',False) for c in matches)
            for name,value in prior.items():
                if name not in item:item[name]=deepcopy(value)
            if any(c['status']=='unavailable' for c in matches):item['status']='unavailable'
            key=prior['candidate_key']
        else:key=None
        if key is None or key in used:
            key=f'view-candidate-{next_key}';next_key+=1
        item['candidate_key']=key;used.add(key)
        # Keep explicit provenance for semantic recovery followed by mature
        # confirmation. A model claim without an executed adjustment is not
        # enough to defer a candidate at this station.
        completed=item.get('observation_recovery_completed_frame')
        frame=review['frame_id']
        claim=before.get(item.get('proposal_id'),{})
        if (claim.get('maturity')=='uncertain' and item['maturity']=='ripe'
                and item['status']=='pending' and item.get('body_pixel_verified') is True
                and item.get('observation_adjusted') and completed is not None
                and isinstance(frame,str) and frame.startswith('f') and frame[1:].isdigit()
                and int(frame[1:])>completed):
            item['semantic_recovery_review_frame']=int(frame[1:])
    entry={'previous_review':deepcopy(old),'review':deepcopy(review),'correction_reason':reason,
           'submitted_review':submitted_review,'deferred_corrections':deferred,'changed_depth_regions':changed_depth,
           'color_checks':colors,'duplicate_checks':duplicate_checks,
           'remaining_block':blocked,'method':'bounded_same_region_RGBD_review',
           'visibility_changes':{k:{'before':before[k]['visibility'],'after':after[k]['visibility']}
                                 for k in before if before[k]['visibility']!=after[k]['visibility']}}
    local.candidates=refreshed.candidates;local.coverage=review['coverage'];local.selection_blocked_reason=blocked
    local.survey_review=deepcopy(review);local.survey_batch=deepcopy(batch)
    local.duplicate_links=deepcopy(links)
    entry['unresolved_claims']=[{'proposal_id':c.get('proposal_id'),
        'candidate_key':c['candidate_key'],'status':c['status'],
        'body_color_context':colors.get(c.get('proposal_id')),
        'ripe_color_conflict':body_color_conflict(local,c)}
        for c in local.candidates if c['status'] in ('unverified_measurement','unverified_semantics')]
    entry['decision_required']='Review recorded, not maturity approval. For orange-dominant bodies reconsider turning/green using the whole image; if identity or maturity is unclear retain uncertain. Keeping ripe does not clear a color conflict. Never hunt a red pixel or drive past unresolved candidates.'
    if getattr(local,'allow_uncertain_deferral',False):
        entry['decision_required']='Review recorded. Keep unknown regions uncertain; they do not veto another clear ripe candidate. Follow local_selection.active for fresh stalk localization; if none is available, use the advertised observation-only search to obtain a new view.'
    return entry


def can_correct_survey(local):
    return bool(local is not None and local.selection_blocked_reason == MISSING_BLOCK
                and getattr(local, 'survey_review', None)
                and getattr(local, 'survey_correction_attempts', 0) < 2)


def correct_survey(local, review, batch, blocked, reason):
    """Revise only the coverage claim, preserving every candidate and retry.

    Caller verifies a fresh whole-frame review and current RGB-D. This remains
    model visual evidence, not an automatic guarantee that no fruit was missed.
    """
    if not can_correct_survey(local):
        raise ValueError('Only a missing-fruit coverage block can be corrected, at most twice')
    local.survey_correction_attempts = getattr(local, 'survey_correction_attempts', 0) + 1
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 1200:
        raise ValueError('Explain the original omission claim and the current visual evidence')
    old = local.survey_review
    if review['frame_id'] == old['frame_id']:
        raise ValueError('Coverage correction requires a newer observation')
    before = {p['proposal_id']: p for p in old['candidates']}
    after = {p['proposal_id']: p for p in review['candidates']}
    if before.keys() != after.keys() or any(
            before[k][field] != after[k][field] for k in before for field in ('maturity','visibility')):
        raise ValueError('Coverage correction cannot add/remove/relabel fruit; retain the original queue')
    previous = {p['proposal_id']: p for p in local.survey_batch['proposals']}
    current = {p['proposal_id']: p for p in batch['proposals']}
    if previous.keys() != current.keys():
        raise ValueError('Proposal correspondence changed; coverage correction cannot reset the queue')
    for key in previous:
        a, b = previous[key]['surface_m'], current[key]['surface_m']
        # Tiny turning/non-fruit regions may intentionally have no usable
        # depth and never enter the queue. Preserve their negative review and
        # exact image region; do not mistake them for changed ripe bodies.
        if (a is None and b is None and before[key]['maturity'] in ('green','turning','not_fruit')
                and previous[key]['pixel']==current[key]['pixel']
                and previous[key]['bbox_xyxy']==current[key]['bbox_xyxy']):
            continue
        if a is None or b is None or np.linalg.norm(np.asarray(a)-b) > .004:
            raise ValueError('Body RGB-D correspondence changed by more than 4 mm')
    entry = {'previous_review': deepcopy(old), 'review': deepcopy(review),
             'correction_reason': reason, 'remaining_block': blocked,
             'method': 'model_whole_frame_coverage_correction',
             'note': 'Model evidence only; candidate identities, order and retry counts unchanged.'}
    local.survey_corrections = getattr(local, 'survey_corrections', []) + [entry]
    local.selection_blocked_reason = blocked
    local.coverage = review['coverage']
    if blocked is None:
        local.survey_review = deepcopy(review)
        local.survey_batch = deepcopy(batch)
    return entry


def uncertain_region_evidence(local, candidate):
    """Measured color region permits looking again, never fruit certification.

    Reuse the detector's >=9 red-pixel and coherent-depth evidence after a
    semantic review. A failed red-body center check remains failed.
    """
    if not (candidate and candidate.get('status')=='unverified_semantics'
            and candidate.get('maturity')=='uncertain'
            and candidate.get('body_pixel_verified') is False
            and getattr(local,'proposal_review_attempts',0)>=1):return None
    batch=getattr(local,'survey_batch',None) or {}
    review=getattr(local,'survey_review',None) or {}
    frame=review.get('frame_id')
    if not frame or candidate.get('source_frame')!=frame:return None
    region=next((p for p in batch.get('proposals',[])
                 if p.get('proposal_id')==candidate.get('proposal_id')),None)
    if not (region and region.get('depth_verified') is True
            and region.get('candidate_color_support')=='red_or_orange'
            and region.get('red_area_pixels',0)>=9):return None
    try:
        point=np.asarray(region['surface_m'],dtype=float)
        body=np.asarray(candidate['surface_m'],dtype=float)
        if (point.shape!=(3,) or body.shape!=(3,) or not np.isfinite(point).all()
                or not np.isfinite(body).all() or np.linalg.norm(point-body)>.004):return None
        distance=float(region['distance_xy_m'])
        if not (0<=distance<=.56 and .12<=point[2]<=.46):return None
    except (KeyError,TypeError,ValueError):return None
    return {'frame_id':frame,'proposal_id':region['proposal_id'],
            'candidate_key':candidate['candidate_key'],'red_area_pixels':region['red_area_pixels'],
            'surface_m':point.tolist(),'body_pixel_verified':False,
            'note':'Depth-supported color region only; fruit identity and maturity remain uncertain.'}


def uncertain_review_candidate(local):
    """A measured unresolved observation, never an approved grasp target."""
    if local is None or local.active or local.effective_selection_blocked_reason:return None
    candidate=next((c for c in local.candidates if c['status'] in ('unverified_semantics','unverified_measurement')),None)
    if candidate and candidate['status']=='unverified_semantics' and (
            candidate.get('body_pixel_verified') is True or uncertain_region_evidence(local,candidate)):
        return candidate
    return None


def missing_stalk_views(session, frame=None):
    """Fresh measured views may authorize looking again, never skipping fruit."""
    local = getattr(session, 'local_targets', None)
    active = local.active if local is not None else None
    evidence = getattr(session, 'stalk_view_evidence', None)
    if not (active and active.get('body_pixel_verified') is True and evidence
            and evidence['candidate_key'] == active['candidate_key']
            and (frame is None or evidence['frame_id'] == frame)):
        return False
    views = evidence['views']
    return (set(views) == {'harvest_close', 'wrist'}
            and all(v in ('no_fit', 'outside_crop') for v in views.values())
            and 'no_fit' in views.values())


def unconfirmed_stalk_options(session, frame=None, *, call_charged=False):
    """Current geometric fits may be reviewed as uncertain, never skipped as fruit."""
    local=getattr(session,'local_targets',None)
    active=local.active if local is not None else None
    evidence=getattr(session,'stalk_view_evidence',None)
    if not (active and active.get('body_pixel_verified') is True
            and active.get('maturity')=='ripe' and not local.effective_selection_blocked_reason
            and getattr(session,'centered',None) is not None
            and session.target_stage is None and not session.grasp_attempted
            and not session.target_confirmed and not getattr(session,'motion_stop_trace',None)
            and not getattr(session,'preclose_recovery_failed',False)
            and not active.get('observation_adjusted')
            and getattr(local,'observation_adjustments',0)<2
            and getattr(session,'station_attempts',0)<11+int(call_charged)
            and evidence and evidence['candidate_key']==active['candidate_key']
            and evidence['frame_id'] in getattr(session,'frames',{})
            and (frame is None or evidence['frame_id']==frame)):
        return []
    views=evidence['views']
    if set(views)!={'harvest_close','wrist'} or not all(v in ('fit_available','no_fit','outside_crop') for v in views.values()):return []
    fits=[p for p in getattr(session,'stalk_proposals',{}).values()
          if p['frame_id']==evidence['frame_id'] and p['candidate_key']==active['candidate_key']]
    if not fits or {p['camera'] for p in fits}!={c for c,v in views.items() if v=='fit_available'}:return []
    return [{'proposal_id':p['proposal_id'],'camera':p['camera']} for p in fits]


def review_unconfirmed_stalks(session, claims, frame):
    options=unconfirmed_stalk_options(session,frame,call_charged=True)
    if not options:raise ValueError('No current same-candidate fitted stalks eligible for observation review')
    if not isinstance(claims,list) or not claims:raise ValueError('List ALL current unconfirmed stalk proposals with image reasons')
    for c in claims:
        if (not isinstance(c,dict) or set(c)!={'proposal_id','reason'}
                or not isinstance(c['proposal_id'],str) or not isinstance(c['reason'],str)
                or not 1<=len(c['reason'].strip())<=600):
            raise ValueError('Each unconfirmed proposal requires its current ID and image reason')
    ids=[c['proposal_id'] for c in claims]
    if len(set(ids))!=len(ids) or set(ids)!={p['proposal_id'] for p in options}:
        raise ValueError('Review ALL current fitted stalk proposals; no omitted, stale or invented IDs')
    return {'frame_id':frame,'candidate_key':session.local_targets.active['candidate_key'],
            'proposals':deepcopy(options),'model_claims':deepcopy(claims),
            'note':'Model cannot confirm stalk identity; not proof of wrong fit, fruit absence, or permission to skip/grasp.'}


def observation_adjustment_available(session, frame=None, *, call_charged=False):
    local = getattr(session, 'local_targets', None)
    active = local.active if local is not None else None
    uncertain = uncertain_review_candidate(local)
    if active is None:active=uncertain
    evidence=bool(active and (active.get('cross_view_failed') or missing_stalk_views(session, frame)))
    if uncertain is not None:
        evidence=(0 < getattr(local,'proposal_review_attempts',0) <= 2
                  and getattr(session,'station_attempts',0)<11+int(call_charged))
    return bool(getattr(session, 'centered', None) is not None and active
                and session.target_stage is None and not session.grasp_attempted
                and not getattr(session, 'motion_stop_trace', None)
                and not getattr(session, 'preclose_recovery_failed', False)
                and not session.target_confirmed and evidence
                and not active.get('observation_adjusted')
                and getattr(local, 'observation_adjustments', 0) < 2)


def out_of_view_after_recovery(session, frame=None):
    """Defer a measured stalk outside a camera after completed semantic recovery.

    Requires an actual post-recovery fit/maturity check and a new calibrated
    projection. A missing crop or model claim alone never authorizes search.
    """
    local=getattr(session,'local_targets',None)
    active=local.active if local is not None else None
    evidence=getattr(session,'stalk_view_evidence',None)
    if not (active and not local.effective_selection_blocked_reason
            and active.get('maturity')=='ripe' and active.get('body_pixel_verified') is True
            and active.get('observation_adjusted') and active.get('failures')==1
            and active.get('cross_view_failed') and getattr(session,'centered',None) is not None
            and session.target_stage is None and not session.grasp_attempted
            and not session.target_confirmed and not getattr(session,'motion_stop_trace',None)
            and not getattr(session,'preclose_recovery_failed',False)
            and evidence and evidence['candidate_key']==active['candidate_key']
            and (frame is None or evidence['frame_id']==frame)):
        return False
    markers=[active.get('observation_recovery_completed_frame'),
             active.get('semantic_recovery_review_frame'),active.get('cross_view_failure_frame')]
    current=evidence['frame_id']
    if not (all(type(n) is int for n in markers) and current.startswith('f') and current[1:].isdigit()
            and markers[0]<markers[1]<markers[2]<int(current[1:])):
        return False
    point=active.get('cross_view_target_m')
    if point is None or np.asarray(point).shape!=(3,) or not np.isfinite(point).all():return False
    views=evidence['views'];checks=active.get('cross_view_checks',{})
    if set(views)!={'harvest_close','wrist'} or 'fit_available' not in views.values():return False
    from vista import verify_projected_point
    for camera,check in checks.items():
        if check.get('reason')!='outside_image' or views.get(camera)!='outside_crop':continue
        path=getattr(session,'depth_frames',{}).get((current,camera))
        if path is None:continue
        with np.load(path) as sample:
            fresh=verify_projected_point(np.asarray(point),sample['depth'],sample)
        if fresh.get('reason')=='outside_image':return True
    return False


def exhausted_observation_recovery(session, frame=None):
    """Completed recovery with no fits after localization or semantic review.

    This exhausts the current station's bounded attempt, not fruit identity,
    maturity or global reachability. No inference from an empty initial view.
    """
    if out_of_view_after_recovery(session,frame):return True
    local=getattr(session,'local_targets',None)
    active=local.active if local is not None else None
    evidence=getattr(session,'stalk_view_evidence',None)
    if not (active and active.get('maturity')=='ripe'
            and active.get('observation_adjusted')
            and active.get('observation_recovery_completed_frame') is not None
            and getattr(session,'centered',None) is not None
            and session.target_stage is None and not session.grasp_attempted
            and not session.target_confirmed
            and not getattr(session,'motion_stop_trace',None)
            and not getattr(session,'preclose_recovery_failed',False)
            and missing_stalk_views(session,frame)):
        return False
    observed=evidence['frame_id']
    if not (isinstance(observed,str) and observed.startswith('f') and observed[1:].isdigit()
            and int(observed[1:])>active['observation_recovery_completed_frame']):return False
    if active.get('cross_view_failed') and active.get('failures')==1:return True
    reviewed=active.get('semantic_recovery_review_frame')
    return bool(active.get('failures')==0 and isinstance(reviewed,int)
                and active['observation_recovery_completed_frame']<reviewed<int(observed[1:]))
