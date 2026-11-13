"""Local-first queue of model-proposed RGB-D observations, never scene IDs."""
import numpy as np
from copy import deepcopy


class CandidateCorrespondenceError(ValueError):
    """The selected RGB-D surface cannot be assigned to the active fruit."""


def body_pixel_evidence(rgb, x, y):
    """Check a claimed ripe-body pixel, not fruit identity or global absence."""
    hsv=np.asarray(rgb.convert('HSV'))
    patch=hsv[max(0,y-1):y+2,max(0,x-1):x+2]
    h,s,v=patch[...,0],patch[...,1],patch[...,2]
    red=((h<8)|(h>244))&(s>=115)&(v>=30)
    fraction=float(red.mean())
    return {'passed':fraction>=.5,'red_fraction':fraction,
            'selected_rgb':list(rgb.convert('RGB').getpixel((x,y))),
            'method':'3x3 RGB red-body consistency; not semantic identity or maturity certification'}


def candidate_crop_region(point, calibration, image_size):
    """Project a context box around a measured body surface, not a fruit ID."""
    w,h=image_size
    offsets=np.array([[x,y,z] for x in (-.03,.03) for y in (-.03,.03) for z in (-.025,.085)])
    points=np.vstack([np.asarray(point),np.asarray(point)+offsets])
    local=(points-np.asarray(calibration['position']))@np.asarray(calibration['rotation'])
    if not np.isfinite(local).all() or np.any(local[:,2]>=-.01):return None
    focal=h/(2*np.tan(np.deg2rad(calibration['fovy'])/2))
    uv=np.column_stack([w/2-.5+focal*local[:,0]/-local[:,2],h/2-.5-focal*local[:,1]/-local[:,2]])
    if not (0<=uv[0,0]<w and 0<=uv[0,1]<h):return None
    low=np.maximum([0,0],np.floor(uv.min(0)-6)).astype(int)
    high=np.minimum([w,h],np.ceil(uv.max(0)+7)).astype(int)
    return [int(low[0]),int(low[1]),int(high[0]-low[0]),int(high[1]-low[1])],uv[0].tolist()


class LocalTargets:
    def __init__(self):
        self.allow_uncertain_deferral = False
        self.surveyed = False
        self.candidates = []
        self.coverage = None
        self.selection_blocked_reason = None

    def register(self, candidates, base_pose, coverage):
        if self.surveyed:
            raise ValueError('Station already surveyed; cannot reset candidate failures before moving')
        measured = []
        for item in candidates:
            point = np.asarray(item['surface_m'], float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError('Invalid measured candidate')
            if not item.get('semantic_unverified') and item.get('body_pixel_verified') is not False and any(
                    not old.get('semantic_unverified') and old['maturity']==item['maturity'] and
                    old.get('body_pixel_verified') is not False and
                    np.linalg.norm(point-np.asarray(old['surface_m'])) < .035 for old in measured):
                continue  # Multiple pixels on one visible fruit are one candidate.
            candidate = dict(item)
            candidate.update(distance_xy_m=float(np.linalg.norm(point[:2]-np.asarray(base_pose)[:2])),
                             failures=0, status='pending' if item['maturity']=='ripe' and item['visibility']!='occluded' else 'visually_unavailable')
            if not .12<=point[2]<=.46 or candidate['distance_xy_m']>.56:
                candidate['status']='outside_working_envelope'
            if item.get('body_pixel_verified') is False:
                candidate['status']='unverified_measurement'
            if item.get('semantic_unverified'):
                candidate['status']='unverified_semantics'
            measured.append(candidate)
        measured.sort(key=lambda item: item['distance_xy_m'])
        for i,item in enumerate(measured):item['candidate_key']=f'view-candidate-{i+1}'
        self.candidates=measured;self.coverage=coverage;self.surveyed=True

    def begin_body_correction(self, key, x, y):
        """Bound corrections around the original image claim; never reset retries."""
        item=next((c for c in self.candidates if c['candidate_key']==key),None)
        if item is None or item['status']!='unverified_measurement':
            raise ValueError('Only an unverified measurement can be corrected')
        if item.get('correction_attempts',0)>=2:
            raise ValueError('Two body corrections exhausted; unresolved evidence still blocks search')
        item['correction_attempts']=item.get('correction_attempts',0)+1
        if type(x) is not int or type(y) is not int or not 0<=x<480 or not 0<=y<360:
            raise ValueError('Pixel outside image')
        if np.linalg.norm(np.array([x-item['x'],y-item['y']]))>32:
            raise ValueError('Correction must stay within 32 pixels of the ORIGINAL survey point; no relabelling a distant fruit')
        return item

    def correct_body(self, item, measurement, base_pose):
        """Commit measured evidence only. Failed corrections leave the claim unresolved."""
        evidence={'x':measurement['x'],'y':measurement['y'],'source_frame':measurement['source_frame'],
                  'body_pixel_check':measurement['body_pixel_check']}
        item.setdefault('correction_history',[]).append(evidence)
        if measurement['body_pixel_verified'] is not True:
            return False,'Corrected pixel still lacks red-body evidence'
        point=np.asarray(measurement['surface_m'],float)
        distance=float(np.linalg.norm(point[:2]-np.asarray(base_pose)[:2]))
        if not np.isfinite(point).all() or not .12<=point[2]<=.46 or distance>.56:
            return False,'Correction must measure a body in the working envelope; cannot clear unknown evidence by choosing background'
        if any(c is not item and c.get('body_pixel_verified') is not False and
               np.linalg.norm(point-np.asarray(c['surface_m']))<.035 for c in self.candidates):
            return False,'Correction overlaps an existing candidate; unresolved claim retained'
        item['original_measurement']={k:item.get(k) for k in ('x','y','source_frame','surface_m','body_pixel_check')}
        item.update(measurement,distance_xy_m=distance,status='pending')
        self.candidates.sort(key=lambda c:c['distance_xy_m'])
        return True,'Remeasured local red body; stalk and maturity validation still required'

    @property
    def effective_selection_blocked_reason(self):
        reason=self.selection_blocked_reason
        if not reason or not self.allow_uncertain_deferral:return reason
        # Retain the raw missing-depth evidence in status, but it is not a
        # global veto on another measured ripe target in collection mode.
        import re
        # The historical depth message itself contains a semicolon.
        cleaned=re.sub(r'P\d+ has unresolved body depth; finish(?:; )?', '', reason).strip('; ')
        return cleaned or None

    @property
    def active(self):
        if self.effective_selection_blocked_reason:return None
        for candidate in self.candidates:
            # Legacy mode blocks on the nearest unknown. Collection mode
            # retains it while selecting another measured ripe candidate.
            if candidate['status'] in ('unverified_measurement','unverified_semantics'):
                if self.allow_uncertain_deferral:continue
                return None
            if candidate['status'] in ('pending','ready'):return candidate
        return None

    @property
    def can_search(self):
        return self.surveyed and not self.selection_blocked_reason and self.active is None and not any(c['status'] in ('unverified_measurement','unverified_semantics') for c in self.candidates)

    def require_active(self):
        if not self.surveyed:raise ValueError('First survey_candidates across the current view; no localization or driving yet')
        if self.effective_selection_blocked_reason:raise ValueError(self.effective_selection_blocked_reason)
        if self.active is None:
            if self.allow_uncertain_deferral:
                raise ValueError('No confirmed local candidate for localization. Retain unknown regions; follow station_search.next_action for review or observation-only search, then select from fresh images.')
            blocked=next((c for c in self.candidates if c['status'] in ('unverified_measurement','unverified_semantics')),None)
            if blocked is not None:
                if blocked['status']=='unverified_semantics':
                    raise ValueError('Nearer proposal maturity/identity uncertain; follow station_search.next_action for visual review/observation recovery, or finish. No stalk localization, skipping or driving yet')
                remaining=max(0,2-blocked.get('correction_attempts',0))
                instruction=(f"Use correct_candidate_body for {blocked['candidate_key']} ({remaining} corrections left), or finish"
                             if remaining else 'Body corrections exhausted; finish')
                from perception_recovery import can_review_proposals
                if can_review_proposals(self):
                    instruction='Use review_proposals on a fresh full numbered image to recheck whole-body maturity, or finish; body correction limits do not exhaust semantic review'
                raise ValueError(f"Selection blocked by nearer unverified body measurement {blocked['candidate_key']}; "
                                 f"candidates are NOT exhausted. {instruction}. Do not skip to a farther candidate or drive")
            raise ValueError('No remaining local candidate; finish or use permitted search')
        return self.active

    def check_target(self, target):
        active=self.require_active()
        delta=np.asarray(target)-np.asarray(active['surface_m'])
        # The model points to the visible fruit body, then selects its stalk.
        # This spatial neighbourhood is correspondence, not an oracle identity.
        if np.linalg.norm(delta[:2])>.035 or not -.005<=delta[2]<=.085:
            raise CandidateCorrespondenceError('Selected surface is outside the nearest active fruit neighbourhood; inspect its body marker. No candidate localization failure counted.')

    def fail(self, reason, terminal=False):
        active=self.require_active()
        active['failures']+=1;active['last_reason']=reason
        if terminal or active['failures']>=2:active['status']='unavailable'

    def ready(self):
        self.require_active()['status']='ready'

    def status(self):
        blocked=next((c['candidate_key'] for c in self.candidates if c['status']=='unverified_measurement'),None) if self.active is None else None
        return {'surveyed':self.surveyed, 'active':deepcopy(self.active), 'candidates':deepcopy(self.candidates),
                'selection_blocked_reason':self.effective_selection_blocked_reason,
                'deferred_depth_evidence':self.selection_blocked_reason if self.allow_uncertain_deferral and self.selection_blocked_reason!=self.effective_selection_blocked_reason else None,
                'uncertain_deferral_enabled':self.allow_uncertain_deferral,
                'survey_correction_attempts':getattr(self,'survey_correction_attempts',0),
                'proposal_review_attempts':getattr(self,'proposal_review_attempts',0),
                'duplicate_links':deepcopy(getattr(self,'duplicate_links',[])),
                'observation_adjustments':getattr(self,'observation_adjustments',0),
                'unverified_semantics':sum(c['status']=='unverified_semantics' for c in self.candidates),
                'blocked_by_unverified_candidate':blocked,
                'unverified_measurements':sum(c['status']=='unverified_measurement' for c in self.candidates),
                'local_candidates_exhausted':self.can_search, 'coverage':self.coverage,
                'distance_rule':'horizontal distance from robot base to model-selected RGB-D fruit surface',
                'note':'Ranked only among visually proposed candidates. Keys are observation labels, not simulator fruit IDs. Maturity and reachability still require stalk validation.'}
