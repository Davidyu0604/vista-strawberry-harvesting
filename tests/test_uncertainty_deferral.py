import sys,unittest
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock,patch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
from local_targets import LocalTargets
from perception_recovery import uncertain_viewpoint_search_available,review_proposals,MISSING_BLOCK,current_body_proposals,observation_adjustment_available
from visual_proposals import reviewed_candidates
from vista import Session

class UncertaintyDeferralTests(unittest.TestCase):
 def fixture(self):
  rgb=Image.new('RGB',(100,80),'red')
  batch=dict(omitted_by_limit=0,proposals=[dict(proposal_id='P1',pixel=[20,30],bbox_xyxy=[15,25,25,35],surface_m=[.2,0,.25]),dict(proposal_id='P2',pixel=[60,30],bbox_xyxy=[55,25,65,35],surface_m=[.3,0,.25])])
  review=dict(frame_id='f10',coverage='whole frame',missing_visible_ripe=False,candidates=[dict(proposal_id='P1',maturity='uncertain',visibility='partial',reason='fragment'),dict(proposal_id='P2',maturity='ripe',visibility='clear',reason='red body')])
  measured,blocked=reviewed_candidates(review,batch,'f10',rgb)
  local=LocalTargets();local.register(measured,[0,0,0],'whole frame');local.allow_uncertain_deferral=True
  local.survey_review=deepcopy(review);local.survey_batch=deepcopy(batch);local.selection_blocked_reason=blocked;local.proposal_review_attempts=1
  s=SimpleNamespace(local_targets=local,centered=Mock(),target_stage=None,grasp_attempted=False,target_confirmed=False,frames={'f11':{}},proposal_frames={'f11':deepcopy(batch)})
  return s,rgb,batch,review
 def test_clear_ripe_target_is_not_blocked_by_nearer_uncertain_region(self):
  s,*_=self.fixture();local=s.local_targets
  self.assertEqual(local.active['proposal_id'],'P2');self.assertEqual(local.require_active()['proposal_id'],'P2')
  self.assertEqual(local.candidates[0]['maturity'],'uncertain');self.assertEqual(local.candidates[0]['failures'],0)
  self.assertFalse(uncertain_viewpoint_search_available(s,'f11'))
  local.allow_uncertain_deferral=False;self.assertIsNone(local.active)
 def test_depth_only_block_does_not_veto_other_target_but_coverage_still_does(self):
  s,*_=self.fixture();local=s.local_targets;local.selection_blocked_reason='P3 has unresolved body depth; finish'
  self.assertEqual(local.require_active()['proposal_id'],'P2')
  self.assertIsNotNone(local.status()['deferred_depth_evidence'])
  local.selection_blocked_reason += '; '+MISSING_BLOCK
  self.assertIsNone(local.active)
  with self.assertRaises(ValueError):local.require_active()
 def test_uncertainty_allows_new_view_without_changing_candidates(self):
  s,*_=self.fixture();s.local_targets.candidates[1]['status']='visually_unavailable'
  before=deepcopy(s.local_targets.candidates)
  self.assertTrue(uncertain_viewpoint_search_available(s,'f11'))
  Session.require_local_search(s,observation_only=True)
  self.assertEqual(s.local_targets.candidates,before)
  self.assertFalse(uncertain_viewpoint_search_available(s,'f12'))
  for key,value in [('grasp_attempted',True),('target_stage','pregrasp'),('fixed_base',True),('motion_stop_trace',{'stopped':True}),('calls',23),('search_steps',4)]:
   existed=hasattr(s,key);old=getattr(s,key,None);setattr(s,key,value);self.assertFalse(uncertain_viewpoint_search_available(s,'f11'))
   if not existed:delattr(s,key)
   else:setattr(s,key,old)
 def test_depth_change_updates_one_candidate_and_keeps_other_review(self):
  for point in (None,[.24,0,.25]):
   with self.subTest(point=point):
    s,rgb,batch,review=self.fixture();batch['proposals'][0]['surface_m']=point
    review['frame_id']='f11';review['candidates'][0]['maturity']='ripe'
    measured,blocked=reviewed_candidates(review,batch,'f11',rgb)
    entry=review_proposals(s.local_targets,review,batch,measured,blocked,'fresh view',rgb,[0,0,0])
    self.assertEqual(len(entry['changed_depth_regions']),1)
    self.assertEqual(s.local_targets.active['proposal_id'],'P2')
    old=next(c for c in s.local_targets.candidates if c['proposal_id']=='P1')
    self.assertEqual(old['maturity'],'uncertain');self.assertEqual(old['status'],'unverified_semantics')
    self.assertEqual(old['failures'],0)
 def test_exhausted_reviews_still_refresh_same_regions_for_viewpoint_search(self):
  s,rgb,batch,_=self.fixture();local=s.local_targets;local.proposal_review_attempts=2
  with patch('visual_proposals.refresh_proposals',return_value=batch) as refresh,patch('visual_proposals.propose_bodies') as detect:
   self.assertEqual(current_body_proposals(local,rgb,None,None,[0,0,0],None),batch)
  refresh.assert_called_once();detect.assert_not_called()
 def test_missing_depth_elsewhere_does_not_block_active_fruit_observation(self):
  s,*_=self.fixture();s.local_targets.selection_blocked_reason='P3 has unresolved body depth; finish'
  s.local_targets.active['cross_view_failed']=True
  self.assertTrue(observation_adjustment_available(s))
  s.local_targets.selection_blocked_reason=MISSING_BLOCK
  self.assertFalse(observation_adjustment_available(s))

if __name__=='__main__':unittest.main()
