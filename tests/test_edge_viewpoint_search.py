
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from PIL import Image
from local_targets import LocalTargets
from perception_recovery import measured_edge_fragments,edge_viewpoint_search_available
from vista import Session


class EdgeViewpointTests(unittest.TestCase):
    def setUp(self):
        self.rgb=Image.new('RGB',(80,60),'red');self.depth=np.full((60,80),.4)
        self.p=dict(proposal_id='P1',bbox_xyxy=[77,30,79,45],pixel=[78,37],
                    body_area_pixels=48,red_area_pixels=48,surface_m=[.3,0,.25],depth_verified=True)
        self.local=LocalTargets();self.local.register([dict(proposal_id='P1',surface_m=[.3,0,.25],
            maturity='uncertain',visibility='occluded',semantic_unverified=True,body_pixel_verified=True)],
            [0,0,0],'whole image')
        self.local.survey_batch=dict(proposals=[self.p],omitted_by_limit=0)
        self.local.proposal_review_attempts=2
        self.s=SimpleNamespace(local_targets=self.local,centered=Mock(),target_stage=None,
            grasp_attempted=False,target_confirmed=False,frames={'f10':{}},
            edge_viewpoint_evidence=dict(frame_id='f10',fragments=self.measure()))
    def measure(self,point=None):
        return measured_edge_fragments(self.local,self.rgb,self.depth,{},[0,0,0],
                                       lambda *args:np.array(point or [.3,0,.25]))
    def test_allows_only_observation_not_grasp_or_exhausted_claim(self):
        self.assertTrue(edge_viewpoint_search_available(self.s,'f10'))
        self.assertFalse(self.local.can_search);self.assertIsNone(self.local.active)
        self.assertEqual(self.local.candidates[0]['maturity'],'uncertain')
        with self.assertRaises(ValueError):self.local.require_active()
        with self.assertRaises(ValueError):Session.require_local_search(self.s)
        Session.require_local_search(self.s,observation_only=True)
    def test_nonedge_or_wide_or_unknown_or_changed_depth_not_accepted(self):
        for change in ({'bbox_xyxy':[40,30,42,45]}, {'bbox_xyxy':[60,30,79,45]},
                       {'surface_m':None}, {'body_area_pixels':200}):
            old=deepcopy(self.p);self.p.update(change)
            self.assertEqual(self.measure(),[]);self.p.clear();self.p.update(old)
        self.assertEqual(self.measure([.31,0,.25]),[])
        self.depth[:]=np.nan;self.assertEqual(self.measure(),[])
    def test_ready_target_or_unverified_depth_or_nonedge_uncertainty_blocks(self):
        for status in ('pending','ready','unverified_measurement','unverified_semantics'):
            c=deepcopy(self.local.candidates[0]);c.update(candidate_key='other',status=status)
            self.local.candidates.append(c)
            self.assertFalse(edge_viewpoint_search_available(self.s,'f10'))
            self.local.candidates.pop()
    def test_stale_frame_budget_coverage_motion_guards(self):
        self.assertFalse(edge_viewpoint_search_available(self.s,'f11'))
        for obj,key,value in ((self.local,'proposal_review_attempts',1),
                (self.local,'selection_blocked_reason','missing ripe fruit'),
                (self.s,'frames',{}),(self.s,'grasp_attempted',True),(self.s,'fixed_base',True),
                (self.s,'target_stage','pregrasp'),(self.s,'motion_stop_trace',{'stopped':True})):
            old=getattr(obj,key,None);setattr(obj,key,value)
            self.assertFalse(edge_viewpoint_search_available(self.s,'f10'))
            setattr(obj,key,old)
    def test_search_retains_movement_and_call_budgets(self):
        self.s.e=SimpleNamespace(meta={'tilted_stalk_prior':True,'mobile_layout':True})
        self.s.search_steps=4;self.s.max_calls=32;self.s.calls=9
        self.s.require_local_search=lambda **kw:Session.require_local_search(self.s,**kw)
        with self.assertRaisesRegex(ValueError,'budget exhausted'):Session.search_row(self.s,1)
        self.s.search_steps=0;self.s.calls=23
        with self.assertRaisesRegex(ValueError,'budget exhausted'):Session.search_row(self.s,1)
        self.assertEqual(self.local.candidates[0]['status'],'unverified_semantics')


if __name__=='__main__':unittest.main()
