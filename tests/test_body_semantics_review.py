
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from local_targets import LocalTargets
from perception_recovery import (body_semantics_review_required, can_review_proposals,
                                 review_proposals, body_color_conflict)
from visual_proposals import reviewed_candidates
from vista import Session


class BodySemanticsReviewTests(unittest.TestCase):
    def setUp(self):
        self.rgb=Image.new('RGB',(80,80),(230,90,25))
        self.batch={'omitted_by_limit':0,'proposals':[dict(proposal_id='P1',pixel=[40,40],
            bbox_xyxy=[20,20,60,60],surface_m=[.3,0,.25],depth_verified=True,
            candidate_color_support='red_or_orange',body_area_pixels=1000,red_area_pixels=30)]}
        self.review=dict(frame_id='f1',coverage='whole image',missing_visible_ripe=False,
            candidates=[dict(proposal_id='P1',maturity='ripe',visibility='clear',reason='mistaken red claim')])
        measured,blocked=reviewed_candidates(self.review,self.batch,'f1',self.rgb)
        self.local=LocalTargets();self.local.register(measured,[0,0,0],'whole image')
        self.local.survey_review=deepcopy(self.review);self.local.survey_batch=deepcopy(self.batch)
        self.candidate=self.local.candidates[0]

    def test_whole_body_conflict_requires_review_without_relabeling(self):
        self.assertTrue(body_semantics_review_required(self.local,self.candidate))
        self.assertEqual(self.candidate['maturity'],'ripe')
        self.assertFalse(self.local.can_search);self.assertIsNone(self.local.active)

    def test_pixel_correction_cannot_hunt_red_flecks_or_consume_correction_budget(self):
        session=Session.__new__(Session);session.local_targets=self.local
        session.target_stage=None;session.grasp_attempted=False;session.e=SimpleNamespace();session.max_wait=90
        args=dict(frame_id='f2',candidate_key=self.candidate['candidate_key'],x=42,y=42,reason='try another pixel')
        with patch('vista.robot_state'),patch('vista.check_observation'):
            with self.assertRaisesRegex(ValueError,'review_proposals'):
                session._dispatch('correct_candidate_body',args,'f2',{},0)
        self.assertEqual(self.candidate.get('correction_attempts',0),0)
        self.assertFalse(self.local.can_search)

    def test_pixel_budget_exhaustion_does_not_hide_remaining_semantic_review(self):
        self.candidate['correction_attempts']=2
        with self.assertRaisesRegex(ValueError,'review_proposals'):self.local.require_active()
        review=deepcopy(self.review);review['frame_id']='f2';review['candidates'][0]['maturity']='turning'
        measured,blocked=reviewed_candidates(review,self.batch,'f2',self.rgb)
        review_proposals(self.local,review,self.batch,measured,blocked,'Orange whole body',self.rgb,[0,0,0])
        self.assertTrue(self.local.can_search)
        self.assertEqual(self.local.candidates[0]['correction_attempts'],2)

    def test_real_red_or_legacy_sample_corrections_remain_available(self):
        p=self.local.survey_batch['proposals'][0]
        for change in ({'red_area_pixels':900},{'depth_verified':False},
                       {'candidate_color_support':'legacy'},{'body_area_pixels':12}):
            old=deepcopy(p);p.update(change)
            self.assertFalse(body_semantics_review_required(self.local,self.candidate))
            p.clear();p.update(old)

    def test_review_budget_and_freshness_still_apply(self):
        measured,blocked=reviewed_candidates(self.review,self.batch,'f1',self.rgb)
        with self.assertRaisesRegex(ValueError,'newer'):
            review_proposals(self.local,self.review,self.batch,measured,blocked,'Same old image',self.rgb,[0,0,0])
        self.local.proposal_review_attempts=2
        self.assertFalse(can_review_proposals(self.local))
        self.assertFalse(body_semantics_review_required(self.local,self.candidate))
        self.assertFalse(self.local.can_search)

    def test_retained_ripe_reports_quantified_unresolved_conflict(self):
        review=deepcopy(self.review);review['frame_id']='f2'
        measured,blocked=reviewed_candidates(review,self.batch,'f2',self.rgb)
        entry=review_proposals(self.local,review,self.batch,measured,blocked,
                               'Still claiming red',self.rgb,[0,0,0])
        conflict=entry['unresolved_claims'][0]
        self.assertTrue(conflict['ripe_color_conflict'])
        self.assertGreater(conflict['body_color_context']['orange_green_fraction'],.9)
        self.assertEqual(self.local.candidates[0]['maturity'],'ripe')
        self.assertFalse(self.local.can_search)
        self.local.proposal_review_attempts=2
        self.assertTrue(body_color_conflict(self.local,self.local.candidates[0]))
        self.assertFalse(can_review_proposals(self.local))

    def test_exhausted_review_does_not_enable_pixel_hunting(self):
        self.local.proposal_review_attempts=2
        self.test_pixel_correction_cannot_hunt_red_flecks_or_consume_correction_budget()

    def test_session_receipt_does_not_present_unresolved_claim_as_review_success(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);self.rgb.save(path/'new.png')
            for name in ('old','new'):
                np.savez(path/(name+'.npz'),position=[0,0,0],rotation=np.eye(3))
            s=Session(SimpleNamespace(meta={'tilted_stalk_prior':True}),SimpleNamespace(out=path),None,visual_proposals=True)
            s.local_targets=self.local;s.frames={'f2':{'harvest_close':path/'new.png'}}
            s.depth_frames={('f1','harvest_close'):path/'old.npz',('f2','harvest_close'):path/'new.npz'}
            s.proposal_frames={'f2':self.batch}
            state={'tcp_world_m':[.19,0,.33],'joints_rad':[0]*6,'base_x_y_yaw':[0,0,0]}
            self.local.survey_state=state
            review=deepcopy(self.review);review.update(frame_id='f2',correction_reason='Retain ripe claim')
            with patch('vista.robot_state',return_value=state):result,_=s.dispatch('review_proposals',review,'f2',state,0)
            self.assertEqual(result['status'],'unresolved')
            self.assertIn('Keeping ripe',result['station_search']['next_action'])
            self.assertEqual(self.local.proposal_review_attempts,1)
            self.assertFalse(self.local.can_search)


if __name__=='__main__':unittest.main()
