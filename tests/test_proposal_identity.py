
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

from copy import deepcopy
import unittest
import numpy as np
from PIL import Image, ImageDraw
from local_targets import LocalTargets
from visual_proposals import reviewed_candidates
from perception_recovery import review_proposals
from proposal_identity import check_duplicate_links


class DuplicateFragmentTests(unittest.TestCase):
    def setUp(self):
        self.rgb=Image.new('RGB',(100,100),'black')
        ImageDraw.Draw(self.rgb).rectangle((45,45,60,60),fill='red')
        self.sample={'position':np.array([.3,0,.5]),'rotation':np.eye(3),
                     'fovy':90.,'depth':np.full((100,100),.3)}
        self.batch={'omitted_by_limit':0,'proposals':[
            {'proposal_id':'P1','pixel':[50,50],'bbox_xyxy':[45,45,60,60],
             'surface_m':[.3,0,.2],'red_area_pixels':200},
            {'proposal_id':'P2','pixel':[52,50],'bbox_xyxy':[62,45,65,50],
             'surface_m':[.312,0,.2],'red_area_pixels':20}]}
        # Original image pixel is independent of wrist projection.
        self.batch['proposals'][1]['pixel']=[63,48]
        self.close=Image.new('RGB',(100,100),'red')
        self.old={'frame_id':'f1','missing_visible_ripe':False,'coverage':'Full view reviewed',
                  'candidates':[{'proposal_id':'P1','maturity':'ripe','visibility':'clear','reason':'red body'},
                                {'proposal_id':'P2','maturity':'uncertain','visibility':'partial','reason':'small fragment'}]}
        self.review={**deepcopy(self.old),'frame_id':'f2'}
        measured,blocked=reviewed_candidates(self.old,self.batch,'f1',self.close)
        self.local=LocalTargets();self.local.register(measured,[0,0,0],self.old['coverage'])
        self.local.survey_batch=deepcopy(self.batch);self.local.survey_review=deepcopy(self.old)
        self.local.candidates[0].update(failures=2,status='unavailable',observation_adjusted=True)
        self.links=[{'fragment_id':'P2','parent_id':'P1','reason':'Both views show fragment on same red body'}]

    def check(self):
        return check_duplicate_links(self.links,self.review['candidates'],self.batch,(self.rgb,self.sample))

    def apply(self):
        measured,blocked=reviewed_candidates(self.review,self.batch,'f2',self.close)
        return review_proposals(self.local,self.review,self.batch,measured,blocked,
                                'Small fragment belongs to same fruit in both images',self.close,[0,0,0],
                                self.links,(self.rgb,self.sample))

    def test_merge_keeps_labels_maturity_and_failure_state(self):
        result=self.apply()
        self.assertEqual(len(result['duplicate_checks']),1)
        self.assertEqual(len(self.local.candidates),1)
        self.assertEqual(self.local.candidates[0]['proposal_id'],'P1')
        self.assertEqual(self.local.candidates[0]['failures'],2)
        self.assertTrue(self.local.candidates[0]['observation_adjusted'])
        self.assertEqual(self.local.candidates[0]['status'],'unavailable')
        self.assertTrue(self.local.can_search)
        self.assertEqual(self.local.survey_review['candidates'][1]['maturity'],'uncertain')
        self.assertEqual(len(self.local.survey_batch['proposals']),2)

    def test_association_cannot_skip_unattempted_parent(self):
        self.local.candidates[0].update(failures=0,status='pending')
        self.apply()
        self.assertEqual(self.local.active['proposal_id'],'P1')
        self.assertFalse(self.local.can_search)

    def test_other_unresolved_fruit_still_blocks_driving(self):
        p={'proposal_id':'P3','pixel':[10,10],'bbox_xyxy':[8,8,12,12],
           'surface_m':[.4,.1,.2],'red_area_pixels':15}
        claim={'proposal_id':'P3','maturity':'uncertain','visibility':'partial','reason':'independent red region'}
        for batch in (self.batch,self.local.survey_batch):batch['proposals'].append(deepcopy(p))
        for review in (self.review,self.local.survey_review):review['candidates'].append(deepcopy(claim))
        self.apply()
        self.assertFalse(self.local.can_search)
        self.assertEqual(self.local.candidates[1]['status'],'unverified_semantics')

    def test_two_full_bodies_cannot_merge_even_if_close(self):
        self.batch['proposals'][1]['red_area_pixels']=150
        with self.assertRaisesRegex(ValueError,'small fragment'):self.check()

    def test_independent_wrist_regions_cannot_merge(self):
        ImageDraw.Draw(self.rgb).line((51,40,51,65),fill='black',width=1)
        with self.assertRaisesRegex(ValueError,'same wrist'):self.check()

    def test_wrong_depth_missing_depth_and_distance_rejected(self):
        for change in ('occluded','missing','far'):
            self.setUp()
            if change=='occluded':self.sample['depth'][:]=.28
            elif change=='missing':self.sample['depth'][:]=np.nan
            else:self.batch['proposals'][1]['surface_m'][0]=.33
            with self.subTest(change=change),self.assertRaises(ValueError):self.check()

    def test_not_fruit_remains_rejected_without_mutating_queue(self):
        before=deepcopy(self.local.candidates)
        self.review['candidates'][1]['maturity']='not_fruit'
        with self.assertRaisesRegex(ValueError,'duplicate_links'):self.apply()
        self.assertEqual(before,self.local.candidates)
        self.assertFalse(self.local.can_search)

    def test_self_duplicate_and_unripe_parent_rejected(self):
        self.links[0]['parent_id']='P2'
        with self.assertRaises(ValueError):self.check()
        self.links[0]['parent_id']='P1';self.review['candidates'][0]['maturity']='green'
        with self.assertRaises(ValueError):self.check()


if __name__=='__main__':unittest.main()
