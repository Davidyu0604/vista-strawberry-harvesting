
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

from copy import deepcopy
import unittest
from PIL import Image
from local_targets import LocalTargets
from perception_recovery import review_proposals, can_review_proposals
from visual_proposals import reviewed_candidates
from visual_proposals import refresh_proposals
import numpy as np


class ProposalReviewTests(unittest.TestCase):
    def setUp(self):
        self.rgb=Image.new('RGB',(100,80),(224,87,46))
        self.batch={'omitted_by_limit':0,'proposals':[
            {'proposal_id':'P1','pixel':[30,30],'bbox_xyxy':[25,25,35,35],'surface_m':None},
            {'proposal_id':'P2','pixel':[60,30],'bbox_xyxy':[55,25,65,35],'surface_m':[.3,.1,.22]}]}
        self.old={'frame_id':'f1','coverage':'entire view','missing_visible_ripe':False,
                  'candidates':[{'proposal_id':p['proposal_id'],'maturity':'ripe','visibility':'clear','reason':'red'} for p in self.batch['proposals']]}
        measured,blocked=reviewed_candidates(self.old,self.batch,'f1',Image.new('RGB',(100,80),'red'))
        self.local=LocalTargets();self.local.register(measured,[0,0,0],'entire view')
        self.local.candidates[0].update(failures=1,status='unavailable')
        self.local.selection_blocked_reason=blocked
        self.local.survey_review=deepcopy(self.old);self.local.survey_batch=deepcopy(self.batch)
        self.review={**deepcopy(self.old),'frame_id':'f2'}
        for item in self.review['candidates']:item['maturity']='turning';item['reason']='Whole body orange'

    def run_review(self):
        measured,blocked=reviewed_candidates(self.review,self.batch,self.review['frame_id'],self.rgb)
        return review_proposals(self.local,self.review,self.batch,measured,blocked,'Old red fragment was orange body',self.rgb,[0,0,0])

    def test_color_correction_unblocks_without_resetting_failed_candidate(self):
        entry=self.run_review()
        self.assertTrue(self.local.can_search)
        self.assertEqual(self.local.candidates[0]['failures'],1)
        self.assertEqual(self.local.candidates[0]['status'],'unavailable')
        self.assertEqual(self.local.proposal_review_attempts,1)
        self.assertIn('P1',entry['color_checks'])

    def test_red_body_cannot_be_relabelled_unripe(self):
        self.rgb=Image.new('RGB',(100,80),'red')
        entry=self.run_review()
        self.assertEqual(len(entry['deferred_corrections']),2)
        self.assertTrue(all(c['maturity']=='uncertain' for c in self.local.survey_review['candidates']))
        self.assertIn('unresolved body depth',self.local.selection_blocked_reason)
        self.assertEqual(self.review['candidates'][0]['maturity'],'turning')
        self.assertFalse(self.local.can_search)

    def test_partial_review_retains_supported_correction_and_uncertain_blocker(self):
        self.batch['proposals'][0]['surface_m']=[.25,0,.22]
        self.rgb.paste('red',(43,13,78,48))
        # P1's orange context and P2's red context are separate.
        self.local.candidates[0]['status']='unverified_measurement'
        entry=self.run_review()
        self.assertEqual([c['maturity'] for c in self.local.survey_review['candidates']],['turning','uncertain'])
        self.assertEqual([c['proposal_id'] for c in entry['deferred_corrections']],['P2'])
        self.assertFalse(self.local.can_search)
        self.assertIsNone(self.local.active)

    def test_third_review_requires_actual_new_adjustment_and_is_not_repeatable(self):
        self.uncertain_fixture()
        self.local.proposal_review_attempts=2
        for completed in (None,0,1):
            self.local.candidates[0]['observation_recovery_completed_frame']=completed
            self.assertFalse(can_review_proposals(self.local))
        self.local.candidates[0]['observation_recovery_completed_frame']=2
        self.assertTrue(can_review_proposals(self.local))
        self.review['frame_id']='f3'
        self.run_review()
        self.assertEqual(self.local.proposal_review_attempts,3)
        self.local.candidates[0]['observation_recovery_completed_frame']=4
        self.assertFalse(can_review_proposals(self.local))

    def test_depth_remeasurement_can_resolve_same_region(self):
        self.rgb=Image.new('RGB',(100,80),'red')
        self.review=deepcopy(self.old);self.review['frame_id']='f2'
        self.batch['proposals'][0]['surface_m']=[.25,0,.22]
        self.run_review()
        self.assertIsNotNone(self.local.active)
        self.assertEqual(self.local.active['proposal_id'],'P1')
        failed=next(c for c in self.local.candidates if c['proposal_id']=='P2')
        self.assertEqual(failed['failures'],1);self.assertEqual(failed['status'],'unavailable')

    def test_unresolved_depth_retained_and_review_limit(self):
        self.review=deepcopy(self.old);self.review['frame_id']='f2'
        self.run_review();self.assertFalse(self.local.can_search)
        self.review['frame_id']='f3'
        measured,blocked=reviewed_candidates(self.review,self.batch,'f3',self.rgb)
        review_proposals(self.local,self.review,self.batch,measured,blocked,'Still unclear',self.rgb,[0,0,0])
        self.assertFalse(can_review_proposals(self.local));self.assertFalse(self.local.can_search)

    def test_changed_region_coverage_or_known_depth_rejected(self):
        for change in ('region','coverage','depth'):
            self.setUp()
            if change=='region':self.batch['proposals'][0]['bbox_xyxy']=[40,25,50,35]
            elif change=='coverage':self.review['missing_visible_ripe']=True
            else:self.batch['proposals'][1]['surface_m']=[.4,.1,.22]
            with self.subTest(change=change),self.assertRaises(ValueError):self.run_review()
            self.assertFalse(self.local.can_search)

    def test_refresh_keeps_numbering_and_depth_holes(self):
        batch=deepcopy(self.batch)
        rgb=Image.new('RGB',(100,80),'red');depth=np.full((80,100),.4)
        depth[25:36,25:36]=np.nan
        def unproject(depth,cal,x,y):return np.array([.25,0,.22])
        refreshed=refresh_proposals(batch,rgb,depth,{},[0,0,0],unproject)
        self.assertEqual([p['proposal_id'] for p in refreshed['proposals']],['P1','P2'])
        self.assertIsNone(refreshed['proposals'][0]['surface_m'])
        self.assertEqual(refreshed['proposals'][0]['bbox_xyxy'],batch['proposals'][0]['bbox_xyxy'])
        depth[:]=.4
        refreshed=refresh_proposals(batch,rgb,depth,{},[0,0,0],unproject)
        self.assertIsNotNone(refreshed['proposals'][0]['surface_m'])

    def uncertain_fixture(self):
        self.rgb=Image.new('RGB',(100,80),'red')
        self.batch['proposals'][0]['surface_m']=[.25,0,.22]
        self.old['candidates'][0].update(maturity='uncertain',visibility='partial')
        measured,blocked=reviewed_candidates(self.old,self.batch,'f1',self.rgb)
        self.local=LocalTargets();self.local.register(measured,[0,0,0],self.old['coverage'])
        self.local.selection_blocked_reason=blocked
        self.local.survey_review=deepcopy(self.old);self.local.survey_batch=deepcopy(self.batch)
        self.local.candidates[0].update(failures=1,observation_adjusted=True)
        self.review={**deepcopy(self.old),'frame_id':'f2'}

    def test_actual_visibility_sequence_occluded_then_clear_preserves_candidate(self):
        self.uncertain_fixture();key=self.local.candidates[0]['candidate_key']
        self.review['candidates'][0]['visibility']='occluded'
        first=self.run_review()
        self.assertIsNone(self.local.active);self.assertFalse(self.local.can_search)
        self.assertEqual(first['visibility_changes']['P1'],{'before':'partial','after':'occluded'})
        self.review['frame_id']='f3'
        self.review['candidates'][0].update(maturity='ripe',visibility='clear')
        measured,blocked=reviewed_candidates(self.review,self.batch,'f3',self.rgb)
        second=review_proposals(self.local,self.review,self.batch,measured,blocked,'Fresh view exposes same red fruit',self.rgb,[0,0,0])
        self.assertEqual(self.local.active['candidate_key'],key)
        self.assertEqual(self.local.active['failures'],1)
        self.assertTrue(self.local.active['observation_adjusted'])
        self.assertEqual(self.local.active['status'],'pending')
        self.assertEqual(self.local.proposal_review_attempts,2)
        self.assertEqual(second['visibility_changes']['P1']['after'],'clear')

    def test_only_completed_semantic_recovery_records_confirmation_provenance(self):
        for completed in (None,100,200):
            self.uncertain_fixture()
            self.local.candidates[0].update(failures=0,observation_recovery_completed_frame=completed)
            self.review['frame_id']='f150'
            self.review['candidates'][0].update(maturity='ripe',visibility='clear')
            measured,blocked=reviewed_candidates(self.review,self.batch,'f150',self.rgb)
            review_proposals(self.local,self.review,self.batch,measured,blocked,'Fresh mature view',self.rgb,[0,0,0])
            candidate=self.local.active
            self.assertEqual(candidate.get('semantic_recovery_review_frame'),150 if completed==100 else None)

    def test_ripe_occluded_stays_unresolved_and_blocks_farther_target(self):
        self.uncertain_fixture()
        self.review['candidates'][0].update(maturity='ripe',visibility='occluded')
        self.run_review()
        self.assertEqual(self.local.candidates[0]['status'],'unverified_semantics')
        self.assertIsNone(self.local.active);self.assertFalse(self.local.can_search)

    def test_clear_visibility_alone_does_not_certify_maturity_or_revive_failure(self):
        self.uncertain_fixture()
        self.review['candidates'][0]['visibility']='clear'
        self.run_review()
        self.assertIsNone(self.local.active);self.assertFalse(self.local.can_search)
        self.uncertain_fixture()
        self.local.candidates[0]['status']='unavailable'
        self.local.candidates[1]['status']='unverified_semantics'
        self.review['candidates'][0].update(maturity='ripe',visibility='clear')
        self.run_review()
        self.assertEqual(self.local.candidates[0]['status'],'unavailable')

    def test_session_exposes_review_counts_budget_and_requires_new_pose_consistent_frame(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from vista import Session
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);self.rgb.save(path/'new.png')
            np.savez(path/'old.npz',position=[0,0,0],rotation=np.eye(3))
            np.savez(path/'new.npz',position=[0,0,0],rotation=np.eye(3))
            s=Session(SimpleNamespace(meta={'tilted_stalk_prior':True}),SimpleNamespace(out=path),None,visual_proposals=True)
            s.local_targets=self.local;s.frames={'f2':{'harvest_close':path/'new.png'}}
            s.depth_frames={('f1','harvest_close'):path/'old.npz',('f2','harvest_close'):path/'new.npz'}
            s.proposal_frames={'f2':self.batch};s.station_attempts=4
            state={'tcp_world_m':[.19,0,.33],'joints_rad':[0]*6,'base_x_y_yaw':[0,0,0]}
            self.local.survey_state=state
            self.assertIn('review_proposals',s.station_status()['next_action'])
            with patch('vista.robot_state',return_value=state):
                result,_=s.dispatch('review_proposals',{**self.review,'correction_reason':'Orange full bodies'},'f2',state,0)
            self.assertEqual(result['status'],'reviewed');self.assertEqual(s.station_attempts,5)
            self.assertTrue(self.local.can_search)
            self.assertEqual(s.perception_recoveries[0]['kind'],'proposal_review')


if __name__=='__main__':unittest.main()
