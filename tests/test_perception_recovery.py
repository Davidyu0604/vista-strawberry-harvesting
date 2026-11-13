
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
from project_paths import PROJECT_ROOT, project_source

from copy import deepcopy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image

from local_targets import LocalTargets
from perception_recovery import MISSING_BLOCK, can_correct_survey, correct_survey, observation_adjustment_available, current_body_proposals, can_review_proposals
from recording_quality import phase_labels
from vista import Session, MotionSafetyStop
from centered_grasp import TrackingError


class CoverageCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.local=LocalTargets()
        self.local.register([{'surface_m':[.25,0,.22],'maturity':'ripe','visibility':'clear'}],[0,0,0],'original')
        self.local.candidates[0]['failures']=1
        self.local.selection_blocked_reason=MISSING_BLOCK
        self.local.survey_review={'frame_id':'f1','coverage':'original','missing_visible_ripe':True,
            'candidates':[{'proposal_id':'P1','maturity':'ripe','visibility':'clear','reason':'red fruit'}]}
        self.batch={'proposals':[{'proposal_id':'P1','surface_m':[.25,0,.22]}]}
        self.local.survey_batch=deepcopy(self.batch)
        self.review={**deepcopy(self.local.survey_review),'frame_id':'f2','missing_visible_ripe':False,'coverage':'whole current image checked'}

    def test_correction_preserves_queue_and_records_before_after(self):
        before=deepcopy(self.local.candidates)
        entry=correct_survey(self.local,self.review,self.batch,None,'Original claim was background; reviewed entire current frame')
        self.assertEqual(before,self.local.candidates)
        self.assertEqual(self.local.active['failures'],1)
        self.assertTrue(entry['previous_review']['missing_visible_ripe'])
        self.assertFalse(entry['review']['missing_visible_ripe'])
        self.assertFalse(can_correct_survey(self.local))

    def test_coverage_only_refresh_keeps_original_region_when_detector_changes(self):
        self.assertTrue(can_correct_survey(self.local))
        self.assertFalse(can_review_proposals(self.local))
        p=self.local.survey_batch['proposals'][0]
        p.update(pixel=[10,10],bbox_xyxy=[8,8,12,12],depth_verified=True)
        # New red segmentation could return no labels or a different P1.
        # Correction must instead remeasure the original pixel, even if red
        # evidence changed; the model still reviews the original region.
        rgb=Image.new('RGB',(100,80),'black');depth=np.full((80,100),.4)
        with patch('visual_proposals.propose_bodies') as detector:
            batch=current_body_proposals(self.local,rgb,depth,{},[0,0,0],lambda *args:np.array([.25,0,.22]))
        detector.assert_not_called()
        self.assertEqual(batch['proposals'][0]['pixel'],[10,10])
        self.assertEqual(batch['proposals'][0]['proposal_id'],'P1')
        self.assertEqual(batch['proposals'][0]['surface_m'],[.25,0,.22])

    def test_coverage_refresh_retains_depth_hole_and_blocks_correction(self):
        p=self.local.survey_batch['proposals'][0]
        p.update(pixel=[10,10],bbox_xyxy=[8,8,12,12])
        batch=current_body_proposals(self.local,Image.new('RGB',(100,80),'red'),
                                     np.full((80,100),np.nan),{},[0,0,0],Mock())
        self.assertEqual(len(batch['proposals']),1)
        self.assertIsNone(batch['proposals'][0]['surface_m'])
        with self.assertRaisesRegex(ValueError,'correspondence'):
            correct_survey(self.local,self.review,batch,None,'New full image reviewed')
        self.assertIsNone(self.local.active)

    def test_unresolved_omission_stays_blocked_and_two_attempts_limit(self):
        self.review['missing_visible_ripe']=True
        for _ in range(2):correct_survey(self.local,self.review,self.batch,MISSING_BLOCK,'Still unresolved')
        self.assertIsNone(self.local.active);self.assertFalse(self.local.can_search)
        with self.assertRaises(ValueError):correct_survey(self.local,self.review,self.batch,None,'third')

    def test_truncation_or_other_block_cannot_be_cleared(self):
        self.local.selection_blocked_reason=MISSING_BLOCK+'; Proposal limit omitted regions'
        self.assertFalse(can_correct_survey(self.local))
        with self.assertRaises(ValueError):correct_survey(self.local,self.review,self.batch,None,'ignore truncation')

    def test_no_relabelling_or_removing_candidates(self):
        for change in ('remove','maturity','visibility'):
            with self.subTest(change=change):
                review=deepcopy(self.review)
                if change=='remove':review['candidates']=[]
                elif change=='maturity':review['candidates'][0]['maturity']='green'
                else:review['candidates'][0]['visibility']='occluded'
                self.local.survey_correction_attempts=0
                with self.assertRaises(ValueError):correct_survey(self.local,review,self.batch,None,'skip nearer')
                self.assertIsNone(self.local.active)
                self.assertEqual(self.local.candidates[0]['failures'],1)

    def test_depth_correspondence_cannot_jump_to_another_fruit(self):
        changed=deepcopy(self.batch);changed['proposals'][0]['surface_m'][1]=.05
        with self.assertRaises(ValueError):correct_survey(self.local,self.review,changed,None,'different fruit')
        self.assertIsNone(self.local.active)

    def test_unchanged_nonripe_region_without_depth_is_not_a_ripe_body_failure(self):
        negative={'proposal_id':'P2','maturity':'turning','visibility':'clear','reason':'orange body'}
        region={'proposal_id':'P2','surface_m':None,'pixel':[40,40],'bbox_xyxy':[39,39,42,42]}
        self.local.survey_review['candidates'].append(deepcopy(negative))
        self.review['candidates'].append(negative)
        self.local.survey_batch['proposals'].append(deepcopy(region));self.batch['proposals'].append(region)
        correct_survey(self.local,self.review,self.batch,None,'Reviewed entire frame, orange region unchanged')
        self.assertIsNotNone(self.local.active)

    def test_unknown_depth_cannot_be_used_to_clear_uncertain_region(self):
        self.local.survey_batch['proposals'][0]['surface_m']=None
        self.batch['proposals'][0]['surface_m']=None
        with self.assertRaises(ValueError):correct_survey(self.local,self.review,self.batch,None,'unknown')
        self.assertIsNone(self.local.active)

    def test_stale_review_and_blank_reason_rejected(self):
        with self.assertRaises(ValueError):correct_survey(self.local,self.local.survey_review,self.batch,None,'old')
        with self.assertRaises(ValueError):correct_survey(self.local,self.review,self.batch,None,' ')

    def test_dispatch_correction_keeps_budget_and_requires_current_full_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);Image.new('RGB',(480,360),(240,20,20)).save(path/'frame.png')
            env=SimpleNamespace(meta={'tilted_stalk_prior':True})
            s=Session(env,SimpleNamespace(out=path),None,visual_proposals=True)
            s.local_targets=self.local
            state={'tcp_world_m':[.18,0,.33],'joints_rad':[0]*6,'base_x_y_yaw':[0,0,0]}
            self.local.survey_state=state
            batch=deepcopy(self.batch);batch['omitted_by_limit']=0;batch['proposals'][0]['pixel']=[10,10]
            s.proposal_frames={'f2':batch};s.frames={'f2':{'harvest_close':path/'frame.png'}}
            s.station_attempts=4
            with patch('vista.robot_state',return_value=state):
                with self.assertRaises(ValueError):
                    s.dispatch('correct_survey',{**self.review,'frame_id':'old','correction_reason':'wrong'},'f2',state,0)
                result,_=s.dispatch('correct_survey',{**self.review,'correction_reason':'Reviewed original claim and entire fresh view'},'f2',state,0)
            self.assertEqual(result['status'],'corrected')
            self.assertEqual(s.station_attempts,6)
            self.assertEqual(s.local_targets.active['failures'],1)
            self.assertEqual(len(s.perception_recoveries),1)


class ObservationRecoveryTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.state={'tcp_world_m':[.18,-.15,.33],'joints_rad':[0,0,0,0,-.7,1.], 'base_x_y_yaw':[0,0,0]}
        e=SimpleNamespace(meta={'tilted_stalk_prior':True},frame=10,last_action=np.array(self.state['joints_rad']),drive=Mock())
        self.s=Session(e,SimpleNamespace(out=Path(tmp.name)),None,visual_proposals=True)
        self.s.centered=Mock()
        self.s.centered.adjust_observation.return_value={'tcp_error_m':.001,'ticks':40}
        self.s.local_targets.register([{'surface_m':[.25,0,.22],'maturity':'ripe','visibility':'clear'}],[0,0,0],'all seen')
        self.active=self.s.local_targets.active
        self.active.update(failures=1,cross_view_failed=True)
        self.s.locked_target=np.array([.25,0,.25])
        self.args={'basis_frame_id':'f10','candidate_key':self.active['candidate_key'],'reason':'Robot blocks close view'}

    def dispatch(self,args=None):
        with patch('vista.robot_state',return_value=self.state):
            return self.s.dispatch('adjust_observation',args or self.args,'f10',self.state,0)[0]

    def test_reobserve_preserves_active_and_failure_count_invalidates_old_targets(self):
        self.s.frames={'old':{}};self.s.depth_frames={'old':{}};self.s.crops={'old':{}};self.s.stalk_proposals={'W1':{}}
        result=self.dispatch()
        self.assertEqual(result['status'],'executed')
        self.assertIs(self.s.local_targets.active,self.active)
        self.assertEqual(self.active['failures'],1)
        self.assertEqual(self.s.station_attempts,1)
        self.assertFalse(self.s.frames or self.s.depth_frames or self.s.crops or self.s.stalk_proposals)
        self.assertIsNone(self.s.locked_target);self.assertFalse(self.s.target_confirmed)
        self.assertFalse(observation_adjustment_available(self.s))
        with self.assertRaises(ValueError):self.dispatch()
        self.s.centered.adjust_observation.assert_called_once()

    def test_no_adjustment_without_failed_cross_view_or_after_approach(self):
        self.active['cross_view_failed']=False
        with self.assertRaises(ValueError):self.dispatch()
        self.active['cross_view_failed']=True;self.s.target_stage='pregrasp'
        with self.assertRaises(ValueError):self.dispatch()
        self.s.centered.adjust_observation.assert_not_called()

    def test_wrong_candidate_or_stale_frame_cannot_move(self):
        for change in ({'candidate_key':'other'},{'basis_frame_id':'old'}):
            with self.assertRaises(ValueError):self.dispatch({**self.args,**change})
        self.s.centered.adjust_observation.assert_not_called()

    def test_station_and_motion_limits_not_reset(self):
        self.s.station_attempts=12
        self.assertEqual(self.dispatch()['status'],'rejected')
        self.s.station_attempts=0;self.s.local_targets.observation_adjustments=2
        with self.assertRaises(ValueError):self.dispatch()
        self.s.centered.adjust_observation.assert_not_called()

    def test_preflight_failure_does_not_move_or_clear_target(self):
        self.s.centered.observation_plan.side_effect=ValueError('self-collision')
        with self.assertRaises(ValueError):self.dispatch()
        self.s.centered.adjust_observation.assert_not_called()
        self.assertIsNotNone(self.s.locked_target)
        self.assertNotIn('observation_adjusted',self.active)

    def test_partial_motion_failure_stops_episode(self):
        exc=TrackingError('tracking');exc.tracking={'error_m':.009}
        self.s.centered.adjust_observation.side_effect=exc
        with self.assertRaises(MotionSafetyStop):self.dispatch()
        self.assertEqual(self.s.motion_stop_trace['phase'],'observation_adjust')
        self.assertEqual(self.s.motion_stop_trace['tracking']['error_m'],.009)
        self.assertIsNone(self.s.locked_target)

    def test_next_action_keeps_same_candidate_and_recovery_labels(self):
        self.assertIn('SAME active',self.s.station_status()['next_action'])
        self.active['observation_adjusted']=True
        self.assertIn(self.active['candidate_key'],self.s.station_status()['next_action'])
        self.assertEqual(phase_labels('observation_adjust_hold'),('recovery',False,True))


if __name__=='__main__':unittest.main()
