
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
import numpy as np
from PIL import Image
from vista import Session,MotionSafetyStop
from vista import check_observation
from centered_grasp import TrackingError
from perception_recovery import observation_adjustment_available,uncertain_review_candidate,missing_stalk_views


class UncertainObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)
        self.state={'tcp_world_m':[.19,-.15,.33],'joints_rad':[0,0,0,0,-.7,1.], 'base_x_y_yaw':[0,0,0]}
        env=SimpleNamespace(meta={'tilted_stalk_prior':True},frame=10,last_action=np.array(self.state['joints_rad']),drive=Mock())
        self.s=Session(env,SimpleNamespace(out=self.path),None,visual_proposals=True)
        self.s.centered=Mock();self.s.centered.adjust_observation.return_value={'ticks':30,'tcp_error_m':.001}
        self.s.log=Mock();local=self.s.local_targets
        local.register([{'surface_m':[.25,0,.22],'maturity':'uncertain','visibility':'partial',
                         'semantic_unverified':True,'body_pixel_verified':True,'proposal_id':'P1'}],[0,0,0],'entire frame')
        self.item=local.candidates[0];self.item['failures']=1
        local.proposal_review_attempts=1;local.survey_review={'frame_id':'f9'};local.survey_state=self.state
        np.savez(self.path/'depth.npz',position=[0,0,.65],rotation=np.eye(3),fovy=60.,depth=np.full((360,480),.43))
        self.s.depth_frames={('f9','harvest_close'):self.path/'depth.npz'}
        self.args={'basis_frame_id':'f10','candidate_key':self.item['candidate_key'],'reason':'Same fruit occluded by robot'}

    def dispatch(self,args=None):
        with patch('vista.robot_state',return_value=self.state):
            return self.s.dispatch('adjust_observation',args or self.args,'f10',self.state,0)[0]

    def test_adjust_keeps_uncertainty_retries_and_calibration(self):
        self.assertTrue(observation_adjustment_available(self.s))
        self.assertIn('SAME uncertain',self.s.station_status()['next_action'])
        result=self.dispatch()
        self.assertTrue(result['recovery']['semantic_review_only'])
        self.assertEqual(self.item['failures'],1)
        self.assertEqual(self.s.local_targets.proposal_review_attempts,1)
        self.assertIsNone(self.s.local_targets.active);self.assertFalse(self.s.local_targets.can_search)
        self.assertFalse(self.s.frames or self.s.depth_frames or self.s.crops)
        self.assertEqual(self.s.local_targets.survey_reference_depth,self.path/'depth.npz')
        self.assertFalse(observation_adjustment_available(self.s))
        self.assertIn('review_proposals',result['next_action'])
        with self.assertRaises(ValueError):self.s.local_targets.require_active()
        with self.assertRaises(ValueError):self.s.require_local_search()

    def test_requires_one_review_and_remaining_review_budget(self):
        for attempts in (0,3):
            self.s.local_targets.proposal_review_attempts=attempts
            self.assertFalse(observation_adjustment_available(self.s))
        self.s.local_targets.proposal_review_attempts=1
        self.s.local_targets.observation_adjustments=2
        self.assertFalse(observation_adjustment_available(self.s))

    def test_second_review_can_recover_but_must_reserve_perception_for_review(self):
        self.s.local_targets.proposal_review_attempts=2
        self.assertTrue(observation_adjustment_available(self.s))
        self.s.station_attempts=11
        self.assertFalse(observation_adjustment_available(self.s))
        self.s.station_attempts=10
        self.dispatch()
        self.assertEqual(self.s.station_attempts,11)
        self.s.centered.adjust_observation.assert_called_once()

    def test_premature_finish_reminds_once_without_motion_or_budget_reset(self):
        self.s.calls=9;self.s.station_attempts=3
        result,_=self.s.dispatch('finish',{'reason':'Review failed'},'f10',self.state,0)
        self.assertFalse(result['episode_ended']);self.assertFalse(self.s.finished)
        self.assertIn('adjust_observation',result['next_action'])
        self.assertEqual(result['remaining_model_calls'],23)
        self.s.centered.adjust_observation.assert_not_called()
        result,_=self.s.dispatch('finish',{'reason':'Recovery unsuitable'},'f10',self.state,0)
        self.assertTrue(result['episode_ended']);self.assertTrue(self.s.finished)
        self.assertEqual(self.s.station_attempts,3)

    def test_finish_is_immediate_when_unsafe_late_or_out_of_budget(self):
        for attr,value in (('motion_stop_trace',{'reason':'tracking'}),('grasp_attempted',True),
                           ('calls',23),('station_attempts',12),('target_stage','approach')):
            with self.subTest(attr=attr):
                old=getattr(self.s,attr,None);setattr(self.s,attr,value)
                result,_=self.s.dispatch('finish',{'reason':'Stop'},'f10',self.state,0)
                self.assertTrue(result['episode_ended'])
                setattr(self.s,attr,old);self.s.finished=False

    def test_no_unknown_depth_or_coverage_or_grasp_bypass(self):
        self.s.local_targets.selection_blocked_reason='unresolved coverage'
        self.assertIsNone(uncertain_review_candidate(self.s.local_targets))
        self.s.local_targets.selection_blocked_reason=None
        self.item['body_pixel_verified']=False
        self.assertFalse(observation_adjustment_available(self.s))
        self.item['body_pixel_verified']=True;self.s.grasp_attempted=True
        self.assertFalse(observation_adjustment_available(self.s))

    def measured_fragment(self):
        self.item.update(body_pixel_verified=False,maturity='uncertain',source_frame='f9')
        region={'proposal_id':'P1','depth_verified':True,'candidate_color_support':'red_or_orange',
                'red_area_pixels':12,'surface_m':self.item['surface_m'][:],'distance_xy_m':.25}
        self.s.local_targets.survey_batch={'proposals':[region]}
        return region

    def test_measured_fragment_can_only_observe_and_review(self):
        self.measured_fragment();self.s.local_targets.proposal_review_attempts=2
        self.assertTrue(observation_adjustment_available(self.s))
        self.s.calls=4
        end,_=self.s.dispatch('finish',{'reason':'Uncertain fragment'},'f10',self.state,0)
        self.assertFalse(end['episode_ended'])
        result=self.dispatch()
        self.assertFalse(result['recovery']['uncertain_region_review']['body_pixel_verified'])
        self.assertTrue(result['recovery']['semantic_review_only'])
        self.assertFalse(self.item['body_pixel_verified'])
        self.assertEqual(self.item['maturity'],'uncertain')
        self.assertEqual(self.item['failures'],1)
        self.assertIsNone(self.s.local_targets.active)
        self.assertFalse(self.s.target_confirmed or self.s.local_targets.can_search)
        self.assertFalse(observation_adjustment_available(self.s))
        with self.assertRaises(ValueError):self.s.require_local_search()

    def test_fragment_requires_reviewed_matching_finite_depth_region(self):
        region=self.measured_fragment()
        for key,value in [('depth_verified',False),('candidate_color_support','none'),
                          ('red_area_pixels',8),('proposal_id','P2'),('surface_m',None),
                          ('surface_m',[float('nan'),0,.22]),('surface_m',[.5,0,.22]),
                          ('distance_xy_m',.57),('distance_xy_m',float('nan'))]:
            with self.subTest(key=key,value=value):
                old=region[key];region[key]=value
                self.assertFalse(observation_adjustment_available(self.s))
                region[key]=old
        self.item['source_frame']='stale'
        self.assertFalse(observation_adjustment_available(self.s))
        self.item['source_frame']='f9';self.s.local_targets.proposal_review_attempts=0
        self.assertFalse(observation_adjustment_available(self.s))

    def test_current_candidate_frame_and_original_calibration_required(self):
        for change in ({'candidate_key':'other'},{'basis_frame_id':'old'}):
            with self.assertRaises(ValueError):self.dispatch({**self.args,**change})
        self.s.depth_frames.clear()
        with self.assertRaisesRegex(ValueError,'calibration'):self.dispatch()
        self.s.centered.adjust_observation.assert_not_called()
        self.assertNotIn('observation_adjusted',self.item)

    def test_partial_motion_stops_and_retains_uncertainty(self):
        self.s.centered.adjust_observation.side_effect=TrackingError('tracking')
        with self.assertRaises(MotionSafetyStop):self.dispatch()
        self.assertIsNone(self.s.local_targets.active)
        self.assertFalse(observation_adjustment_available(self.s))

    def test_ripe_adjustment_refreshes_later_review_baseline_and_keeps_camera(self):
        # Seed 3: adjust for a ripe fruit, exhaust it, then review a different
        # uncertain fragment. The old arm posture must not veto every new view.
        local=self.s.local_targets
        self.item.update(maturity='ripe',semantic_unverified=False,status='pending',cross_view_failed=True)
        before=self.state
        after={**before,'tcp_world_m':[.15,-.15,.37],
               'joints_rad':[0,.386,0,0,-.7,1.]}
        with patch('vista.robot_state',side_effect=[before,after]):
            result,_=self.s.dispatch('adjust_observation',self.args,'f10',before,0)
        self.assertFalse(result['recovery']['semantic_review_only'])
        self.assertEqual(local.survey_reference_depth,self.path/'depth.npz')
        self.assertEqual(local.survey_state,after)
        self.assertFalse(self.s.depth_frames)
        self.assertEqual(self.item['failures'],1)
        check_observation(local.survey_state,after,4.,30.)
        with self.assertRaisesRegex(ValueError,'stale'):
            check_observation(before,after,4.,30.)
        # Genuine subsequent drift still fails; refreshing is not bypassing it.
        with self.assertRaisesRegex(ValueError,'stale'):
            check_observation(local.survey_state,before,4.,30.)

    def test_ripe_adjustment_requires_calibration_and_failed_motion_keeps_baseline(self):
        self.item.update(maturity='ripe',semantic_unverified=False,status='pending',cross_view_failed=True)
        self.s.depth_frames.clear()
        with self.assertRaisesRegex(ValueError,'calibration'):self.dispatch()
        self.s.centered.adjust_observation.assert_not_called()
        self.s.local_targets.survey_reference_depth=self.path/'depth.npz'
        self.s.centered.adjust_observation.side_effect=TrackingError('tracking')
        with self.assertRaises(MotionSafetyStop):self.dispatch()
        self.assertEqual(self.s.local_targets.survey_state,self.state)

    def test_no_fit_views_allow_look_without_charging_failure_or_authorizing_grasp(self):
        self.item.update(maturity='ripe',semantic_unverified=False,status='pending',failures=0)
        self.s.stalk_view_evidence={'frame_id':'f10','candidate_key':self.item['candidate_key'],
                                    'views':{'harvest_close':'no_fit','wrist':'outside_crop'}}
        self.assertTrue(observation_adjustment_available(self.s,'f10'))
        self.assertIn('adjust_observation',self.s.station_status()['next_action'])
        self.dispatch()
        self.assertIs(self.s.local_targets.active,self.item)
        self.assertEqual(self.item['failures'],0)
        self.assertFalse(self.s.target_confirmed or self.s.local_targets.can_search)
        self.assertIsNone(self.s.stalk_view_evidence)
        self.assertFalse(observation_adjustment_available(self.s,'f10'))

    def test_no_fit_recovery_requires_two_current_measured_views_of_verified_body(self):
        self.item.update(maturity='ripe',semantic_unverified=False,status='pending',failures=0)
        evidence={'frame_id':'f10','candidate_key':self.item['candidate_key'],
                  'views':{'harvest_close':'no_fit','wrist':'no_fit'}}
        for change in ({'frame_id':'old'},{'candidate_key':'other'},
                       {'views':{'harvest_close':'no_fit'}},
                       {'views':{'harvest_close':'no_fit','wrist':'fit_available'}},
                       {'views':{'harvest_close':'outside_crop','wrist':'outside_crop'}}):
            self.s.stalk_view_evidence={**evidence,**change}
            self.assertFalse(missing_stalk_views(self.s,'f10'))
            with self.assertRaises(ValueError):self.dispatch()
        self.s.stalk_view_evidence=evidence;self.item['body_pixel_verified']=False
        self.assertFalse(observation_adjustment_available(self.s,'f10'))
        self.s.centered.adjust_observation.assert_not_called()

    def test_crop_is_review_only_without_stalk_or_selection_handles(self):
        Image.new('RGB',(480,360),'red').save(self.path/'rgb.png')
        self.s.frames={'f10':{'harvest_close':self.path/'rgb.png'}}
        self.s.depth_frames[('f10','harvest_close')]=self.path/'depth.npz'
        info,image=self.s.active_candidate_crop('f10',self.state,review_only=True)
        self.assertEqual(info['purpose'],'semantic_review_only');self.assertEqual(info['proposal_id'],'P1')
        self.assertNotIn('crop_id',info);self.assertNotIn('stalk_proposals',info)
        self.assertFalse(self.s.crops or self.s.stalk_proposals)
        self.assertIsNotNone(image)

if __name__=='__main__':unittest.main()
