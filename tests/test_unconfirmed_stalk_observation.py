"""Geometric proposals are not semantic approval; allow bounded reobservation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'src'))
import tempfile,unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from vista import Session,MotionSafetyStop
from perception_recovery import unconfirmed_stalk_options,observation_adjustment_available


class UnconfirmedStalkObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);path=Path(self.tmp.name)
        self.state={'tcp_world_m':[.19,-.04,.34],'joints_rad':[0,0,0,0,-.9,1.], 'base_x_y_yaw':[0,0,0]}
        env=SimpleNamespace(meta={'tilted_stalk_prior':True},frame=10,drive=Mock())
        self.s=Session(env,SimpleNamespace(out=path),None,visual_proposals=True)
        self.s.centered=Mock();self.s.centered.adjust_observation.return_value={'ticks':30}
        local=self.s.local_targets
        local.register([dict(surface_m=[.27,.3,.22],maturity='ripe',visibility='clear',
                             body_pixel_verified=True,proposal_id='P1')],[0,0,0],'whole image')
        self.c=local.active;local.survey_review={'frame_id':'f1'};local.survey_state=self.state
        np.savez(path/'depth.npz',position=[0,0,.65],rotation=np.eye(3))
        self.s.depth_frames={('f1','harvest_close'):path/'depth.npz'}
        self.s.frames={'f10':{}}
        self.s.stalk_view_evidence={'frame_id':'f10','candidate_key':self.c['candidate_key'],
                                   'views':{'harvest_close':'no_fit','wrist':'fit_available'}}
        self.s.stalk_proposals={'W1':{'frame_id':'f10','candidate_key':self.c['candidate_key'],'camera':'wrist','proposal_id':'W1'}}
        self.args={'basis_frame_id':'f10','candidate_key':self.c['candidate_key'],'reason':'Stalk connection occluded',
                   'unconfirmed_stalk_proposals':[{'proposal_id':'W1','reason':'Cannot see connection to this fruit in rotated wrist view'}]}

    def dispatch(self,args=None):
        with patch('vista.robot_state',return_value=self.state):
            return self.s.dispatch('adjust_observation',args or self.args,'f10',self.state,0)[0]

    def test_explicit_review_moves_once_without_grasp_skip_or_fake_failure(self):
        self.assertFalse(observation_adjustment_available(self.s,'f10'))
        self.assertTrue(unconfirmed_stalk_options(self.s,'f10'))
        self.assertIn('unconfirmed_stalk_proposals',self.s.station_status()['next_action'])
        result=self.dispatch()
        self.assertEqual(result['recovery']['unconfirmed_stalk_review']['model_claims'],self.args['unconfirmed_stalk_proposals'])
        self.assertEqual(self.c['failures'],0);self.assertEqual(self.c['status'],'pending')
        self.assertFalse(self.s.target_confirmed or self.s.local_targets.can_search)
        self.assertFalse(self.s.stalk_proposals or self.s.frames)
        self.assertFalse(unconfirmed_stalk_options(self.s,'f10'))
        with self.assertRaises(ValueError):self.dispatch()
        self.s.centered.adjust_observation.assert_called_once()
        self.s.centered.observation_plan.assert_called_once_with(
            {'tcp_world_m':self.state['tcp_world_m'],'roll':self.state['joints_rad'][4]},
            0,body_point=self.c['surface_m'])

    def test_camera_framing_unavailable_preserves_original_safe_recovery(self):
        self.s.centered.observation_plan.side_effect=[ValueError('No bounded camera framing'), 'original-plan']
        self.dispatch()
        self.assertEqual(self.s.centered.observation_plan.call_count,2)
        self.s.centered.adjust_observation.assert_called_once_with('original-plan')
        self.assertEqual(self.c['failures'],0)
        self.assertFalse(self.s.local_targets.can_search)
        self.assertIn('camera_framing_fallback_reason',self.s.centered.observation_plan_diagnostics)

    def test_framing_and_fallback_rejected_before_motion(self):
        self.s.centered.observation_plan.side_effect=ValueError('unreachable path')
        with self.assertRaises(ValueError):self.dispatch()
        self.s.centered.adjust_observation.assert_not_called()
        self.assertFalse(self.c.get('observation_adjusted',False))

    def test_all_current_ids_and_nonempty_reasons_required(self):
        for claims in ([],[{'proposal_id':'C9','reason':'wrong'}],
                       [{'proposal_id':'W1','reason':' '}],self.args['unconfirmed_stalk_proposals']*2):
            with self.subTest(claims=claims),self.assertRaises(ValueError):
                self.dispatch({**self.args,'unconfirmed_stalk_proposals':claims})
        self.s.stalk_proposals['W2']={**self.s.stalk_proposals['W1'],'proposal_id':'W2'}
        with self.assertRaisesRegex(ValueError,'ALL current'):self.dispatch()
        self.s.centered.adjust_observation.assert_not_called()

    def test_stale_views_candidate_or_proposal_cannot_authorize(self):
        for where,field,value in ((self.s.stalk_view_evidence,'frame_id','old'),
                (self.s.stalk_view_evidence,'candidate_key','other'),
                (self.s.stalk_proposals['W1'],'frame_id','old'),
                (self.s.stalk_proposals['W1'],'candidate_key','other')):
            old=where[field];where[field]=value
            self.assertFalse(unconfirmed_stalk_options(self.s,'f10'));where[field]=old
        self.s.stalk_view_evidence['views'].pop('harvest_close')
        self.assertFalse(unconfirmed_stalk_options(self.s,'f10'))

    def test_budget_safety_and_coverage_remain_required(self):
        for obj,field,value in ((self.s,'target_stage','pregrasp'),(self.s,'grasp_attempted',True),
                (self.s,'target_confirmed',True),(self.s,'motion_stop_trace',{'reason':'stop'}),
                (self.s,'station_attempts',11),(self.s.local_targets,'observation_adjustments',2),
                (self.s.local_targets,'selection_blocked_reason','missing fruit')):
            existed=hasattr(obj,field);old=getattr(obj,field,None);setattr(obj,field,value)
            self.assertFalse(unconfirmed_stalk_options(self.s,'f10'))
            if existed:setattr(obj,field,old)
            else:delattr(obj,field)
        self.s.motion_stop_trace=None;self.s.target_stage=None;self.s.local_targets.selection_blocked_reason=None
        self.s.station_attempts=10
        self.dispatch();self.assertEqual(self.s.station_attempts,11)

    def test_finish_redirects_once_to_identity_review_without_moving(self):
        self.s.calls=4
        result,_=self.s.dispatch('finish',{'reason':'Cannot confirm W1'},'f10',self.state,0)
        self.assertFalse(result['episode_ended']);self.assertEqual(result['remaining_model_calls'],28)
        self.assertIn('unconfirmed_stalk_proposals',result['next_action'])
        self.s.centered.adjust_observation.assert_not_called()
        result,_=self.s.dispatch('finish',{'reason':'Recovery unsuitable'},'f10',self.state,0)
        self.assertTrue(result['episode_ended'])

    def test_failed_adjustment_stays_stopped(self):
        self.s.centered.adjust_observation.side_effect=ValueError('tracking')
        with self.assertRaises(MotionSafetyStop):self.dispatch()
        self.assertFalse(unconfirmed_stalk_options(self.s,'f10'))
        self.assertEqual(self.c['failures'],0)
        self.assertFalse(self.s.local_targets.can_search)
