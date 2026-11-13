
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from copy import deepcopy
from vista import Session
from perception_recovery import exhausted_observation_recovery
from perception_recovery import out_of_view_after_recovery
import numpy as np

class RecoveryExhaustionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.s=Session(SimpleNamespace(meta={'tilted_stalk_prior':True}),SimpleNamespace(out=Path(self.tmp.name)),None,visual_proposals=True)
        self.s.centered=Mock();self.s.log=Mock()
        self.s.local_targets.register([dict(surface_m=[.3,.35,.25],maturity='ripe',visibility='partial',body_pixel_verified=True,proposal_id='P1')],[0,0,0],'whole view')
        self.c=self.s.local_targets.active
        self.c.update(failures=1,cross_view_failed=True,observation_adjusted=True,observation_recovery_completed_frame=100)
        self.s.stalk_view_evidence=dict(frame_id='f110',candidate_key=self.c['candidate_key'],views={'harvest_close':'no_fit','wrist':'outside_crop'})
        self.state=dict(joints_rad=[0]*6,tcp_world_m=[.2,0,.3],base_x_y_yaw=[0,0,0])
        self.s.search_row=Mock(side_effect=self.search)
    def search(self,direction):
        self.s.require_local_search()
        return {'status':'executed','stage':'search'}
    def dispatch(self,frame='f110',state=None):
        with patch('vista.robot_state',return_value=self.state):
            return self.s.dispatch('search_row',{'basis_frame_id':frame,'direction':1},frame,state or self.state,0)[0]
    def test_completed_recovery_is_second_attempt_and_allows_search(self):
        self.assertIn('resolve the failed',self.s.station_status()['next_action'])
        result=self.dispatch()
        self.assertEqual(result['status'],'executed');self.assertEqual(self.c['failures'],2)
        self.assertEqual(self.c['status'],'unavailable')
        self.assertEqual(result['candidate_transition']['views']['wrist'],'outside_crop')
        self.assertIsNone(self.s.stalk_view_evidence)
        self.assertEqual(len(self.s.perception_recoveries),1)
    def test_empty_initial_views_or_unfinished_motion_are_not_exhaustion(self):
        for field,value in [('failures',0),('cross_view_failed',False),('observation_adjusted',False),('observation_recovery_completed_frame',None),('maturity','uncertain'),('body_pixel_verified',False)]:
            old=self.c[field];self.c[field]=value
            with self.subTest(field=field):self.assertFalse(exhausted_observation_recovery(self.s,'f110'))
            self.c[field]=old
    def test_a_fit_in_either_view_requires_actual_localization(self):
        for camera in ('harvest_close','wrist'):
            old=self.s.stalk_view_evidence['views'][camera]
            self.s.stalk_view_evidence['views'][camera]='fit_available'
            self.assertFalse(exhausted_observation_recovery(self.s))
            self.s.stalk_view_evidence['views'][camera]=old
    def test_semantic_recovery_defers_station_without_fake_pixel_failures(self):
        self.c.update(failures=0,cross_view_failed=False,semantic_recovery_review_frame=105)
        result=self.dispatch()
        self.assertEqual(result['status'],'executed')
        self.assertEqual(result['candidate_transition']['resolution'],'station_viewpoint_exhausted')
        self.assertEqual(self.c['failures'],0);self.assertEqual(self.s.station_failures,0)
        self.assertTrue(self.c['station_deferred']);self.assertEqual(self.c['status'],'unavailable')
    def test_semantic_recovery_needs_post_motion_review_and_later_empty_views(self):
        self.c.update(failures=0,cross_view_failed=False)
        for reviewed in (None,99,100,110,111):
            self.c['semantic_recovery_review_frame']=reviewed
            self.assertFalse(exhausted_observation_recovery(self.s,'f110'))
        self.c['semantic_recovery_review_frame']=105
        self.s.stalk_view_evidence['views']['wrist']='fit_available'
        self.assertFalse(exhausted_observation_recovery(self.s,'f110'))
    def test_old_frame_or_wrong_identity_not_counted(self):
        self.assertFalse(exhausted_observation_recovery(self.s,'f111'))
        for field,value in [('frame_id','f100'),('candidate_key','different')]:
            old=self.s.stalk_view_evidence[field];self.s.stalk_view_evidence[field]=value
            self.assertFalse(exhausted_observation_recovery(self.s))
            self.s.stalk_view_evidence[field]=old
    def test_safety_stop_or_confirmed_target_cannot_be_bypassed(self):
        for field,value in [('target_stage','pregrasp'),('grasp_attempted',True),('target_confirmed',True),('motion_stop_trace',{'reason':'stopped'})]:
            old=getattr(self.s,field,None);setattr(self.s,field,value)
            self.assertFalse(exhausted_observation_recovery(self.s));setattr(self.s,field,old)
    def test_other_candidates_still_prevent_driving(self):
        for status in ('pending','unverified_semantics','unverified_measurement'):
            self.setUp()
            other={**deepcopy(self.c),'proposal_id':'P2','candidate_key':'view-candidate-2','status':status,'failures':0}
            self.s.local_targets.candidates.append(other)
            result=self.dispatch()
            self.assertEqual(result['status'],'rejected');self.s.search_row.assert_not_called()
            self.assertEqual(other['failures'],0);self.assertEqual(self.c['failures'],2)
    def test_stale_robot_state_does_not_spend_attempt(self):
        with self.assertRaisesRegex(ValueError,'stale'):
            self.dispatch(state={**self.state,'tcp_world_m':[.4,0,.3]})
        self.assertEqual(self.c['failures'],1);self.s.search_row.assert_not_called()
    def test_unknown_coverage_blocks_even_with_recovery(self):
        self.s.local_targets.selection_blocked_reason='missing ripe fruit'
        self.assertFalse(exhausted_observation_recovery(self.s))
        with self.assertRaisesRegex(ValueError,'Local-first'):self.dispatch()
        self.assertEqual(self.c['failures'],1)
    def test_actual_search_still_enforces_movement_and_call_budgets(self):
        self.s.search_row=Session.search_row.__get__(self.s)
        self.s.search_steps=4
        with self.assertRaisesRegex(ValueError,'budget exhausted'):self.dispatch()
        self.assertEqual(self.c['failures'],2)
        self.s.search_steps=0;self.s.max_calls=32;self.s.calls=23
        with self.assertRaisesRegex(ValueError,'budget exhausted'):self.dispatch()
        self.assertEqual(len(self.s.perception_recoveries),1)

    def outside_fixture(self):
        self.c.update(semantic_recovery_review_frame=105,cross_view_failure_frame=108,
                      cross_view_target_m=[3.,0.,-1.],
                      cross_view_checks={'wrist':{'consistent':False,'reason':'outside_image'}})
        self.s.stalk_view_evidence['views']={'harvest_close':'fit_available','wrist':'outside_crop'}
        path=Path(self.tmp.name)/'depth.npz'
        np.savez(path,depth=np.ones((80,80)),rotation=np.eye(3),position=np.zeros(3),fovy=60.)
        self.s.depth_frames={('f110','wrist'):path}

    def test_post_semantic_recovery_outside_camera_defers_without_fake_failures(self):
        self.outside_fixture()
        self.assertTrue(out_of_view_after_recovery(self.s,'f110'))
        result=self.dispatch()
        self.assertEqual(result['status'],'executed')
        self.assertEqual(result['candidate_transition']['resolution'],'station_viewpoint_exhausted')
        self.assertEqual(self.c['failures'],1)
        self.assertTrue(self.c['station_deferred'])
        self.assertFalse(self.s.target_confirmed)

    def test_outside_requires_actual_fit_after_recovery_and_new_projection(self):
        self.outside_fixture()
        for field,value in [('cross_view_failure_frame',100),('semantic_recovery_review_frame',109),
                            ('cross_view_target_m',[0.,0.,-1.]),('cross_view_checks',{'wrist':{'reason':'invalid_depth'}}),
                            ('body_pixel_verified',False),('failures',0)]:
            old=self.c[field];self.c[field]=value
            with self.subTest(field=field):self.assertFalse(out_of_view_after_recovery(self.s,'f110'))
            self.c[field]=old
        self.assertFalse(out_of_view_after_recovery(self.s,'f111'))
        self.s.depth_frames={}
        self.assertFalse(out_of_view_after_recovery(self.s,'f110'))

    def test_outside_does_not_bypass_other_candidates_or_coverage(self):
        self.outside_fixture()
        self.s.local_targets.selection_blocked_reason='missing ripe fruit'
        self.assertFalse(out_of_view_after_recovery(self.s))
        self.s.local_targets.selection_blocked_reason=None
        other={**deepcopy(self.c),'candidate_key':'view-candidate-2','proposal_id':'P2',
               'status':'unverified_measurement','failures':0}
        self.s.local_targets.candidates.append(other)
        result=self.dispatch()
        self.assertEqual(result['status'],'rejected');self.s.search_row.assert_not_called()
        self.assertEqual(other['failures'],0)

if __name__=='__main__':unittest.main()
