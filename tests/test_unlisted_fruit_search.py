
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from PIL import Image
from local_targets import LocalTargets
from perception_recovery import MISSING_BLOCK
from unlisted_fruit_search import measure_unlisted,review_available
from vista import Session


class UnlistedSearchTests(unittest.TestCase):
    def setUp(self):
        a=np.zeros((60,80,3),dtype=np.uint8);a[20:40,30:50]=[220,0,0]
        self.rgb=Image.fromarray(a);self.depth=np.full((60,80),.8)
        self.claims=[dict(x=40,y=30,reason='Visible red body, no numbered proposal')]
        self.point=np.array([.7,.3,.25])
        self.unproject=lambda *args:self.point.copy()
        self.local=LocalTargets();self.local.register([], [0,0,0],'red fruit outside numbered queue')
        self.local.selection_blocked_reason=MISSING_BLOCK
        self.local.survey_review={'missing_visible_ripe':True}
        self.s=SimpleNamespace(local_targets=self.local,visual_proposals=True,target_stage=None,
            grasp_attempted=False,target_confirmed=False,frames={'f1':{}},
            unlisted_search_evidence={'frame_id':'f1'},e=SimpleNamespace(meta={'tilted_stalk_prior':True,'mobile_layout':True}))

    def measure(self):
        return measure_unlisted(self.rgb,self.depth,{},[0,0,0],self.claims,self.unproject)

    def test_far_body_permits_observation_without_clearing_claim(self):
        result=self.measure();self.assertEqual(result['direction'],1)
        self.assertGreater(result['bodies'][0]['minimum_distance_xy_m'],.58)
        self.assertTrue(review_available(self.s))
        Session.require_local_search(self.s,observation_only=True)
        with self.assertRaises(ValueError):Session.require_local_search(self.s)
        self.assertEqual(self.local.selection_blocked_reason,MISSING_BLOCK)
        self.assertTrue(self.local.survey_review['missing_visible_ripe'])
        self.assertFalse(self.local.can_search)

    def test_near_boundary_and_distant_bodies_rejected(self):
        for point in ([.3,0,.25],[.57,0,.25],[1.3,.2,.25],[.7,.3,.5]):
            self.point=np.array(point)
            with self.assertRaises(ValueError):self.measure()

    def test_unknown_depth_and_wrong_pixel_rejected(self):
        self.depth[30,40]=np.nan
        with self.assertRaises(ValueError):self.measure()
        self.depth[:]=.8;self.claims[0]['x']=10
        with self.assertRaises(ValueError):self.measure()
        self.claims[0]['x']=True
        with self.assertRaises(ValueError):self.measure()

    def test_other_local_region_cannot_be_skipped(self):
        def measured(d,c,x,y):return np.array([.3,0,.25]) if x<38 else self.point.copy()
        with self.assertRaises(ValueError):
            measure_unlisted(self.rgb,self.depth,{},[0,0,0],self.claims,measured)
        self.local.candidates=[{'status':'unverified_measurement'}]
        self.assertFalse(review_available(self.s))

    def test_duplicate_reports_rejected_and_reverse_direction(self):
        self.claims*=2
        with self.assertRaises(ValueError):self.measure()
        self.claims=self.claims[:1];self.point[1]=-.3
        self.assertEqual(self.measure()['direction'],-1)

    def test_motion_and_frame_guards(self):
        for key,value in [('fixed_base',True),('grasp_attempted',True),('target_stage','pregrasp'),
                          ('motion_stop_trace',{'reason':'stopped'}),('target_confirmed',True)]:
            old=getattr(self.s,key,None);setattr(self.s,key,value)
            self.assertFalse(review_available(self.s));setattr(self.s,key,old)
        self.s.frames={}
        with self.assertRaises(ValueError):Session.require_local_search(self.s,observation_only=True)

    def test_existing_move_and_call_limits_retained(self):
        self.s.require_local_search=lambda **kw:Session.require_local_search(self.s,**kw)
        self.s.search_steps=4;self.s.calls=3;self.s.max_calls=32
        with self.assertRaisesRegex(ValueError,'budget exhausted'):Session.search_row(self.s,1)
        self.s.search_steps=0;self.s.calls=23
        with self.assertRaisesRegex(ValueError,'budget exhausted'):Session.search_row(self.s,1)

    def test_stale_observation_rejected_before_read_or_motion(self):
        self.s.max_wait=30;self.s.search_row=Mock()
        args=dict(basis_frame_id='f1',all_missing_claims_reported=True,bodies=self.claims)
        with patch('vista.robot_state',return_value={}),patch('vista.check_observation',side_effect=ValueError('Observation stale')):
            with self.assertRaisesRegex(ValueError,'stale'):Session._dispatch(self.s,'search_after_unlisted_review',args,'f1',{},40)
        self.s.search_row.assert_not_called()

    def test_failed_motion_does_not_leave_reusable_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.rgb.save(p/'rgb.jpg');np.savez(p/'depth.npz',depth=self.depth)
            self.s.frames={'f1':{'harvest_close':p/'rgb.jpg'}}
            self.s.depth_frames={('f1','harvest_close'):p/'depth.npz'}
            self.s.max_wait=30;self.s.unlisted_search_evidence=None
            def fail(direction):
                self.assertIsNotNone(self.s.unlisted_search_evidence)
                raise ValueError('Search stow blocked')
            self.s.search_row=Mock(side_effect=fail)
            args=dict(basis_frame_id='f1',all_missing_claims_reported=True,bodies=self.claims)
            with patch('vista.robot_state',return_value={}),patch('vista.check_observation'),patch('vista.depth_point',self.unproject):
                with self.assertRaisesRegex(ValueError,'stow blocked'):
                    Session._dispatch(self.s,'search_after_unlisted_review',args,'f1',{'base_x_y_yaw':[0,0,0]},0)
            self.assertIsNone(self.s.unlisted_search_evidence)
            self.assertEqual(self.local.selection_blocked_reason,MISSING_BLOCK)


if __name__=='__main__':unittest.main()
