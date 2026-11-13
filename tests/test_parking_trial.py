
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import unittest
from unittest.mock import Mock,patch
from types import SimpleNamespace
import numpy as np
from vista import Session


class ParkingTrialChecks(unittest.TestCase):
    def session(self):
        s=Session.__new__(Session)
        s.e=SimpleNamespace(meta={'harvest_mechanism':True,'tilted_stalk_prior':True},base_pose=lambda:np.zeros(3))
        s.locked_target=np.array([.3,0,.3]);s.target_confirmed=True;s.target_stage=None;s.grasp_attempted=False
        s.maturity_evidence={'passed':True,'target_m':[.3,0,.3],'axis_world':[0,0,1]}
        s.check_maturity=Mock(return_value=(s.maturity_evidence,None));s.execute_approach=Mock(return_value={'status':'executed'})
        s.search_row=Mock(return_value={'status':'executed','stage':'search'})
        s.max_wait=90;s.parking_trial={'enabled':True,'status':'pending','requested_delta_y_m':.04,'grasp_height_offset_m':0.}
        return s

    def dispatch(self,s):
        with patch('vista.robot_state',return_value={}),patch('vista.check_observation'),patch('vista.parking_plan',return_value={'current':{'reachable':True}}):
            return s._dispatch('approach_target',{'basis_frame_id':'f','stage':'pregrasp'},'f',{},0)[0]

    def test_first_pregrasp_parks_once_and_does_not_approach(self):
        s=self.session();result=self.dispatch(s)
        s.search_row.assert_called_once_with(1,distance=.04)
        self.assertFalse(result['pregrasp_executed']);s.execute_approach.assert_not_called()
        self.assertEqual(s.parking_trial['status'],'applied')
        # Simulate a fresh confirmed localization after actual cache invalidation.
        self.dispatch(s)
        self.assertEqual(s.search_row.call_count,1);s.execute_approach.assert_called_once_with('pregrasp')

    def test_maturity_failure_does_not_move(self):
        s=self.session();s.check_maturity.return_value=({'passed':False},None)
        with patch('vista.image_block',return_value={}):result=self.dispatch(s)
        self.assertEqual(result['status'],'rejected');s.search_row.assert_not_called()
        self.assertEqual(s.parking_trial['status'],'pending')

    def test_parking_failure_cannot_be_bypassed_by_retry(self):
        s=self.session();s.search_row.side_effect=ValueError('Outside envelope')
        with self.assertRaisesRegex(ValueError,'Outside'):self.dispatch(s)
        self.assertEqual(s.parking_trial['status'],'failed')
        with self.assertRaisesRegex(ValueError,'failed'):self.dispatch(s)
        self.assertEqual(s.search_row.call_count,1);s.execute_approach.assert_not_called()

    def test_stale_frame_and_adaptive_refinement_rejected(self):
        s=self.session()
        with self.assertRaisesRegex(ValueError,'Invalid'):
            s._dispatch('approach_target',{'basis_frame_id':'old','stage':'pregrasp'},'new',{},0)
        with self.assertRaisesRegex(ValueError,'disabled'):s.refine_parking()
        s.search_row.assert_not_called()
