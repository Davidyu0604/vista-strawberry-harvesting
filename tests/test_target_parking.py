
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from target_parking import approach_reachability, parking_plan
from vista import Session


class TargetParkingChecks(unittest.TestCase):
    def env(self):
        # Translation-equivariant toy arm: Y workspace [-.1,.1].
        return SimpleNamespace(state=lambda:np.zeros(6),base_pose=lambda:np.zeros(3),
            meta={'harvest_base_y':0},
            ik=lambda p,*a,**k:(np.zeros(6),max(0,abs(p[1])-.1)))

    def test_reachable_target_does_not_move(self):
        plan=parking_plan(self.env(),[.3,.05,.3])
        self.assertTrue(plan['current']['reachable'])
        self.assertIsNone(plan['recommended_delta_y_m'])

    def test_shortest_valid_refinement_and_unreachable_fallback(self):
        plan=parking_plan(self.env(),[.3,.135,.3])
        self.assertFalse(plan['current']['reachable'])
        self.assertEqual(plan['recommended_delta_y_m'],.04)
        plan=parking_plan(self.env(),[.3,.3,.3])
        self.assertIsNone(plan['recommended_delta_y_m'])
        self.assertIn('search_row',plan['next_action'])

    def test_envelope_and_nonfinite_solver_result(self):
        e=self.env();e.base_pose=lambda:np.array([0,.29,0])
        self.assertIsNone(parking_plan(e,[.3,.135,.3])['recommended_delta_y_m'])
        e.ik=lambda *a,**k:(np.zeros(6),float('nan'))
        self.assertFalse(approach_reachability(e,[.3,0,.3])['reachable'])

    def test_intermediate_unreachable_blocks_despite_reachable_endpoints(self):
        e=self.env();e.ik=lambda p,*a,**k:(np.zeros(6),.02 if .26<p[0]<.28 else 0)
        self.assertFalse(approach_reachability(e,[.3,0,.3])['reachable'])

    def test_preflight_rejects_before_maturity_or_motion(self):
        s=Session.__new__(Session);s.e=self.env();s.e.meta.update(harvest_mechanism=True,tilted_stalk_prior=True)
        s.locked_target=np.array([.3,.135,.3]);s.target_confirmed=True;s.grasp_attempted=False
        s.maturity_evidence={'passed':True,'target_m':s.locked_target.copy()}
        s.execute_approach=Mock();s.check_maturity=Mock()
        result,_=s._dispatch('approach_target',{'basis_frame_id':'f','stage':'pregrasp'},'f',{},0)
        self.assertEqual(result['status'],'rejected')
        s.execute_approach.assert_not_called();s.check_maturity.assert_not_called()

    def test_refinement_forbidden_with_arm_extended(self):
        s=Session.__new__(Session);s.locked_target=np.ones(3);s.target_confirmed=True
        s.grasp_attempted=False;s.target_stage='pregrasp'
        with self.assertRaisesRegex(ValueError,'before approach'):s.refine_parking()

    def test_refinement_uses_program_selected_distance(self):
        s=Session.__new__(Session);s.e=self.env();s.locked_target=np.array([.3,.135,.3])
        s.target_confirmed=True;s.grasp_attempted=False;s.target_stage=None
        s.maturity_evidence={'passed':True,'target_m':s.locked_target.copy()};s.search_row=Mock(return_value={'status':'executed'})
        result=s.refine_parking()
        s.search_row.assert_called_once_with(1,distance=.04)
        self.assertTrue(result['requires_relocalization'])
