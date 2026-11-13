
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
from target_parking import optimized_parking_plan
from vista import Session


class OptimizedParkingChecks(unittest.TestCase):
    def env(self):
        return SimpleNamespace(base_pose=lambda:np.zeros(3),meta={'harvest_base_y':0})

    def candidate(self, env, target, delta):
        # Objective changes with the measured fruit, not a constant direction.
        return {'reachable':True,'base_delta_y_m':delta,'score':10*abs(target[1]-delta)}

    def test_direction_depends_on_measured_target_even_when_current_reachable(self):
        with patch('target_parking.scored_parking_candidate',side_effect=self.candidate):
            for y in (-.07,.03):
                p=optimized_parking_plan(self.env(),[.3,y,.3])
                self.assertTrue(p['current']['reachable'])
                self.assertAlmostEqual(p['recommended_delta_y_m'],y)

    def test_keep_good_current_parking_and_bound_candidates(self):
        with patch('target_parking.scored_parking_candidate',side_effect=self.candidate):
            p=optimized_parking_plan(self.env(),[.3,.003,.3])
            self.assertIsNone(p['recommended_delta_y_m'])
            e=self.env();e.base_pose=lambda:np.array([0,.295,0])
            p=optimized_parking_plan(e,[.3,.08,.3])
            self.assertTrue(all(c['base_delta_y_m']<=0 for c in p['candidates']))

    def test_no_reachable_candidate_does_not_drive(self):
        with patch('target_parking.scored_parking_candidate',side_effect=lambda e,t,d:{'reachable':False,'score':None,'base_delta_y_m':d}):
            p=optimized_parking_plan(self.env(),[.3,0,.3])
            self.assertIsNone(p['recommended_delta_y_m'])
            self.assertIn('search_row',p['next_action'])

    def session(self):
        s=Session.__new__(Session);s.e=self.env();s.target_aware_parking=True
        s.locked_target=np.array([.3,0,.3]);s.target_confirmed=True;s.grasp_attempted=False
        s.target_stage=None;s.max_calls=32;s.calls=10;s.search_steps=0
        s.maturity_evidence={'passed':True,'target_m':s.locked_target.copy(),'axis_world':[0,0,1]}
        s.parking_history=[];s.search_row=Mock(return_value={'status':'executed'})
        return s

    def plan(self):
        return {'current':{'reachable':True},'recommended_delta_y_m':-.06,'next_action':'refine_parking'}

    def test_relocalization_anchor_prevents_oscillation_and_budget_keeps_reachable(self):
        s=self.session();s.parking_anchor={'target_m':[.3,0,.3],'actual_parking_y_m':0}
        with patch('vista.optimized_parking_plan',side_effect=lambda *a:self.plan()):
            self.assertIsNone(s.plan_parking()['recommended_delta_y_m'])
            s.locked_target[1]=.04
            self.assertEqual(s.plan_parking()['recommended_delta_y_m'],-.06)
            s.search_steps=4
            self.assertIsNone(s.plan_parking()['recommended_delta_y_m'])

    def test_pregrasp_parks_instead_of_extending_arm(self):
        s=self.session();s.e.meta.update(harvest_mechanism=True,tilted_stalk_prior=True)
        s.max_wait=90;s.check_maturity=Mock(return_value=(s.maturity_evidence,None))
        s.execute_approach=Mock()
        with patch.object(s,'plan_parking',return_value=self.plan()),patch('vista.check_observation'),patch('vista.robot_state',return_value={}):
            receipt,_=s._dispatch('approach_target',{'basis_frame_id':'f','stage':'pregrasp'},'f',{},0)
        s.search_row.assert_called_once_with(-1,distance=.06)
        s.execute_approach.assert_not_called()
        self.assertTrue(receipt['requires_relocalization']);self.assertFalse(receipt['pregrasp_executed'])
        self.assertEqual(len(s.parking_history),1)

    def test_stale_maturity_cannot_drive(self):
        s=self.session();s.maturity_evidence['target_m']=[.4,0,.3]
        with self.assertRaisesRegex(ValueError,'maturity'):s.refine_parking()
        s.search_row.assert_not_called()

class RealKinematicChecks(unittest.TestCase):
    def test_candidate_translation_equivalence_and_live_state_unchanged(self):
        import mujoco
        from peduncle import PeduncleEnv
        from target_parking import scored_parking_candidate
        e=PeduncleEnv(seed=7,single=False,render_quality='fast')
        try:
            original=e.data.qpos.copy();velocity=e.data.qvel.copy();stamp=e.data.time
            target=np.array([.3333,.0288,.3203]);delta=-.06
            predicted=scored_parking_candidate(e,target,delta)
            np.testing.assert_array_equal(e.data.qpos,original)
            np.testing.assert_array_equal(e.data.qvel,velocity)
            self.assertEqual(e.data.time,stamp)
            # Scratch kinematic verification only, not a physical driving test.
            e.data.qpos[e.base_qa[1]]+=delta
            mujoco.mj_forward(e.model,e.data)
            observe,error=e.ik([.19,e.base_pose()[1]-.15,.33],1,approach_pitch=0)
            e.data.qpos[e.qa]=observe;mujoco.mj_forward(e.model,e.data)
            actual=scored_parking_candidate(e,target,0.)
            self.assertEqual(actual['reachable'],predicted['reachable'])
            # Moving candidates also include the observation-posture IK error.
            self.assertAlmostEqual(max(error,actual['max_ik_error_m']),predicted['max_ik_error_m'],delta=1e-6)
            self.assertAlmostEqual(actual['approach_arc_deviation_m'],predicted['approach_arc_deviation_m'],delta=1e-6)
            with self.assertRaises(ValueError):scored_parking_candidate(e,[float('nan'),0,0],0)
        finally:e.close()

if __name__=='__main__':unittest.main()
