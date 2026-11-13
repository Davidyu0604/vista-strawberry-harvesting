import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
import numpy as np
from centered_grasp import CenteredGrasp
from unittest.mock import Mock, patch
from vista import Session


class DeliveryBudgetTests(unittest.TestCase):
    def setUp(self):
        self.controller=CenteredGrasp.__new__(CenteredGrasp)
        self.controller.e=SimpleNamespace(last_action=np.zeros(6))
        self.controller.fk=lambda q:np.asarray(q[:3])

    def test_tcp_limit_determines_duration_without_moving(self):
        c=self.controller;goal=np.array([.4,0,0,0,0,0])
        budget=c.waypoint_ticks(goal,70)
        self.assertGreater(budget['command_ticks'],70)
        self.assertEqual(budget['total_ticks'],budget['command_ticks']+10)
        np.testing.assert_array_equal(c.e.last_action,np.zeros(6))
        q=c.e.last_action.copy()
        for _ in range(budget['command_ticks']):
            new,_=c.command(goal,start=q)
            self.assertLessEqual(np.linalg.norm(new[:3]-q[:3]),.005+1e-12)
            self.assertLessEqual(max(abs(new-q)),.04+1e-12);q=new
        np.testing.assert_allclose(q,goal,atol=1e-8)

    def test_short_path_keeps_original_duration(self):
        budget=self.controller.waypoint_ticks(np.array([.02,0,0,0,0,0]),70)
        self.assertEqual(budget['total_ticks'],70)

    def test_excessive_path_is_rejected_before_motion(self):
        with self.assertRaisesRegex(ValueError,'180'):
            self.controller.waypoint_ticks(np.array([2.,0,0,0,0,0]),70)
        np.testing.assert_array_equal(self.controller.e.last_action,np.zeros(6))

    def test_longer_budget_still_stops_after_four_lost_contact_ticks(self):
        s=Session.__new__(Session);s.grasp_attempted=True
        s.e=SimpleNamespace(meta={'harvest_mechanism':True},frame=0,tcp=0,
            data=SimpleNamespace(site_xpos=np.zeros((1,3)),warning=[]),drive=Mock(),last_action=np.zeros(6))
        s.centered=SimpleNamespace(ik=lambda *args:(np.zeros(6),0.),
                                  waypoint_ticks=lambda *args:{'total_ticks':110})
        def tick(*args):s.e.frame+=1
        s.tick=tick
        with patch('vista.jaw_contact_state',side_effect=[{'both':True}]+[{'both':False}]*4):
            with self.assertRaisesRegex(ValueError,'contact lost'):s.deliver_to_basket()
        self.assertEqual(s.e.frame,4)
        self.assertEqual(s.motion_stop_trace['phase'],'delivery_carry')
