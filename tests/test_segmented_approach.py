
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from vista import Session


class SegmentedApproachChecks(unittest.TestCase):
    def session(self,blocked=False):
        s=Session.__new__(Session)
        e=SimpleNamespace(meta={'tilted_stalk_prior':True},tcp=0,data=SimpleNamespace(site_xpos=np.zeros((1,3)),warning=[]),drive=Mock())
        e.last_action=np.zeros(3)
        e.ik=lambda point,gripper,approach_pitch,start=None:(np.asarray(point).copy(),0.)
        s.e=e;s.locked_target=np.array([.055,0,0]);s.target_confirmed=True
        s.maturity_evidence={'passed':True};s.target_stage='pregrasp';s.commands=[]
        def tick(q,phase):
            s.commands.append((q.copy(),phase));e.data.site_xpos[0]=q;e.last_action=q.copy()
            if blocked and phase=='visual_grasp' and q[0]>.017:
                e.data.site_xpos[0,2]+=.012
        s.tick=tick
        return s

    def test_small_segments_reach_target(self):
        s=self.session()
        with patch('vista.robot_state',return_value={}), patch('vista.command_tcp',side_effect=lambda e,q:q.copy()):result=s.execute_approach('grasp')
        points=np.array([[0,0,0]]+[r['waypoint_m'] for r in result['approach_trace']])
        self.assertLessEqual(np.linalg.norm(np.diff(points,axis=0),axis=1).max(),.00500001)
        self.assertEqual(s.target_stage,'grasp')
        self.assertLess(result['tcp_error_m'],.002)

    def test_blocked_segment_stops_deeper_commands_and_retreats(self):
        s=self.session(blocked=True)
        with patch('vista.command_tcp',side_effect=lambda e,q:q.copy()):
            with self.assertRaisesRegex(ValueError,'withdrew open') as caught:s.execute_approach('grasp')
        forward=[q for q,phase in s.commands if phase=='visual_grasp']
        self.assertLessEqual(max(q[0] for q in forward),.0200001)
        self.assertTrue(caught.exception.approach_trace[-1]['holding_for_tracking'])
        np.testing.assert_allclose(s.e.data.site_xpos[0],[0,0,0])
        self.assertIsNone(s.locked_target);self.assertIsNone(s.maturity_evidence)
        self.assertIsNone(s.target_stage)
