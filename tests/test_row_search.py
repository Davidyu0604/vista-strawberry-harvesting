
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from peduncle import PeduncleEnv
from run import Recorder
from vista import Session,robot_state
from recording_quality import phase_labels


class RowSearchChecks(unittest.TestCase):
    def test_real_search_camera_and_stale_targets(self):
        e=PeduncleEnv(seed=3,single=False,render_quality='fast')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                s=Session(e,Recorder(e,Path(tmp)/'episode',False,False),None)
                for _ in range(20):s.tick(e.last_action,'idle')
                frame,state,_=s.observe()
                camera=e.model.camera('harvest_close').id
                before=e.data.cam_xpos[camera].copy();base=e.base_pose().copy()
                s.station_attempts=6;s.station_failures=2
                s.local_targets.register([],state['base_x_y_yaw'],'No visible local mature candidate in this motion fixture')
                s.crops['old']={};s.locked_target=np.array([.3,0,.3]);s.target_confirmed=True
                s.maturity_evidence={'passed':True}
                result,_=s.dispatch('search_row',{'basis_frame_id':frame,'direction':1},frame,state,0)
                self.assertEqual(result['status'],'executed')
                self.assertAlmostEqual(e.base_pose()[1]-base[1],.10,delta=.005)
                np.testing.assert_allclose(e.data.cam_xpos[camera]-before,[0,.10,0],atol=.006)
                self.assertEqual(s.station_attempts,0);self.assertEqual(s.station_failures,0)
                self.assertFalse(s.crops);self.assertFalse(s.depth_frames);self.assertFalse(s.frames)
                self.assertIsNone(s.locked_target);self.assertIsNone(s.maturity_evidence)
                self.assertIsNone(s.target_stage)
                np.testing.assert_array_equal(e.drive_command,[0,0])
                with self.assertRaises(ValueError):
                    s.dispatch('locate_pixel',{'frame_id':frame,'camera':'harvest_close','x':200,'y':100},frame,state,0)
                frame,state,_=s.observe()
                self.assertFalse(s.local_targets.surveyed)
                s.local_targets.register([],state['base_x_y_yaw'],'No local candidate in this motion fixture')
                result,_=s.dispatch('search_row',{'basis_frame_id':frame,'direction':-1},frame,state,0)
                self.assertEqual(result['status'],'executed')
                self.assertAlmostEqual(e.base_pose()[1],base[1],delta=.005)
                self.assertTrue(e.attached.all())
                self.assertFalse(any(w.number for w in e.data.warning))
                self.assertEqual(phase_labels('search_drive'),('search',False,False))
                frame,state,_=s.observe()
                s.locked_target=np.array([.3,0,.3]);s.target_confirmed=True
                s.maturity_evidence={'passed':True,'target_m':s.locked_target.copy()}
                s.maturity_evidence['axis_world']=[0,0,1]
                s.parking_trial={'enabled':True,'status':'pending','requested_delta_y_m':.04,'grasp_height_offset_m':0.}
                s.local_targets=None  # Explicit legacy +4cm mechanical trial, not local-first policy.
                with patch.object(s,'check_maturity',return_value=(s.maturity_evidence,None)):
                    result,_=s.dispatch('approach_target',{'basis_frame_id':frame,'stage':'pregrasp'},frame,state,0)
                self.assertTrue(result['requires_relocalization'])
                self.assertFalse(result['pregrasp_executed'])
                self.assertEqual(s.parking_trial['status'],'applied')
                with self.assertRaisesRegex(ValueError,'not pending'):s.apply_parking_trial()
                self.assertAlmostEqual(e.base_pose()[1]-base[1],.04,delta=.005)
                self.assertIsNone(s.locked_target);self.assertIsNone(s.target_reachability)
                self.assertFalse(s.frames);self.assertFalse(s.depth_frames);self.assertFalse(s.crops)
                np.testing.assert_array_equal(e.drive_command,[0,0])
                with patch.object(e,'ik',return_value=(e.last_action.copy(),.1)):
                    with self.assertRaisesRegex(ValueError,'unreachable'):s.search_row(1)
                np.testing.assert_array_equal(e.drive_command,[0,0])
                self.assertEqual(s.target_stage,'searching')
                with self.assertRaisesRegex(ValueError,'before approach'):s.search_row(1)
        finally:e.close()

    def test_budget_and_approach_guards(self):
        s=Session.__new__(Session)
        s.e=SimpleNamespace(meta={'tilted_stalk_prior':True})
        s.grasp_attempted=False;s.target_stage=None;s.search_steps=4;s.calls=0;s.max_calls=32
        with self.assertRaisesRegex(ValueError,'budget'):s.search_row(1)
        s.search_steps=0;s.calls=23
        with self.assertRaisesRegex(ValueError,'budget'):s.search_row(1)
        s.calls=0;s.target_stage='grasp'
        with self.assertRaisesRegex(ValueError,'before approach'):s.search_row(1)
