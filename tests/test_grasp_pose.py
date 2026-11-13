
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import unittest
import numpy as np
from grasp_pose import JawCalibration, line_interval, solve_stalk_pose


class SectionChecks(unittest.TestCase):
    def test_box_and_missing_cross_section(self):
        planes=np.array([[1,0,0,-.02],[-1,0,0,.01],[0,1,0,-.01],
                         [0,-1,0,-.01],[0,0,1,.06],[0,0,-1,-.1]])
        np.testing.assert_allclose(line_interval(planes,0,-.08),[.01,.02])
        self.assertIsNone(line_interval(planes,.02,-.08))
        self.assertIsNone(line_interval(planes,0,-.04))

    def test_invalid_vision_rejected_before_kinematics(self):
        for point,axis in [([np.nan,0,0],[0,0,1]),([0,0,0],[0,0,0])]:
            with self.assertRaises(ValueError):
                solve_stalk_pose(None,point,axis,np.zeros(6),.2,np.zeros(3))


class FullSceneCalibrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from peduncle import PeduncleEnv
        cls.env=PeduncleEnv(seed=7,single=False,render_quality='fast')
        cls.cal=JawCalibration(cls.env)

    @classmethod
    def tearDownClass(cls):cls.env.close()

    def test_full_open_finger_does_not_cover_pad_depth(self):
        with self.assertRaisesRegex(ValueError,'does not cover'):self.cal.center(1.)

    def test_center_between_faces_and_moves_with_opening(self):
        wider=self.cal.aperture(.2);narrower=self.cal.aperture(.1)
        self.assertGreater(wider,narrower)
        self.assertGreater(narrower,.003)
        np.testing.assert_allclose(self.cal.center(.2),[wider/2,0,-.080],atol=1e-8)
        with self.assertRaisesRegex(ValueError,'positive'):self.cal.center(-.15)

    def test_robot_only_calibration_and_solver_do_not_mutate_live_state(self):
        e=self.env;before=e.data.qpos.copy();vel=e.data.qvel.copy();stamp=e.data.time
        center=self.cal.center(.2)
        # Archived RGB-D target; no object-state access by the planner.
        point=np.array([.2627338681,-.1515617338,.2299090231])
        axis=np.array([.0868240888,-.1503837332,.984807753])
        q,error=solve_stalk_pose(e,point,axis,e.state(),.2,center)
        self.assertLess(error,.006)
        self.assertEqual(q[-1],.2)
        rot=e.ikdata.xmat[e.grip_body].reshape(3,3)
        self.assertLess(abs(rot[:,0]@axis),.02)
        actual_center=e.ikdata.xpos[e.grip_body]+rot@center
        self.assertAlmostEqual(np.linalg.norm(actual_center-point),error,places=10)
        np.testing.assert_array_equal(e.data.qpos,before)
        np.testing.assert_array_equal(e.data.qvel,vel)
        self.assertEqual(e.data.time,stamp)

    def test_transport_ik_holds_roll_without_mutating_physics(self):
        from validate_centered_cycle import fixed_roll_ik
        e=self.env;before=e.data.qpos.copy();velocity=e.data.qvel.copy()
        roll=-1.6
        for pos,pitch in [([.228,-.151,.230],np.deg2rad(-37)),([.15,-.15,.28],-.35)]:
            q,error=fixed_roll_ik(e,np.array(pos),-.15,roll,pitch)
            self.assertEqual(q[4],roll)
            self.assertEqual(q[5],-.15)
            self.assertLess(error,.006)
        np.testing.assert_array_equal(e.data.qpos,before)
        np.testing.assert_array_equal(e.data.qvel,velocity)

    def test_recorded_pregrasp_self_collision_detected_using_robot_only(self):
        from grasp_pose import RobotSelfCollision
        e=self.env;check=RobotSelfCollision(e);before=e.data.qpos.copy()
        q=np.array([.1503073057,-1.0178197635,1.6126745229,-.5949012850,-1.4354070397,.2])
        hit=check.collision(q)
        self.assertIsNotNone(hit)
        self.assertEqual(set(hit['bodies']),{'shoulder','lower_arm'})
        self.assertGreater(hit['penetration_m'],.0001)
        self.assertIsNone(check.collision(e.last_action))
        for pair in check.pairs:
            for geom in pair:self.assertEqual(e.model.body_rootid[e.model.geom_bodyid[geom]],e.model.body_rootid[e.grip_body])
        np.testing.assert_array_equal(e.data.qpos,before)


if __name__=='__main__':unittest.main()
