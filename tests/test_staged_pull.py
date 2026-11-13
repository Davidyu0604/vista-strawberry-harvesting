
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from centered_grasp import CenteredGrasp, PullPlanError
from recording_quality import phase_labels


class StagedPullChecks(unittest.TestCase):
    def controller(self):
        c=CenteredGrasp.__new__(CenteredGrasp);c.carry_roll=-1.
        c.e=SimpleNamespace(dt=.05,last_action=np.zeros(6),tcp=0,grip_body=0,
            data=SimpleNamespace(site_xpos=np.array([[.3,0,.25]]),
                                 xmat=np.array([[0,0,-1,0,1,0,1,0,0]],dtype=float),warning=[]))
        c.self_collision=SimpleNamespace(path_collision=Mock(return_value=None))
        def tick(q,phase):
            c.e.last_action=np.array(q);c.e.data.site_xpos[0]=q[:3]
        c.tick=Mock(side_effect=tick)
        c.command=Mock(side_effect=lambda q,start=None:(np.array(q),np.asarray(q[:3])))
        return c

    def fit(self,e,point,grip,roll,pitch,start=None):
        return np.r_[point,pitch,roll,grip],0.

    def test_partial_withdrawal_then_combined_motion_preserves_endpoint(self):
        c=self.controller()
        with patch('centered_grasp.fixed_roll_ik',side_effect=self.fit):
            result=c.pull([-.035,0,0],np.deg2rad(-37),3,lambda:True)
        calls=c.tick.call_args_list
        labels=[a.args[1] for a in calls]
        first_rotate=labels.index('bend_pull_blend')
        self.assertTrue(all(a=='bend_pull_translate' for a in labels[:first_rotate]))
        self.assertTrue(all(a!='bend_pull_translate' for a in labels[first_rotate:]))
        self.assertEqual(len(calls),74)
        for call in calls[:first_rotate]:self.assertAlmostEqual(call.args[0][3],0.)
        self.assertAlmostEqual(calls[first_rotate-1].args[0][0],.286)
        # Bend while still retreating; no height offset or extra pull distance.
        self.assertGreater(calls[first_rotate].args[0][0],.265)
        for call in calls:np.testing.assert_allclose(call.args[0][1:3],[0,.25])
        np.testing.assert_allclose(calls[-1].args[0][:3],[.265,0,.25])
        self.assertAlmostEqual(calls[-1].args[0][3],np.deg2rad(-37))
        self.assertEqual(result['mode'],'withdraw_then_blend')

    def test_self_collision_rejects_before_motion(self):
        c=self.controller();c.self_collision.path_collision.side_effect=[None,{'hit':True}]
        with patch('centered_grasp.fixed_roll_ik',side_effect=self.fit),self.assertRaisesRegex(ValueError,'self-collide'):
            c.pull([-.035,0,0],-.6,3,lambda:True)
        c.tick.assert_not_called()

    def test_unreachable_world_x_uses_preflighted_gripper_retreat(self):
        c=self.controller()
        c.e.data.xmat[0]=[0,0,-.6,0,1,-.8,1,0,0]
        def fit(e,point,*args,**kwargs):
            q,error=self.fit(e,point,*args,**kwargs)
            return q, .01 if point[0]<.28 and abs(point[1])<1e-9 else error
        with patch('centered_grasp.fixed_roll_ik',side_effect=fit):
            result=c.pull([-.035,0,0],-.6,3,lambda:True)
        self.assertEqual(result['selected_path'],'gripper_outward')
        self.assertEqual(len(result['planning_trials']),2)
        np.testing.assert_allclose(result['displacement_m'],[-.021,-.028,0])
        np.testing.assert_allclose(result['requested_displacement_m'],[-.035,0,0])
        self.assertAlmostEqual(np.linalg.norm(result['displacement_m']),.035)
        self.assertTrue(all(call.args[0][1]<0 for call in c.tick.call_args_list))

    def test_all_plans_rejected_never_execute_or_change_command(self):
        c=self.controller();c.e.data.xmat[0]=[0,0,-.6,0,1,-.8,1,0,0]
        before=c.e.last_action.copy()
        with patch('centered_grasp.fixed_roll_ik',return_value=(np.zeros(6),.01)):
            with self.assertRaises(PullPlanError) as caught:c.pull([-.035,0,0],-.6,3,lambda:True)
        self.assertEqual(len(caught.exception.planning_trials),2)
        c.tick.assert_not_called();np.testing.assert_array_equal(c.e.last_action,before)

    def test_contact_loss_stops_after_four_ticks(self):
        c=self.controller()
        with patch('centered_grasp.fixed_roll_ik',side_effect=self.fit),self.assertRaisesRegex(ValueError,'contact lost'):
            c.pull([-.035,0,0],-.6,3,lambda:False)
        self.assertEqual(c.tick.call_count,4)

    def test_no_unbounded_wait_for_controller(self):
        c=self.controller();c.command.side_effect=lambda q,start=None:(start+1e-7,np.zeros(3))
        with patch('centered_grasp.fixed_roll_ik',side_effect=self.fit),self.assertRaisesRegex(ValueError,'motion budget'):
            c.pull([-.035,0,0],-.6,3,lambda:True)
        c.tick.assert_not_called()

    def test_observation_framing_requires_finite_visual_point_before_any_motion(self):
        c=self.controller();c.e.state=lambda:np.array([0,0,0,0,-.7,1.])
        for body in ([0,0,float('nan')],[0,0],[0,0,float('inf')]):
            with self.assertRaisesRegex(ValueError,'Finite measured body'):
                c.observation_plan({'tcp_world_m':[.18,0,.33],'roll':-.7},0,body_point=body)
        c.tick.assert_not_called()

    def test_staged_labels_are_training_detach_phase(self):
        for label in ('bend_pull_translate','bend_pull_blend','bend_pull_rotate','bend_pull_hold'):
            self.assertEqual(phase_labels(label),('detach',False,False))


if __name__=='__main__':unittest.main()
