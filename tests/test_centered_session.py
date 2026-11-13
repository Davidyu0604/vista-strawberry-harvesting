
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from vista import Session, MotionSafetyStop
from centered_grasp import CenteredGrasp, TrackingError, PullPlanError
from recording_quality import phase_labels


class CenteredSessionChecks(unittest.TestCase):
    def session(self):
        s=Session.__new__(Session)
        s.e=SimpleNamespace(last_action=np.zeros(6),frame=0,dt=.05,
            data=SimpleNamespace(warning=[]),drive=Mock(),meta={'harvest_mechanism':True})
        s.centered=Mock();s.tick=Mock();s.target_stage='grasp';s.grasp_attempted=False
        s.calls=0;s.max_calls=32
        return s

    def test_short_close_rejected_before_any_motion(self):
        s=self.session();r=s.execute([{'action':'gripper','opening':-.15,'seconds':2.5}])
        self.assertEqual(r[0]['status'],'rejected');self.assertFalse(s.grasp_attempted)
        s.tick.assert_not_called();s.centered.close.assert_not_called()

    def test_centered_close_runs_once_and_uses_controller(self):
        s=self.session()
        with patch('vista.robot_state',return_value={}),patch('vista.jaw_contact_state',return_value={'both':True}):
            r=s.execute([{'action':'gripper','opening':-.15,'seconds':3}])
        self.assertEqual(r[0]['status'],'executed');self.assertTrue(s.grasp_attempted)
        s.centered.close.assert_called_once();s.tick.assert_not_called()

    def test_failed_centered_close_prevents_followup_motion(self):
        s=self.session();s.centered.close.side_effect=ValueError('tracking')
        r=s.execute([{'action':'gripper','opening':-.15,'seconds':3}])
        self.assertEqual(r[0]['status'],'rejected');self.assertTrue(s.preclose_recovery_failed)
        r=s.execute([{'action':'bend_pull','dx':-.035,'dy':0,'dz':0,'pitch_deg':-37,'seconds':3}])
        self.assertEqual(r[0]['status'],'rejected');s.tick.assert_not_called()

    def test_observation_restore_requires_open_jaw_and_return_label(self):
        c=CenteredGrasp.__new__(CenteredGrasp);c.observation_roll=0.
        c.e=SimpleNamespace(state=lambda:np.zeros(6));c.goto=Mock()
        with self.assertRaisesRegex(ValueError,'opening'):c.restore_observation()
        c.goto.assert_not_called()
        self.assertEqual(phase_labels('delivery_restore_observation'),('return',False,False))

    def test_pregrasp_recovery_reverses_traversed_commands_not_failed_destination(self):
        c=CenteredGrasp.__new__(CenteredGrasp)
        c.approach_commands=[np.full(6,x) for x in (.1,.2,.3)]
        c.e=SimpleNamespace(last_action=np.full(6,.4));c.pre_q=np.full(6,9.)
        c.goto=Mock(side_effect=lambda q,phase:setattr(c.e,'last_action',q.copy()))
        c.settle=Mock();c.retreat()
        for call,wanted in zip(c.goto.call_args_list[:3],(.3,.2,.1)):
            np.testing.assert_array_equal(call.args[0],np.full(6,wanted))
        self.assertEqual(c.goto.call_args_list[-1].args[0][-1],1.)
        self.assertEqual(c.approach_commands,[]);self.assertIsNone(c.point)

    def test_tracking_threshold_is_not_relaxed(self):
        c=CenteredGrasp.__new__(CenteredGrasp);c.blocked=0
        c.e=SimpleNamespace(tcp=0,data=SimpleNamespace(site_xpos=np.zeros((1,3))))
        for _ in range(2):c.check_tracking(np.array([.009,0,0]))
        with self.assertRaisesRegex(ValueError,'three ticks'):c.check_tracking(np.array([.009,0,0]))
        self.assertAlmostEqual(c.last_tracking['error_m'],.009)

    def test_failed_recovery_stops_run_before_next_wait_or_model_call(self):
        s=self.session();s.messages=[];s.max_calls=32;s.max_wait=90;s.wait_ticks=0
        s.observe=Mock(return_value=('f',{}, {'role':'user','content':[]}));s.log=Mock()
        message={'role':'assistant','tool_calls':[{'id':'one','function':{'name':'approach_target','arguments':'{}'}}]}
        s.client=SimpleNamespace(call=Mock(return_value=(message,{})))
        s.preclose_recovery_failed=False
        def tick(*args,**kwargs):self.assertFalse(s.preclose_recovery_failed)
        s.tick=Mock(side_effect=tick)
        def failed(*args):
            s.preclose_recovery_failed=True
            return {'status':'rejected','reason':'retreat blocked'},[]
        s.dispatch=Mock(side_effect=failed)
        with self.assertRaises(MotionSafetyStop):s.run()
        s.client.call.assert_called_once();self.assertEqual(s.calls,1)

    def test_recovery_error_preserves_both_failure_reasons(self):
        s=self.session();s.target_stage=None;s.target_confirmed=True;s.locked_target=np.ones(3)
        s.maturity_evidence={'axis_world':[0,0,1]};s.centered.last_tracking={'error_m':.012}
        def blocked(*args):s.e.frame+=3;raise ValueError('initial tracking failure')
        s.centered.approach.side_effect=blocked;s.centered.retreat.side_effect=ValueError('reverse tracking failure')
        with self.assertRaises(ValueError) as caught:s.execute_approach('pregrasp')
        self.assertTrue(s.preclose_recovery_failed)
        self.assertEqual(caught.exception.approach_trace['approach_error'],'initial tracking failure')
        self.assertEqual(caught.exception.approach_trace['retreat_error'],'reverse tracking failure')

    def test_delivery_tracking_stops_before_next_settle_or_model_call(self):
        s=self.session();s.messages=[];s.max_wait=90;s.wait_ticks=0
        s.observe=Mock(return_value=('f',{}, {'role':'user','content':[]}));s.log=Mock()
        message={'role':'assistant','tool_calls':[{'id':'one','function':{'name':'deliver_to_basket','arguments':'{}'}}]}
        s.client=SimpleNamespace(call=Mock(return_value=(message,{})))
        failed=False
        def tick(*args,**kwargs):self.assertFalse(failed)
        def dispatch(*args):
            nonlocal failed
            failed=True
            exc=TrackingError('Tracking error at carry');exc.phase='delivery_carry'
            exc.tracking={'error_m':.00984,'consecutive_ticks':3}
            raise exc
        s.tick=Mock(side_effect=tick);s.dispatch=Mock(side_effect=dispatch)
        with self.assertRaises(MotionSafetyStop):s.run()
        s.client.call.assert_called_once();self.assertEqual(s.calls,1)
        self.assertEqual(s.motion_stop_trace['phase'],'delivery_carry')
        self.assertEqual(s.log.call_args.kwargs['result']['status'],'stopped')

    def test_wait_tracking_stop_is_saved_without_model_request(self):
        s=self.session();s.messages=[];s.max_wait=90;s.wait_ticks=0;s.log=Mock()
        exc=TrackingError('Tracking error while waiting');exc.tracking={'error_m':.009}
        s.tick=Mock(side_effect=exc);s.client=SimpleNamespace(call=Mock())
        with self.assertRaises(MotionSafetyStop):s.run()
        s.client.call.assert_not_called()
        self.assertEqual(s.tick.call_count,1)

    def test_partial_pull_failure_latches_stop_and_skips_remaining_actions(self):
        s=self.session();s.grasp_attempted=True;s.centered.ik.return_value=(np.zeros(6),0.)
        s.check_contact_hold=Mock(return_value={'stable':True})
        s.e.data.site_xpos=np.zeros((1,3));s.e.tcp=0;s.approach_pitch=0.
        def tick(*args):
            s.e.frame+=1
            raise TrackingError('tracking during pull')
        s.tick=Mock(side_effect=tick)
        s.centered.pull.side_effect=lambda *args:s.tick(None,'bend_pull_translate')
        with patch('vista.jaw_contact_state',return_value={'both':True}):
            results=s.execute([{'action':'bend_pull','dx':-.035,'dy':0,'dz':0,'pitch_deg':-37,'seconds':3},
                               {'action':'gripper','opening':1,'seconds':1}])
        self.assertEqual(results[1]['status'],'not_executed')
        self.assertEqual(s.motion_stop_trace['action'],'bend_pull')
        self.assertEqual(s.tick.call_count,1)

    def test_contact_hold_does_not_turn_pull_preflight_rejection_into_motion_stop(self):
        s=self.session();s.grasp_attempted=True;s.e.meta['tilted_stalk_prior']=True
        s.e.data.site_xpos=np.zeros((1,3));s.e.tcp=0
        def hold():
            s.e.frame+=10
            return {'stable':True}
        s.check_contact_hold=Mock(side_effect=hold)
        exc=PullPlanError('No bounded pull plan');exc.planning_trials=[{'rejected':'unreachable'}]
        s.centered.pull.side_effect=exc
        with patch('vista.jaw_contact_state',return_value={'both':True}):
            result=s.execute([dict(action='bend_pull',dx=-.035,dy=0,dz=0,pitch_deg=-37,seconds=3)])
        self.assertTrue(result[0]['recoverable']);self.assertFalse(result[0]['pull_motion_executed'])
        self.assertFalse(getattr(s,'motion_stop_trace',None));s.tick.assert_not_called()


if __name__=='__main__':unittest.main()
