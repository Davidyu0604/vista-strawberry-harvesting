"""Negative mechanical checks. Positive roundtrips live in verify_peduncle.py."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
from peduncle import PeduncleEnv
from vista import validate_actions, Session, robot_state, jaw_contact_state, verify_projected_point
from run import Recorder


class Checks(unittest.TestCase):
    def test_grasp_requires_budget_for_delivery(self):
        e=PeduncleEnv(render_quality='fast')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                s=Session(e,Recorder(e,Path(tmp)/'episode',False,False),None,max_calls=16)
                s.target_stage='grasp';s.calls=14
                frame=e.frame
                result=s.execute([{'action':'gripper','opening':-.15,'seconds':2.5}])
                self.assertEqual(result[0]['status'],'rejected')
                self.assertIn('budget',result[0]['reason'])
                self.assertEqual(e.frame,frame);self.assertFalse(s.grasp_attempted)
                s.calls=13
                result=s.execute([{'action':'gripper','opening':-.15,'seconds':.1}])
                self.assertEqual(result[0]['status'],'executed')
        finally:e.close()

    def test_full_greenhouse_idle_stability_and_control_period(self):
        e=PeduncleEnv(seed=0,single=False,render_quality='fast')
        try:
            start=e.data.xpos[e.fruit_bodies].copy();peak=0.
            for _ in range(40):
                e.step(e.last_action)
                peak=max(peak,float(np.linalg.norm(e.data.xpos[e.fruit_bodies]-start,axis=1).max()))
            self.assertLess(peak,.0001)
            self.assertAlmostEqual(e.data.time,2.,places=7)
            self.assertTrue(e.attached.all())
            self.assertFalse(any(w.number for w in e.data.warning))
        finally:e.close()

    def test_diverse_scene_without_ripe_target_can_record(self):
        e=PeduncleEnv(seed=1,single=False,render_quality='fast')
        try:
            e.meta['ripe']=[False]*len(e.fruit_bodies)
            e.target=e.select_target()
            self.assertIsNone(e.target)
            with tempfile.TemporaryDirectory() as tmp:
                rec=Recorder(e,Path(tmp)/'episode',False,False)
                rec.tick(e.last_action,'no_ripe_target')
                self.assertIsNone(rec.rows[0]['fruit_position'])
                report=rec.finish()
                self.assertFalse(report['success'])
                self.assertIsNone(report['terminal_fruit_position'])
        finally:e.close()

    def test_depth_projection_rejects_occlusion_and_outside(self):
        calibration={'rotation':np.eye(3),'position':np.zeros(3),'fovy':60.}
        depth=np.full((100,100),.5)
        self.assertTrue(verify_projected_point(np.array([0.,0.,-.5]),depth,calibration)['consistent'])
        self.assertFalse(verify_projected_point(np.array([0.,0.,-.6]),depth,calibration)['consistent'])
        self.assertFalse(verify_projected_point(np.array([0.,0.,.5]),depth,calibration)['consistent'])
        self.assertFalse(verify_projected_point(np.array([2.,0.,-.5]),depth,calibration)['consistent'])

    def test_close_camera_selection_is_confirmed_without_pixel_guess(self):
        e=PeduncleEnv(seed=0,render_quality='fast')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                s=Session(e,Recorder(e,Path(tmp)/'episode',False,False),None)
                for _ in range(40):s.tick(e.last_action,'idle')
                frame,state,_=s.observe()
                # Model-selected pixel from the real failed episode; no object site lookup.
                result,images=s.dispatch('locate_pixel',{'frame_id':frame,'camera':'harvest_close','x':238,'y':157},frame,state,0)
                self.assertTrue(result['target_confirmed'])
                self.assertTrue(result['cross_view_checks']['top']['consistent'])
                self.assertTrue(result['cross_view_checks']['wrist']['consistent'])
                self.assertEqual(sum(i['type']=='image_url' for i in images),2)
                self.assertLess(s.locked_target[2],result['surface_world_m'][2]-.005)
                for stage in ('pregrasp','grasp'):
                    receipt,_=s.dispatch('approach_target',{'basis_frame_id':frame,'stage':stage},frame,robot_state(e),0)
                    self.assertLess(receipt['tcp_error_m'],.006)
                s.execute([{'action':'gripper','opening':-.15,'seconds':2.5}])
                for _ in range(40):s.tick(e.last_action,'hold')
                s.execute([{'action':'bend_pull','dx':-.035,'dy':0,'dz':0,'pitch_deg':-37,'seconds':3}])
                self.assertFalse(e.attached[0]);self.assertTrue(e.contacts(0))
                s.deliver_to_basket()
                self.assertTrue(e.success);self.assertLess(abs(e.base_pose()[1]),.012)
        finally:e.close()

    def test_blocked_high_approach_withdraws_and_clears_target(self):
        e=PeduncleEnv(seed=0,render_quality='fast')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                s=Session(e,Recorder(e,Path(tmp)/'episode',False,False),None)
                for _ in range(40):s.tick(e.last_action,'idle')
                # Deliberately bypass fitting to reproduce the historical branch collision.
                s.locked_target=np.array([.27822918,-.14139772,.26311232]);s.target_confirmed=True
                # Isolate mechanical recovery: visual gate has separate real-image tests.
                s.maturity_evidence={'passed':True,'target_m':s.locked_target.tolist(),'axis_world':[0,0,1]}
                s.check_maturity=lambda *args:(s.maturity_evidence,None)
                s.dispatch('approach_target',{'basis_frame_id':'f','stage':'pregrasp'},'f',robot_state(e),0)
                with self.assertRaisesRegex(ValueError,'withdrew open'):
                    s.dispatch('approach_target',{'basis_frame_id':'f','stage':'grasp'},'f',robot_state(e),0)
                self.assertIsNone(s.target_stage);self.assertIsNone(s.locked_target)
                self.assertGreater(e.state()[5],.9)
                self.assertFalse(any(w.number for w in e.data.warning))
        finally:e.close()

    def test_logged_visual_target_approach_and_grasp(self):
        e=PeduncleEnv(seed=0,render_quality='fast')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                session=Session(e,Recorder(e,Path(tmp)/'episode',False,False),None)
                for _ in range(40):session.tick(e.last_action,'idle')
                move={'action':'move_tcp','dx':.06,'dy':.02,'dz':0,'seconds':1}
                self.assertEqual(session.execute([move])[0]['status'],'rejected')
                self.assertEqual(session.execute([{'action':'gripper','opening':-.15,'seconds':1}])[0]['status'],'rejected')
                # Actual first depth measurement from the failed model episode, not a scene site.
                point=np.array([.279272583151081,-.14231427157736953,.2580945979903726])
                session.confirm_target(point,robot_state(e),'top')
                with self.assertRaises(ValueError):
                    session.dispatch('approach_target',{'basis_frame_id':'test','stage':'pregrasp'},'test',robot_state(e),0)
                session.confirm_target(point+[.001,0,0],robot_state(e),'harvest_close')
                self.assertTrue(session.target_confirmed)
                # This fixture tests legacy mechanical execution, bypassing perception.
                session.maturity_evidence={'passed':True,'target_m':session.locked_target.tolist(),'axis_world':[0,0,1]}
                session.check_maturity=lambda *args:(session.maturity_evidence,None)
                for bad in ([1.037,-.3225,.0773],[.3975,-.0035,.1319],point+[.03,0,0]):
                    with self.assertRaises(ValueError):session.lock_target(np.array(bad),robot_state(e))
                    np.testing.assert_array_equal(session.locked_target,point)
                for stage in ('pregrasp','grasp'):
                    result,_=session.dispatch('approach_target',{'basis_frame_id':'test','stage':stage},'test',robot_state(e),0)
                    self.assertLess(result['tcp_error_m'],.008)
                result=session.execute([{'action':'gripper','opening':-.15,'seconds':2.5}])
                self.assertEqual(result[0]['status'],'executed')
                self.assertTrue(e.contacts(0));self.assertTrue(e.attached[0])
                self.assertTrue(jaw_contact_state(e)['both'])
                self.assertFalse(any(w.number for w in e.data.warning))
        finally:e.close()

    def test_visual_executor_preserves_bend_pitch(self):
        e=PeduncleEnv(render_quality='fast')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rec=Recorder(e,Path(tmp)/'episode',False,False)
                session=Session(e,rec,None)
                session.grasp_attempted=True  # This test exercises post-grasp motion.
                # Isolate pitch persistence from the separately tested tactile guard.
                with patch('vista.jaw_contact_state',return_value={'both':True}):
                    result=session.execute([{'action':'bend_pull','dx':-.005,'dy':0,'dz':0,'pitch_deg':-10,'seconds':1}])
                self.assertEqual(result[0]['status'],'executed')
                self.assertAlmostEqual(session.approach_pitch,np.deg2rad(-10))
                result=session.execute([{'action':'move_tcp','dx':0,'dy':0,'dz':.005,'seconds':1}])
                self.assertEqual(result[0]['status'],'executed')
                self.assertAlmostEqual(session.approach_pitch,np.deg2rad(-10))
                self.assertIn('果柄',session.messages[0]['content'])
        finally:e.close()

    def test_wrong_first_target_and_empty_pull_are_blocked(self):
        e=PeduncleEnv(render_quality='fast')
        try:
            with tempfile.TemporaryDirectory() as tmp:
                s=Session(e,Recorder(e,Path(tmp)/'episode',False,False),None)
                state=robot_state(e)
                wrong=np.array([.3363493686308072,-.13647682757421908,.23927857977547928])
                s.confirm_target(wrong,state,'wrist')
                s.confirm_target(wrong,state,'wrist')
                self.assertFalse(s.target_confirmed)
                with self.assertRaises(ValueError):s.confirm_target(np.array([.27927,-.14231,.25809]),state,'top')
                with self.assertRaises(ValueError):s.dispatch('approach_target',{'basis_frame_id':'f','stage':'pregrasp'},'f',state,0)
                s.dispatch('clear_target',{},'f',state,0)
                self.assertIsNone(s.locked_target)
                s.grasp_attempted=True
                frame=e.frame
                self.assertFalse(jaw_contact_state(e)['both'])
                r=s.execute([{'action':'bend_pull','dx':-.035,'dy':0,'dz':0,'pitch_deg':-37,'seconds':3}])
                self.assertEqual(r[0]['status'],'rejected');self.assertEqual(e.frame,frame)
                s.observe()
                self.assertIn('harvest_close',next(iter(s.frames.values())))
        finally:e.close()

    def test_open_jaw_motion_cannot_detach(self):
        e=PeduncleEnv(render_quality='fast')
        try:
            self.assertEqual(e.fruit_geom_lookup[e.fruit_geoms[0]],-1)
            self.assertEqual(e.fruit_geom_lookup[e.stalk_geoms[0]],0)
            p=e.data.site_xpos[e.grasp_sites[0]].copy()
            for target,pitch in [(p-[.055,0,0],0),(p,0),(p-[.035,0,0],-.65),(p-[.07,0,-.04],-.35)]:
                q,err=e.ik(target,1,approach_pitch=pitch)
                self.assertLess(err,.006)
                for _ in range(55):e.step(q)
            self.assertTrue(e.attached[0]);self.assertEqual(e.events,[])
            self.assertFalse(any(w.number for w in e.data.warning))
        finally:e.close()

    def test_bend_angle_is_bounded(self):
        a={'basis_frame_id':'f1','expectation':'bend and pull','actions':[
            {'action':'bend_pull','dx':-.035,'dy':0,'dz':0,'pitch_deg':-37,'seconds':3}]}
        validate_actions(a,'f1')
        a['actions'][0]['pitch_deg']=-90
        with self.assertRaises(ValueError):validate_actions(a,'f1')


if __name__=='__main__':unittest.main()
