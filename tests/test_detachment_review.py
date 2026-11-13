
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from detachment_review import review_separation,gripper_context_region
import numpy as np
from vista import Session


def views(a='visible_gap',b='visible_gap'):
    return [{'camera':camera,'held_fruit_visible':True,'separation':separation,'reason':'Observed stalk end and plant connection'}
            for camera,separation in zip(('harvest_close','wrist'),(a,b))]


class ReviewChecks(unittest.TestCase):
    def test_context_crop_uses_robot_projection_and_clips_to_image(self):
        cal={'position':[0,0,1],'rotation':np.eye(3),'fovy':60}
        box=gripper_context_region([0,0,.3],cal,(480,360))
        x,y,w,h=box
        self.assertLessEqual(x,240);self.assertGreaterEqual(x+w,240)
        self.assertLessEqual(y,180);self.assertGreaterEqual(y+h,180)
        self.assertGreaterEqual(x,0);self.assertLessEqual(x+w,480)
        self.assertIsNone(gripper_context_region([0,0,1.1],cal,(480,360)))

    def test_unknown_or_conflicting_view_blocks_transport(self):
        for a,b in (('occluded','occluded'),('visible_gap','connected'),('connected','visible_gap')):
            self.assertFalse(review_separation(views(a,b),('harvest_close','wrist'))['allowed'])
        self.assertTrue(review_separation(views('visible_gap','occluded'),('harvest_close','wrist'))['allowed'])
        lost=views();lost[1]['held_fruit_visible']=False
        self.assertFalse(review_separation(lost,('harvest_close','wrist'))['allowed'])

    def test_each_camera_required_once(self):
        for v in ([views()[0]], [views()[0],views()[0]]):
            with self.assertRaises(ValueError):review_separation(v,('harvest_close','wrist'))

    def session(self):
        s=Session.__new__(Session);s.centered=Mock();s.max_wait=90
        s.policy_cameras=('harvest_close','wrist');s.pull_reference_frame='f0000010'
        s.frames={'f0000010':{},'f0000080':{}}
        s.pull_completed={'end_frame':70};s.detachment_confirmation=None
        s.e=SimpleNamespace();s.grasp_attempted=True
        s.deliver_to_basket=Mock(return_value={'status':'executed'})
        return s

    def state(self):
        return {'joints_rad':[0]*6,'tcp_world_m':[0,0,.3],'base_x_y_yaw':[0,0,0]}

    def args(self):
        return {'basis_frame_id':'f0000080','reference_frame_id':'f0000010','views':views()}

    def test_cannot_deliver_from_tactile_or_motion_completion_only(self):
        s=self.session();state=self.state()
        with patch('vista.robot_state',return_value=state),self.assertRaisesRegex(ValueError,'not confirmed'):
            s._dispatch('deliver_to_basket',{'basis_frame_id':'f0000080'},'f0000080',state,0)
        s.deliver_to_basket.assert_not_called()

    def test_review_then_deliver_and_changed_robot_pose_rejected(self):
        s=self.session();state=self.state()
        with patch('vista.robot_state',return_value=state),patch('vista.jaw_contact_state',return_value={'both':True}):
            result,_=s._dispatch('confirm_detachment',self.args(),'f0000080',state,0)
            self.assertTrue(result['allowed'])
            s._dispatch('deliver_to_basket',{'basis_frame_id':'f0000080'},'f0000080',state,0)
        s.deliver_to_basket.assert_called_once()
        moved={**state,'tcp_world_m':[.02,0,.3]}
        with patch('vista.robot_state',return_value=moved),self.assertRaisesRegex(ValueError,'stale'):
            s._dispatch('deliver_to_basket',{'basis_frame_id':'f0000080'},'f0000080',moved,0)
        self.assertEqual(s.deliver_to_basket.call_count,1)

    def test_new_negative_review_clears_previous_confirmation(self):
        s=self.session();s.detachment_confirmation={'allowed':True};state=self.state()
        args={**self.args(),'views':views('occluded','occluded')}
        with patch('vista.robot_state',return_value=state),patch('vista.jaw_contact_state',return_value={'both':True}):
            result,_=s._dispatch('confirm_detachment',args,'f0000080',state,0)
        self.assertFalse(result['allowed']);self.assertIsNone(s.detachment_confirmation)

    def test_pre_pull_or_wrong_reference_frames_rejected(self):
        s=self.session();state=self.state()
        for args,frame in (({**self.args(),'reference_frame_id':'unknown'},'f0000080'),
                           ({**self.args(),'basis_frame_id':'f0000060'},'f0000060')):
            with self.assertRaises(ValueError):s._dispatch('confirm_detachment',args,frame,state,0)

    def test_tool_only_advertised_in_centered_mode(self):
        # Avoid constructing physics: non-centered case must hide the tool.
        with tempfile.TemporaryDirectory() as tmp:
            client=SimpleNamespace()
            Session(SimpleNamespace(meta={}),SimpleNamespace(out=Path(tmp)),client)
            self.assertNotIn('confirm_detachment',[t['function']['name'] for t in client.tools])

    def test_mixed_pull_actions_rejected_before_motion(self):
        s=self.session();s.pull_completed=None;s.execute=Mock();state=self.state()
        args={'basis_frame_id':'f0000080','expectation':'detach',
              'actions':[{'action':'bend_pull','dx':-.035,'dy':0,'dz':0,'pitch_deg':-37,'seconds':3},
                         {'action':'move_tcp','dx':-.03,'dy':0,'dz':.03,'seconds':1}]}
        with patch('vista.robot_state',return_value=state),self.assertRaisesRegex(ValueError,'alone'):
            s._dispatch('play',args,'f0000080',state,0)
        s.execute.assert_not_called()

    def test_repeated_pull_rejected_before_replacing_reference(self):
        s=self.session();s.execute=Mock();state=self.state()
        args={'basis_frame_id':'f0000080','expectation':'detach',
              'actions':[{'action':'bend_pull','dx':-.035,'dy':0,'dz':0,'pitch_deg':-37,'seconds':3}]}
        with patch('vista.robot_state',return_value=state),self.assertRaisesRegex(ValueError,'already completed'):
            s._dispatch('play',args,'f0000080',state,0)
        s.execute.assert_not_called();self.assertEqual(s.pull_reference_frame,'f0000010')

    def test_lost_contact_revokes_visual_confirmation_during_wait(self):
        s=self.session();s.detachment_confirmation={'allowed':True}
        s.e.dt=.05;s.viewer=None;s.rec=SimpleNamespace(tick=Mock())
        s.centered.command.return_value=([0]*6,[0,0,.3])
        with patch('vista.jaw_contact_state',return_value={'both':False}):
            s.tick([0]*6,'model_wait')
        self.assertIsNone(s.detachment_confirmation)

    def test_observation_contains_labeled_before_after_camera_context(self):
        state=self.state()
        cal={'position':np.array([0,0,1.]),'rotation':np.eye(3),'fovy':60.}
        e=SimpleNamespace(meta={'policy_cameras':['harvest_close','wrist']},frame=10,
            render=lambda *args:np.zeros((360,480,3),dtype=np.uint8),
            depth_view=lambda *args:(np.ones((360,480)),cal),data=SimpleNamespace(time=.5))
        with tempfile.TemporaryDirectory() as tmp,patch('vista.robot_state',return_value=state):
            s=Session(e,SimpleNamespace(out=Path(tmp)),None)
            before,_,_=s.observe()
            s.pull_reference_frame=before;s.pull_reference_state=state
            s.pull_completed={'end_frame':70};s.grasp_attempted=True;s.target_stage='grasp'
            e.frame=80
            after,_,message=s.observe()
        labels=[item['text'] for item in message['content'] if item['type']=='text']
        self.assertEqual(sum(item['type']=='image_url' for item in message['content']),6)
        for camera in ('harvest_close','wrist'):
            self.assertTrue(any(f'BEFORE pull: {before}, {camera}' in text for text in labels))
            self.assertTrue(any(f'AFTER pull: {after}, {camera}' in text for text in labels))


if __name__=='__main__':unittest.main()
