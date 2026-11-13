"""Full-scene rear mount, two-view records and independent acceptance checks."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import PROJECT_ROOT, project_source

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from mounted_basket import BASKET_IN_CHASSIS, chassis_point, ready_point
from peduncle import PeduncleEnv
from run import Recorder
from vista import Session, evaluate, depth_point
from recording_quality import audit_recording, atomic_json


class MountedBasketChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = PeduncleEnv(7, single=False, render_quality='fast')

    @classmethod
    def tearDownClass(cls):
        cls.e.close()

    def test_mount_tracks_translation_and_yaw(self):
        e = self.e
        qpos = e.data.qpos.copy()
        try:
            e.data.qpos[e.base_qa] = [.11, -.2, .4]
            mujoco.mj_forward(e.model, e.data)
            self.assertEqual(e.model.body('basket').parentid, e.model.body('chassis').id)
            np.testing.assert_allclose(e.basket_position(), chassis_point(e, BASKET_IN_CHASSIS), atol=1e-10)
            for name in ('electronics_board', 'compute_module'):
                g=e.model.geom(name)
                self.assertLess(g.pos[2]+g.size[2], .09)
            self.assertGreater(len(e.fruit_bodies), 1)
        finally:
            e.data.qpos[:]=qpos
            mujoco.mj_forward(e.model,e.data)

    def test_wrist_camera_is_fixed_to_robot_with_recorded_profile(self):
        e=self.e;camera=e.model.camera('wrist')
        self.assertEqual(camera.bodyid,e.grip_body)
        np.testing.assert_allclose(camera.pos,[-.04,-.085,.015])
        self.assertAlmostEqual(float(camera.fovy[0]),80.)
        self.assertEqual(e.meta['wrist_camera_profile'],'workspace-side-v2')

    def test_two_views_and_export_gate(self):
        e=self.e
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'episode'
            rec=Recorder(e,out,images=True,video=False)
            client=SimpleNamespace()
            s=Session(e,rec,client)
            frame,_,message=s.observe()
            self.assertEqual(set(s.frames[frame]), {'harvest_close','wrist'})
            self.assertEqual(sum(c['type']=='image_url' for c in message['content']),2)
            with np.load(s.depth_frames[(frame,'harvest_close')]) as sample:
                measured=depth_point(sample['depth'],sample,288,152)
            s.local_targets.register([{'x':288,'y':152,'surface_m':measured.tolist(),
                                      'maturity':'ripe','visibility':'partial','reason':'test visible body'}],
                                     e.base_pose(),'fixture candidate for crop linkage only')
            info,_=s.active_candidate_crop(frame,{'base_x_y_yaw':e.base_pose().tolist()})
            self.assertEqual(info['candidate_key'],s.local_targets.active['candidate_key'])
            self.assertEqual(s.crops[info['crop_id']]['candidate_key'],info['candidate_key'])
            self.assertEqual(s.crops[info['crop_id']]['frame'],frame)
            self.assertEqual(set(s.frames[frame]),{'harvest_close','wrist'})
            for tool in client.tools:
                props=tool['function']['parameters']['properties']
                if 'camera' in props:self.assertEqual(set(props['camera']['enum']),{'harvest_close','wrist'})
            self.assertNotIn('地面筐',s.messages[0]['content'])
            rec.tick(e.last_action,'model_wait')
            rec.tick(e.last_action,'visual_pregrasp')
            report=rec.finish()
            q=audit_recording(out)
            self.assertTrue(q['integrity_passed'],q)
            self.assertTrue(q['camera_layout_valid'])
            self.assertEqual(q['decoded_images'],4)
            self.assertFalse(q['eligible_for_training_export_review'])
            # Gate mechanics only: this synthetic report is never a harvest demo.
            report.update(success=True,episode_completed=True,stop_reason='model_finish')
            atomic_json(out/'report.json',report)
            self.assertTrue(audit_recording(out)['eligible_for_training_export_review'])
            meta=json.loads((out/'recording.json').read_text())
            meta['cameras']=['wrist'];atomic_json(out/'recording.json',meta)
            self.assertFalse(audit_recording(out)['eligible_for_training_export_review'])

    def test_acceptance_requires_single_detached_fruit_release_and_ready(self):
        e=self.e
        attached=e.attached.copy();qpos=e.data.qpos.copy();qvel=e.data.qvel.copy()
        try:
            e.data.qpos[e.base_qa]=[.12,.05,.35]
            mujoco.mj_forward(e.model,e.data)
            # Acceptance-only fixture, never used for policy input or demonstrations.
            fruit=int(np.flatnonzero(e.meta['ripe'])[0])
            e.attached[fruit]=False
            e.data.qvel[:]=0
            position=chassis_point(e,BASKET_IN_CHASSIS+[0,0,.025])
            rows=[{'phase':p,'base_pose':e.base_pose().tolist()} for p in ('delivery_release','delivery_stow')]
            with patch.object(e,'fruit_pos',side_effect=lambda i:position if i==fruit else np.array([2.,2.,2.])), patch.object(e,'contacts',return_value=False), patch('mounted_basket.ready_point',return_value=e.data.site_xpos[e.tcp].copy()):
                self.assertTrue(evaluate(e,rows)['success'])
                self.assertFalse(evaluate(e,rows[1:])['success'])
                e.attached[(fruit+1)%len(e.attached)]=False
                self.assertFalse(evaluate(e,rows)['success'])
                e.attached[:]=attached;e.attached[fruit]=False
                with patch('mounted_basket.ready_point',return_value=np.array([2.,2.,2.])):
                    self.assertFalse(evaluate(e,rows)['success'])
        finally:
            e.attached[:]=attached;e.data.qpos[:]=qpos;e.data.qvel[:]=qvel
            mujoco.mj_forward(e.model,e.data)


if __name__=='__main__':unittest.main()
