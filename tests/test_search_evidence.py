"""Direction guidance and missing-body feedback must never grant motion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from copy import deepcopy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
from local_targets import LocalTargets
from perception_recovery import MISSING_BLOCK
from search_guidance import row_guidance
from unlisted_fruit_search import inspect_claim_pixels, inspection_available
from vista import Session


class SearchEvidenceTests(unittest.TestCase):
    def test_direction_uses_measured_world_y_not_image_side(self):
        calibration = {'position': np.array([0., 0., 1.]), 'rotation': np.eye(3), 'fovy': 60.}
        hint = {'measured_stalk_m': [.34, 1.39, .28], 'frame_id': 'f10'}
        original = deepcopy(hint)
        result = row_guidance([0, 1.10, 0], calibration, [hint], [])
        self.assertEqual(result['previously_confirmed_ripe_but_unreachable_here'][0]['toward_measured_target_direction'], 1)
        self.assertEqual(hint, original)
        reverse = row_guidance([0, 1.60, 0], calibration, [hint], [])
        self.assertEqual(reverse['previously_confirmed_ripe_but_unreachable_here'][0]['toward_measured_target_direction'], -1)
        centered = row_guidance([0, 1.39, 0], calibration, [hint], [])
        self.assertIsNone(centered['previously_confirmed_ripe_but_unreachable_here'][0]['toward_measured_target_direction'])

    def test_guidance_preserves_history_and_has_no_target_when_empty(self):
        c = {'position': np.array([0., 0., 1.]), 'rotation': np.eye(3), 'fovy': 60.}
        history = [{'direction': -1}] * 4
        r = row_guidance([0, 0, 0], c, [], history)
        self.assertEqual(r['remaining_search_steps'], 0)
        self.assertEqual(r['previously_confirmed_ripe_but_unreachable_here'], [])
        self.assertEqual(history, [{'direction': -1}] * 4)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.state = {'tcp_world_m': [.19, -.04, .34], 'joints_rad': [0, 0, 0, 0, 0, 1.], 'base_x_y_yaw': [0, 0, 0]}
        env = SimpleNamespace(meta={'tilted_stalk_prior': True}, frame=10, drive=Mock())
        self.s = Session(env, SimpleNamespace(out=self.path), None, visual_proposals=True)
        self.s.local_targets.register([{'surface_m': [.3, 0, .25], 'maturity': 'uncertain',
                                       'visibility': 'partial', 'semantic_unverified': True}], [0, 0, 0], 'blocked')
        self.s.local_targets.selection_blocked_reason = MISSING_BLOCK
        self.s.local_targets.survey_review = {'missing_visible_ripe': True}
        self.rgb = Image.new('RGB', (480, 360), (19, 24, 18))
        self.rgb.save(self.path/'rgb.png')
        np.savez(self.path/'depth.npz', depth=np.ones((360, 480))*.5,
                 position=[0, 0, .7], rotation=np.eye(3), fovy=60.)
        self.s.frames = {'f10': {'harvest_close': self.path/'rgb.png'}}
        self.s.depth_frames = {('f10', 'harvest_close'): self.path/'depth.npz'}
        self.args = {'basis_frame_id': 'f10', 'bodies': [{'x': 310, 'y': 300, 'reason': 'claimed red berry'}]}

    def dispatch(self, args=None, age=0):
        with patch('vista.robot_state', return_value=self.state):
            return self.s.dispatch('inspect_unlisted', args or self.args, 'f10', self.state, age)

    def test_nonempty_queue_inspection_keeps_block_and_candidate_status(self):
        before = deepcopy(self.s.local_targets.candidates)
        result, images = self.dispatch()
        self.assertFalse(result['inspection']['checks'][0]['body_pixel_check']['passed'])
        self.assertTrue(images)
        self.assertEqual(before, self.s.local_targets.candidates)
        self.assertEqual(self.s.local_targets.selection_blocked_reason, MISSING_BLOCK)
        self.assertFalse(self.s.local_targets.can_search)
        self.s.e.drive.assert_not_called()
        self.assertEqual(self.s.station_attempts, 1)

    def test_only_confirmed_unreachable_stalk_is_remembered_for_search(self):
        self.s.local_targets = LocalTargets()
        self.s.local_targets.register([{'surface_m': [.3, .2, .25], 'maturity': 'ripe',
            'visibility': 'clear', 'body_pixel_verified': True}], [0, 0, 0], 'one ripe body')
        self.s.plan_parking = Mock(return_value={'current': {'reachable': False}})
        result = {'target_confirmed': True, 'locked_target_m': [.3, .2, .28],
                  'maturity_check': {'passed': True}, 'reachability': {'current': {'reachable': False}}}
        with patch.object(self.s, '_dispatch', return_value=(result, [])):
            self.s.dispatch('locate_stalk_proposal', {}, 'f10', self.state, 0)
        self.assertEqual(self.s.visual_search_hints[0]['measured_stalk_m'], [.3, .2, .28])
        self.assertEqual(self.s.local_targets.candidates[0]['status'], 'unavailable')
        self.s.e.drive.assert_not_called()
        self.s.local_targets = LocalTargets()
        self.s.local_targets.register([{'surface_m': [.3, .2, .25], 'maturity': 'ripe',
            'visibility': 'clear', 'body_pixel_verified': True}], [0, 0, 0], 'unconfirmed stalk')
        unconfirmed = {'target_confirmed': False, 'locked_target_m': [.3, .2, .28],
                      'maturity_check': {'passed': True}, 'reachability': {'current': {'reachable': False}}}
        with patch.object(self.s, '_dispatch', return_value=(unconfirmed, [])):
            self.s.dispatch('locate_stalk_proposal', {}, 'f10', self.state, 0)
        self.assertEqual(len(self.s.visual_search_hints), 1)

    def test_inspection_limits_and_stale_frames(self):
        with self.assertRaises(ValueError): self.dispatch({**self.args, 'basis_frame_id': 'old'})
        self.assertEqual(getattr(self.s.local_targets, 'unlisted_inspections', 0), 0)
        self.dispatch(); self.dispatch()
        self.assertFalse(inspection_available(self.s))
        with self.assertRaises(ValueError): self.dispatch()

    def test_motion_and_budget_guards(self):
        self.s.target_stage = 'pregrasp'
        with self.assertRaises(ValueError): self.dispatch()
        self.s.target_stage = None; self.s.grasp_attempted = True
        with self.assertRaises(ValueError): self.dispatch()
        self.s.grasp_attempted = False; self.s.station_attempts = 12
        result, _ = self.dispatch()
        self.assertEqual(result['status'], 'rejected')
        self.s.e.drive.assert_not_called()

    def test_invalid_pixel_and_depth_do_not_clear_claim(self):
        claims = [{'x': True, 'y': 10, 'reason': 'bad'}]
        with self.assertRaises(ValueError):
            inspect_claim_pixels(self.rgb, None, {}, [0, 0, 0], claims, Mock(), [])
        with patch('vista.depth_point', side_effect=ValueError('invalid depth')):
            result, _ = self.dispatch()
        self.assertIn('depth_error', result['inspection']['checks'][0])
        self.assertFalse(self.s.local_targets.can_search)


if __name__ == '__main__': unittest.main()
