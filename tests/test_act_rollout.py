"""Check the training/inference boundary and actuator command interpretation."""
import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class ActionBoundaryTests(unittest.TestCase):
    def test_success_requires_one_ripe_fruit_return_and_stop(self):
        from evaluate_act import harvest_outcome
        env = SimpleNamespace(
            attached=np.array([False, True]), fruit_dofs=[0, 3], base_da=np.array([6, 7, 8]), tcp=0,
            meta={'basket_acceptance_half_xy_m': [.045, .034], 'ripe': [True, True]},
            data=SimpleNamespace(qvel=np.zeros(9), site_xpos=np.zeros((1, 3)),
                                 body=lambda _: SimpleNamespace(xmat=np.eye(3).ravel())),
            fruit_pos=lambda _: np.array([0., 0., .025]), basket_position=lambda: np.zeros(3),
            contacts=lambda _: False, state=lambda: np.zeros(6))
        with patch('mounted_basket.ready_point', return_value=np.zeros(3)):
            self.assertTrue(harvest_outcome(env, 0.)['physical_success_now'])
            env.meta['ripe'][0] = False
            self.assertFalse(harvest_outcome(env, 0.)['physical_success_now'])
            env.meta['ripe'][0] = True
            env.attached[1] = False
            self.assertFalse(harvest_outcome(env, 0.)['physical_success_now'])
            env.attached[1] = True
            env.data.site_xpos[0, 0] = .1
            self.assertFalse(harvest_outcome(env, 0.)['physical_success_now'])
            env.data.site_xpos[:] = 0
            env.data.qvel[6] = .1
            self.assertFalse(harvest_outcome(env, 0.)['physical_success_now'])

    def test_physical_units_and_limits(self):
        from evaluate_act import bounded_action
        limits = np.array([[-1., 1.]] * 6)
        requested = np.array([.1, .2, .3, .4, .5, -.1, .06, -.2])
        np.testing.assert_array_equal(bounded_action(requested, limits), requested)
        large = bounded_action([10, -10, 0, 0, 0, 0, 3, -3], limits)
        np.testing.assert_allclose(large, [1, -1, 0, 0, 0, 0, .15, -.5])
        with self.assertRaises(ValueError):
            bounded_action([float('nan')] * 8, limits)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires ACT environment')
class PreprocessingTests(unittest.TestCase):
    def test_diagnostic_scope_and_chunk_execution(self):
        from act_inference_worker import configure_rollout
        config = SimpleNamespace(chunk_size=50, n_action_steps=1, temporal_ensemble_coeff=None)
        with self.assertRaisesRegex(ValueError, 'Diagnostic'):
            configure_rollout(config, {'diagnostic': True})
        configure_rollout(config, {'diagnostic': True}, allow_diagnostic=True, n_action_steps=25)
        self.assertEqual(config.n_action_steps, 25)
        for invalid in (0, 51):
            with self.assertRaises(ValueError):
                configure_rollout(config, {'diagnostic': False}, n_action_steps=invalid)

    def test_training_normalization_and_action_inverse(self):
        import torch
        from act_inference_worker import Preprocessor
        stats = {'observation.state': {'mean': list(range(9)), 'std': [2.] * 9},
                 'action': {'mean': list(range(8)), 'std': [.5] * 8}}
        prep = Preprocessor(stats, 'cpu')
        # Constant red stays red through resizing, with training's fixed ImageNet statistics.
        red = np.zeros((360, 480, 3), np.uint8)
        red[:, :, 0] = 255
        images = {'harvest_close': red, 'wrist': red}
        batch = prep.prepare(np.arange(9) + 2, images)
        self.assertEqual(set(batch), {'observation.state', 'observation.images.harvest_close', 'observation.images.wrist'})
        torch.testing.assert_close(batch['observation.state'], torch.ones(1, 9))
        expected = torch.tensor([(1-.485)/.229, - .456/.224, -.406/.225])
        torch.testing.assert_close(batch['observation.images.wrist'][0, :, 100, 100], expected)
        self.assertEqual(tuple(batch['observation.images.wrist'].shape), (1, 3, 240, 320))
        np.testing.assert_allclose(prep.decode_action(torch.ones(1, 8)), np.arange(8) + .5)
        with self.assertRaises(ValueError):
            prep.prepare(np.arange(9), {**images, 'oracle': red})


if __name__ == '__main__':
    unittest.main()
