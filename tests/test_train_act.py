"""Training split and normalization must not leak held-out demonstrations."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


@unittest.skipUnless(importlib.util.find_spec('lerobot'), 'Use ACT environment')
class TrainingDataTests(unittest.TestCase):
    def test_diagnostic_continuation_requires_explicit_same_episode(self):
        from train_act_overfit import initialization_steps
        self.assertEqual(initialization_steps({'diagnostic': False}, 46), 0)
        checkpoint = dict(diagnostic=True, diagnostic_kind='single_episode_finetune',
                          training_episode_seed=46, step=500)
        with self.assertRaisesRegex(ValueError, 'allow-diagnostic-init'):
            initialization_steps(checkpoint, 46)
        with self.assertRaisesRegex(ValueError, 'same training episode'):
            initialization_steps(checkpoint, 47, True)
        self.assertEqual(initialization_steps(checkpoint, 46, True), 500)
        checkpoint['total_single_episode_steps'] = 1500
        self.assertEqual(initialization_steps(checkpoint, 46, True), 1500)

    def test_overfit_selection_never_uses_validation_or_ambiguous_seed(self):
        from train_act_overfit import select_episode
        episode = dict(seed=46, episode_index=0, frames=12, dataset_from_index=5, dataset_to_index=17)
        manifest = dict(sources=[episode], split=dict(train=[0], validation=[]))
        self.assertEqual(select_episode(manifest, 46), episode)
        manifest['split'] = dict(train=[], validation=[0])
        with self.assertRaisesRegex(ValueError, 'held-out'):
            select_episode(manifest, 46)
        manifest['sources'].append(dict(episode))
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            select_episode(manifest, 46)

    def test_50_10_disjoint_and_complete(self):
        from train_act import training_indices
        episodes = [dict(episode_index=i, seed=i, dataset_from_index=2*i, dataset_to_index=2*i+2)
                    for i in range(60)]
        split = dict(train=list(range(50)), validation=list(range(50, 60)))
        self.assertEqual(training_indices(episodes, split), list(range(100)))
        episodes[59]['seed'] = 1
        with self.assertRaisesRegex(ValueError, 'seed'):
            training_indices(episodes, split)
        with self.assertRaisesRegex(ValueError, '50 training'):
            training_indices(episodes, dict(train=list(range(60)), validation=[]))

    def test_normalization_excludes_validation_outlier(self):
        from train_act import normalization
        values = np.array([[1., 3.], [3., 3.], [10000., -9000.]], dtype=np.float32)
        stats = normalization({'observation.state': values, 'action': values}, [0, 1])
        self.assertEqual(stats['action']['mean'], [2., 3.])
        np.testing.assert_allclose(stats['action']['std'], [1., .001])
        values[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            normalization({'observation.state': values, 'action': values}, [0, 1])


if __name__ == '__main__':
    unittest.main()
