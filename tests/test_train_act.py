"""Training split and normalization must not leak held-out demonstrations."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(importlib.util.find_spec('lerobot'), 'Use ACT environment')
class TrainingDataTests(unittest.TestCase):
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
