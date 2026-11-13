"""Wait edits must preserve motion, event transitions and aligned source rows."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from training_preparation import trim_waits, validate_selection


def still(count=20, phase='model_wait'):
    return [dict(phase=phase, state=[0.] * 6, action=[0.] * 6,
                 tcp_position=[0.] * 3, base_pose=[0.] * 3, base_command=[0., 0.],
                 fruit_position=[0.] * 3, attached=True, bilateral_contact=True)
            for _ in range(count)]


class WaitTrimmingTests(unittest.TestCase):
    def test_stationary_wait_keeps_boundaries_and_input(self):
        rows = still(); before = copy.deepcopy(rows)
        indices, cuts = trim_waits(rows)
        self.assertEqual(indices, [0, 1, 2, 17, 18, 19])
        self.assertEqual(cuts[0]['removed_frames'], 14)
        self.assertEqual(rows, before)

    def test_short_wait_and_physical_holds_remain(self):
        for phase in ('gripper', 'grasp_contact_check', 'bend_pull_hold',
                      'terminal_settle', 'delivery_release', 'delivery_clear', 'search_park'):
            rows = still(20, phase)
            self.assertEqual(trim_waits(rows)[0], list(range(20)))
        self.assertEqual(trim_waits(still(6))[0], list(range(6)))

    def test_moving_joint_and_chassis_not_compressed(self):
        for field in ('state', 'tcp_position', 'base_pose', 'fruit_position', 'action'):
            rows = still()
            for i, row in enumerate(rows):
                row[field][0] = i * .005
            self.assertEqual(trim_waits(rows)[0], list(range(20)), field)
        rows = still()
        for row in rows:
            row['base_command'][0] = .1
        self.assertEqual(trim_waits(rows)[0], list(range(20)))

    def test_excursion_and_contact_transition_are_not_hidden(self):
        rows = still(40)
        rows[20]['state'][0] = .1
        indices, _ = trim_waits(rows)
        self.assertIn(19, indices); self.assertIn(20, indices); self.assertIn(21, indices)
        with self.assertRaises(ValueError):
            validate_selection(rows, [0, 39])
        rows = still(40)
        for row in rows[20:]:
            row['attached'] = False
        indices, _ = trim_waits(rows)
        self.assertIn(19, indices); self.assertIn(20, indices)
        with self.assertRaises(ValueError):
            validate_selection(rows, [0, 39])

    def test_cumulative_drift_not_just_framewise_delta(self):
        rows = still(30)
        for i, row in enumerate(rows):
            row['state'][0] = .0002 * i
        self.assertEqual(trim_waits(rows)[0], list(range(30)))

    def test_invalid_selections_rejected(self):
        for indices in ([0, 3, 2, 19], [0, 0, 19], [1, 19], [0, 18], [0, 30, 19]):
            with self.assertRaises(ValueError):
                validate_selection(still(), indices)
        with self.assertRaises(ValueError):
            validate_selection(still(20, 'gripper'), [0, 19])


if __name__ == '__main__':
    unittest.main()
