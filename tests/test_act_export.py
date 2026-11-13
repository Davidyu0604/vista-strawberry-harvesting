"""Guard seed isolation and physical acceptance before training export."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from export_act_dataset import check_harvest, seed_split
from recording_quality import audit_recording


class ExportChecks(unittest.TestCase):
    def test_repeated_seeds_stay_on_one_side(self):
        split = seed_split([1, 169, 2, 169], [169])
        self.assertEqual(split['train'], [0, 2])
        self.assertEqual(split['validation'], [1, 3])
        self.assertFalse(set(split['train_seeds']) & set(split['validation_seeds']))

    def test_unknown_or_all_validation_seeds_rejected(self):
        for held in ([3], [1, 2]):
            with self.assertRaises(ValueError):
                seed_split([1, 2], held)

    def test_episode_completed_is_not_harvest_acceptance(self):
        with self.assertRaises(ValueError):
            check_harvest({'episode_completed': True, 'success': True})

    def test_detached_and_delivered_must_match_and_be_single(self):
        report = dict(success=True, detached_fruit_count=1, ripe_fruit_in_basket=[7],
                      single_fruit_delivered=True, placement_executed=True, returned_to_ready=True,
                      basket_mount='chassis', delivery_base_parked=True,
                      events=[{'event': 'detached', 'fruit': 7}])
        check_harvest(report)
        for change in ({'detached_fruit_count': 2}, {'ripe_fruit_in_basket': [8]},
                       {'returned_to_ready': False}, {'delivery_base_parked': False},
                       {'events': [{'event': 'detached', 'fruit': 7}, {'event': 'detached', 'fruit': 8}]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_harvest({**report, **change})

    def test_read_only_audit_preserves_existing_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'quality.json'
            p.write_text(json.dumps({'old_report': True}))
            before = p.read_bytes()
            result = audit_recording(tmp, write_report=False)
            self.assertFalse(result['integrity_passed'])
            self.assertEqual(p.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
