"""Real rendered RGB synchronization and interrupted recording tests, no API."""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from peduncle import PeduncleEnv
from run import Recorder
from recording_quality import audit_recording, atomic_json


class RecordingChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.e=PeduncleEnv(seed=0,render_quality='fast')
        self.out=Path(self.tmp.name)/'episode'
        self.rec=Recorder(self.e,self.out,images=True,video=False)

    def tearDown(self):
        self.e.close();self.tmp.cleanup()

    def test_sync_labels_and_actual_limited_command(self):
        goal=self.e.last_action.copy();goal[0]+=.1
        self.rec.tick(goal,'model_wait')
        self.rec.tick(goal,'approach_retreat')
        rows=[json.loads(line) for line in (self.out/'trajectory.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),2)  # Persisted before finish.
        self.assertEqual(set(rows[0]['images']),{'top','wrist','harvest_close'})
        self.assertEqual(set(rows[0]['image_timestamps'].values()),{rows[0]['timestamp']})
        self.assertAlmostEqual(rows[0]['requested_action'][0]-rows[0]['action'][0],.06)
        self.assertTrue(rows[0]['is_model_wait']);self.assertTrue(rows[1]['is_recovery'])
        self.rec.finish()
        q=audit_recording(self.out)
        self.assertTrue(q['integrity_passed'],q)
        self.assertEqual(q['decoded_images'],6)
        self.assertFalse(q['eligible_for_training_export_review'])

    def test_interruption_preserves_committed_prefix(self):
        self.rec.tick(self.e.last_action,'visual_pregrasp')
        with patch.object(self.e,'step',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):self.rec.tick(self.e.last_action,'visual_grasp')
        self.assertEqual(len((self.out/'trajectory.jsonl').read_text().splitlines()),1)
        self.assertTrue((self.out/'pending-step.json').exists())
        q=audit_recording(self.out)
        self.assertFalse(q['integrity_passed']);self.assertFalse(q['episode_completed'])
        self.assertIn('Uncommitted step',str(q['errors']))
        self.assertFalse(q['eligible_for_training_export_review'])

    def test_corrupt_image_missing_camera_and_truncated_row(self):
        self.rec.tick(self.e.last_action,'gripper')
        row=self.rec.rows[0]
        (self.out/row['images']['wrist']).write_bytes(b'not a JPEG')
        self.assertFalse(audit_recording(self.out)['integrity_passed'])
        row['images'].pop('wrist')
        (self.out/'trajectory.jsonl').write_text(json.dumps(row)+'\n{"frame":')
        q=audit_recording(self.out)
        self.assertIn('Truncated',str(q['errors']))
        self.assertIn('missing/extra camera',str(q['errors']))

    def test_timestamp_mismatch_and_interrupted_success_not_eligible(self):
        self.rec.tick(self.e.last_action,'gripper')
        report=self.rec.finish()
        report.update(success=True,episode_completed=True,stop_reason='interrupted',error='KeyboardInterrupt')
        atomic_json(self.out/'report.json',report)
        self.assertFalse(audit_recording(self.out)['eligible_for_training_export_review'])
        row=self.rec.rows[0];row['image_timestamps']['top']+=.05
        (self.out/'trajectory.jsonl').write_text(json.dumps(row)+'\n')
        q=audit_recording(self.out)
        self.assertFalse(q['integrity_passed']);self.assertIn('timestamp mismatch',str(q['errors']))

    def test_behaviour_rejection_blocks_export_without_rewriting_physical_result(self):
        self.rec.tick(self.e.last_action,'gripper')
        report=self.rec.finish()
        report.update(success=True,episode_completed=True,stop_reason='model_finish')
        atomic_json(self.out/'report.json',report)
        self.assertTrue(audit_recording(self.out)['eligible_for_training_export_review'])
        for review in ('{"training_approved":false,"reason":"unverified search evidence"}', '{broken'):
            (self.out/'behaviour-review.json').write_text(review)
            quality=audit_recording(self.out)
            self.assertTrue(quality['integrity_passed'])
            self.assertTrue(quality['harvest_success'])
            self.assertTrue(quality['behaviour_review_blocks_training'])
            self.assertFalse(quality['eligible_for_training_export_review'])
        self.assertTrue(json.loads((self.out/'report.json').read_text())['success'])

if __name__=='__main__':unittest.main()
