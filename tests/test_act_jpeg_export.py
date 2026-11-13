"""Temporary synthetic fixtures: these are NOT harvested training demonstrations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_act_dataset import export, jpeg_episode_data, verify_export
from recording_quality import audit_recording, phase_labels


CAMERAS = ['harvest_close', 'wrist']


def fixture(path, seed, count=12):
    path.mkdir()
    meta = dict(cameras=CAMERAS, image_size=[480, 360], control_dt=.05,
                joint_control_limits=[[-2, 2]] * 6, schema_version=3,
                basket_mount='chassis', status='finalized', fixture_only=True)
    (path / 'recording.json').write_text(json.dumps(meta))
    phases = ['model_wait', 'search_move', 'observation_adjust', 'approach', 'bend_pull', 'delivery_transport']
    rows = []
    rng = np.random.default_rng(seed)
    for i in range(count):
        stage, wait, recovery = phase_labels(phases[i % len(phases)])
        row = dict(frame=i, next_frame=i+1, timestamp=i*.05, next_timestamp=(i+1)*.05,
                   state=[i*.01]*6, action=[i*.01]*6, requested_action=[i*.01]*6,
                   base_pose=[0., seed*.01, 0.], base_command=[0., 0.],
                   phase=phases[i % len(phases)], stage=stage, is_model_wait=wait,
                   is_recovery=recovery, images={}, image_timestamps={})
        for camera in CAMERAS:
            rel = f'{camera}/{i:06d}.jpg'
            image = rng.integers(0, 256, (360, 480, 3), dtype=np.uint8)
            (path / camera).mkdir(exist_ok=True)
            Image.fromarray(image).save(path / rel, quality=92)
            row['images'][camera] = rel
            row['image_timestamps'][camera] = row['timestamp']
        rows.append(row)
    (path / 'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report = dict(seed=seed, frames=count, success=True, episode_completed=True, stop_reason='model_finish',
                  detached_fruit_count=1, ripe_fruit_in_basket=[7], single_fruit_delivered=True,
                  placement_executed=True, returned_to_ready=True, basket_mount='chassis',
                  delivery_base_parked=True, events=[{'event':'detached','fruit':7}],
                  fixture_only=True, scene_version='synthetic-export-test-not-a-harvest')
    (path / 'report.json').write_text(json.dumps(report))
    assert audit_recording(path)['eligible_for_training_export_review']
    return rows


class JpegExportTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('lerobot'), 'Use run-act-offline.sh for ACT integration')
    def test_prepared_export_keeps_source_alignment_and_jpeg_bytes(self):
        from training_preparation import prepare, load_prepared
        import datasets
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_cache = datasets.config.HF_DATASETS_CACHE
            datasets.config.HF_DATASETS_CACHE = root / 'cache'
            self.addCleanup(setattr, datasets.config, 'HF_DATASETS_CACHE', old_cache)
            source = root / 'raw'
            rows = fixture(source, 1, count=20)
            for i, row in enumerate(rows):
                row.update(tcp_position=[0., 0., 0.], fruit_position=None,
                           attached=None, bilateral_contact=False)
                row['state'] = row['action'] = row['requested_action'] = [max(0, i - 13) * .01] * 6
                if i < 14:
                    row['phase'] = 'model_wait'
                row['stage'], row['is_model_wait'], row['is_recovery'] = phase_labels(row['phase'])
            (source / 'trajectory.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            self.assertTrue(audit_recording(source)['eligible_for_training_export_review'])
            before = hashlib.sha256((source / 'trajectory.jsonl').read_bytes()).hexdigest()
            prepared = root / 'prepared'
            manifest = prepare([source], prepared)
            self.assertEqual(manifest['totals']['removed_frames'], 8)
            out = root / 'dataset'
            export(source, out, curation_manifest=prepared / 'manifest.json')
            result = json.loads((out / 'export_verification.json').read_text())
            self.assertTrue(result['passed'])
            self.assertEqual(result['frames'], 12)
            self.assertEqual(result['original_jpeg_verification']['images_checked'], 24)
            alignment = json.loads((out / 'source_alignment.json').read_text())
            self.assertEqual(alignment['rows'][3]['frame'], 11)
            self.assertEqual(alignment['rows'][3]['timestamp'], rows[11]['timestamp'])
            self.assertEqual(before, hashlib.sha256((source / 'trajectory.jsonl').read_bytes()).hexdigest())
            # Prepared rows cannot be silently edited after verification.
            frame_file = prepared / manifest['episodes'][0]['file']
            frame_file.write_text(frame_file.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'frame file changed'):
                load_prepared(prepared / 'manifest.json')

    def test_rejects_non_jpeg_and_external_image_before_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            rows = fixture(source, 1, count=1)
            good = jpeg_episode_data(source, rows, CAMERAS, 0)
            self.assertEqual(good['timestamp'], [0.])
            Image.new('RGB', (480, 360)).save(source / 'wrong.png')
            rows[0]['images']['wrist'] = 'wrong.png'
            with self.assertRaisesRegex(ValueError, 'JPEG'):
                jpeg_episode_data(source, rows, CAMERAS, 0)
            Image.new('RGB', (480, 360)).save(Path(tmp) / 'outside.jpg')
            rows[0]['images']['wrist'] = '../outside.jpg'
            with self.assertRaisesRegex(ValueError, 'outside'):
                jpeg_episode_data(source, rows, CAMERAS, 0)

    @unittest.skipUnless(importlib.util.find_spec('lerobot'), 'Use run-act-offline.sh for ACT integration')
    def test_full_export_bytes_boundaries_act_and_portable_package(self):
        # Keep all decoded Arrow caches in this test's temporary directory as well.
        import datasets
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from smoke_act_dataset import smoke
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_cache = datasets.config.HF_DATASETS_CACHE
            datasets.config.HF_DATASETS_CACHE = root / 'cache'
            self.addCleanup(setattr, datasets.config, 'HF_DATASETS_CACHE', old_cache)
            a, b = root / 'raw-a', root / 'raw-b'
            fixture(a, 1); fixture(b, 2)
            before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [a,b] for p in folder.rglob('*') if p.is_file()}
            out = root / 'dataset'
            export(a, out, additional_sources=[b], validation_seeds=[2])
            verification = json.loads((out / 'export_verification.json').read_text())
            self.assertEqual(verification['original_jpeg_verification']['images_checked'], 48)
            self.assertTrue(verification['all_timestamps_and_frame_indices_matched'])
            self.assertEqual(verification['episode_boundary_chunks_checked'], 20)
            self.assertFalse(list(out.rglob('*.png')))
            self.assertFalse(list(out.rglob('*.jpg')))
            self.assertFalse((out / 'EXPORT_INCOMPLETE.json').exists())
            for name, h in before.items():
                self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), h)
            # A short real ACT optimization and reload, with no downloaded backbone.
            smoke(out, root / 'smoke')
            result = json.loads((root / 'smoke/smoke_report.json').read_text())
            self.assertTrue(result['parameter_update_verified'])
            self.assertTrue(result['checkpoint_reload_matches'])
            # Moving the dataset and hiding original JPEGs must not break normal loading.
            out.rename(root / 'relocated'); out = root / 'relocated'
            a.rename(root / 'hidden-a'); b.rename(root / 'hidden-b')
            loaded = LeRobotDataset('local/strawberry_greenhouse_pilot_v1', root=out)
            self.assertEqual(tuple(loaded[0]['observation.images.wrist'].shape), (3, 360, 480))
            self.assertTrue(torch.isfinite(loaded[23]['action']).all())
            # Integrity check detects replacement of the source JPEG bytes.
            (root / 'hidden-a').rename(a); (root / 'hidden-b').rename(b)
            changed = a / 'wrist/000000.jpg'
            # Trailing bytes leave decoded pixels identical: only byte verification catches it.
            changed.write_bytes(changed.read_bytes() + b'changed-after-export')
            with self.assertRaisesRegex(ValueError, 'Embedded JPEG differs'):
                verify_export(out)
            self.assertTrue((out / 'EXPORT_INCOMPLETE.json').exists())


if __name__ == '__main__':
    unittest.main()
