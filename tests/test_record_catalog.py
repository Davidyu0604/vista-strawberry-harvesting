"""Seed browsing must preserve originals, verdicts, and unrelated files."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from record_catalog import refresh, register_episode


class SeedCatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'datasets').mkdir()

    def episode(self, path, seed=7, modern=True, approved=None):
        path = self.root / path
        path.mkdir(parents=True)
        (path / 'trajectory.jsonl').write_text('{"frame":0}\n')
        (path / 'report.json').write_text(json.dumps(dict(seed=seed, success=True,
            episode_completed=True, scene_version='test-v1')))
        (path / 'quality.json').write_text('{"integrity_passed":true}')
        if modern:
            (path / 'recording.json').write_text('{}')
        if approved is not None:
            (path / 'behaviour-review.json').write_text(json.dumps({'training_approved': approved}))
        os.utime(path / 'trajectory.jsonl', (1700000000, 1700000000))
        register_episode(path, self.root)
        return path

    def test_same_seed_same_time_runs_stay_distinct_and_refresh_is_idempotent(self):
        one = self.episode('reports/batch-one/seed-7')
        two = self.episode('reports/batch-two/seed-7', approved=False)
        before = {str(f): f.read_bytes() for p in [one, two] for f in p.iterdir()}
        rows = refresh(self.root)
        aliases = [self.root / 'recordings' / r['seed_view_path'] for r in rows]
        self.assertEqual(len(set(aliases)), 2)
        self.assertEqual({p.resolve() for p in aliases}, {one, two})
        self.assertTrue(all(p.parent.name == 'seed-007' for p in aliases))
        self.assertEqual(rows, refresh(self.root))
        self.assertEqual(before, {str(f): f.read_bytes() for p in [one, two] for f in p.iterdir()})
        self.assertFalse(next(r for r in rows if r['source'] == str(two))['behaviour_training_approved'])
        self.assertIn('行为未通过', (self.root / 'recordings/index.html').read_text())

    def test_legacy_and_untrusted_seed_are_not_mixed(self):
        self.episode('reports/old', modern=False)
        self.episode('reports/unknown', seed='../outside')
        rows = refresh(self.root)
        old = next(r for r in rows if r['group'] == 'legacy')
        self.assertNotIn('seed_view_path', old)
        current = next(r for r in rows if r['group'] == 'episodes')
        self.assertTrue(current['seed_view_path'].startswith('by-seed/seed-unknown/'))
        self.assertFalse((self.root / 'recordings/outside').exists())

    def test_removed_registration_cleans_only_managed_alias(self):
        source = self.episode('reports/run')
        row = refresh(self.root)[0]
        alias = self.root / 'recordings' / row['seed_view_path']
        note = alias.parent / 'my-notes.txt'
        note.write_text('keep')
        (self.root / 'recordings' / row['path']).unlink()
        refresh(self.root)
        self.assertFalse(alias.is_symlink())
        self.assertTrue(source.exists())
        self.assertEqual(note.read_text(), 'keep')

    def test_empty_catalog_removes_latest_links_and_does_not_advertise_them(self):
        self.episode('reports/run')
        row = refresh(self.root)[0]
        home = self.root / 'recordings'
        self.assertTrue((home / 'latest-success').is_symlink())
        (home / row['path']).unlink()
        self.assertEqual(refresh(self.root), [])
        self.assertFalse((home / 'latest').is_symlink())
        self.assertFalse((home / 'latest-success').is_symlink())
        self.assertNotIn('href="latest', (home / 'index.html').read_text())
        self.assertEqual(json.loads((home / 'by-seed/manifest.json').read_text())['episode_count'], 0)


if __name__ == '__main__':
    unittest.main()
