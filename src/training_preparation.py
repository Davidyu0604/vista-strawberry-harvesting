"""Offline training timeline preparation; raw recordings remain untouched.

Only stationary model/decision waits are shortened. This is an edited training
timeline, not a physically continuous simulator replay or a training approval.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from recording_quality import atomic_json, audit_recording

POLICY = {
    'version': 'stationary-waits-v1',
    'phases': ['model_wait', 'decision_settle'],
    'boundary_frames': 3,
    'joint_range_rad': .001,
    'tcp_range_m': .0005,
    'base_range_m_rad': .0001,
    'fruit_range_m': .0005,
    'command_range': 1e-8,
    'fps': 20,
}
RAW_FILES = ('trajectory.jsonl', 'report.json', 'recording.json', 'quality.json',
             'run-config.json', 'vista-events.jsonl', 'behaviour-review.json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _vector(row):
    """Offline diagnostics only; these extra fields never enter policy features."""
    fruit = row.get('fruit_position')
    values = (row['state'] + row['tcp_position'] + row['base_pose']
              + (fruit if fruit is not None else [0., 0., 0.]) + row['action'])
    vector = np.asarray(values, dtype=float)
    if vector.shape != (21,) or not np.isfinite(vector).all():
        raise ValueError('Invalid wait stability measurements')
    signature = (fruit is not None, row.get('attached'), row.get('bilateral_contact'))
    limits = np.asarray([POLICY['joint_range_rad']] * 6 + [POLICY['tcp_range_m']] * 3
                        + [POLICY['base_range_m_rad']] * 3 + [POLICY['fruit_range_m']] * 3
                        + [POLICY['command_range']] * 6)
    return vector, limits, signature


def trim_waits(rows):
    """Remove interiors of bounded stationary plateaus, preserving both ends.

    Range tests apply to every intermediate frame, not just equal endpoints:
    an excursion that later returns must not disappear. Keep at least 150 ms
    at each end. All non-wait phases, including physical holds, remain intact.
    """
    if not rows:
        raise ValueError('Empty trajectory')
    keep = np.ones(len(rows), dtype=bool)
    cuts = []
    edge = POLICY['boundary_frames']
    i = 0
    while i < len(rows):
        row = rows[i]
        if row['phase'] not in POLICY['phases'] or np.any(np.asarray(row['base_command']) != 0):
            i += 1
            continue
        low, limits, signature = _vector(row)
        high = low.copy()
        j = i + 1
        while j < len(rows) and rows[j]['phase'] in POLICY['phases']:
            if np.any(np.asarray(rows[j]['base_command']) != 0):
                break
            value, _, other_signature = _vector(rows[j])
            next_low, next_high = np.minimum(low, value), np.maximum(high, value)
            if other_signature != signature or np.any(next_high - next_low > limits):
                break
            low, high = next_low, next_high
            j += 1
        if j - i > 2 * edge:
            first, end = i + edge, j - edge
            keep[first:end] = False
            cuts.append({'removed_start': first, 'removed_end_exclusive': end,
                         'left_source_row': first - 1, 'right_source_row': end,
                         'removed_frames': end - first,
                         'phases': dict(Counter(r['phase'] for r in rows[first:end]))})
        i = j
    indices = np.flatnonzero(keep).tolist()
    validate_selection(rows, indices)
    return indices, cuts


def validate_selection(rows, indices):
    if not indices or indices[0] != 0 or indices[-1] != len(rows) - 1:
        raise ValueError('Episode endpoints must be retained')
    if any(type(i) is not int for i in indices) or indices != sorted(set(indices)):
        raise ValueError('Source indices must be unique increasing integers')
    if any(i < 0 or i >= len(rows) for i in indices):
        raise ValueError('Source index out of range')
    selected = set(indices)
    if any(i not in selected and row['phase'] not in POLICY['phases']
           for i, row in enumerate(rows)):
        raise ValueError('Cannot remove physical action/hold frames')
    for left, right in zip(indices, indices[1:]):
        if right == left + 1:
            continue
        span = rows[left:right + 1]
        if any(r['phase'] not in POLICY['phases'] or np.any(np.asarray(r['base_command']) != 0)
               for r in span):
            raise ValueError('Cut includes movement or a physical hold')
        measured = [_vector(r) for r in span]
        values = np.asarray([v[0] for v in measured])
        if any(v[2] != measured[0][2] for v in measured):
            raise ValueError('Cut hides a contact or attachment transition')
        if np.any(np.ptp(values, axis=0) > measured[0][1]):
            raise ValueError('Cut exceeds stationary range')
        if np.max(np.abs(np.asarray(rows[right]['action']) - rows[left]['action'])) > .0400001:
            raise ValueError('Cut violates joint command slew')


def load_prepared(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('status') != 'verified' or manifest.get('policy') != POLICY:
        raise ValueError('Unsupported or incomplete preparation')
    if not manifest.get('episodes'):
        raise ValueError('Preparation contains no episodes')
    records = {}
    for entry in manifest['episodes']:
        prepared_file = (path.parent / entry['file']).resolve()
        if not prepared_file.is_relative_to(path.parent) or sha256(prepared_file) != entry['file_sha256']:
            raise ValueError('Prepared frame file changed or outside preparation directory')
        source = Path(entry['source']).resolve()
        if str(source) in records:
            raise ValueError('Duplicate prepared source')
        for name, digest in entry['source_files_sha256'].items():
            if sha256(source / name) != digest:
                raise ValueError(f'Prepared source changed: {source / name}')
        rows = [json.loads(line) for line in (source / 'trajectory.jsonl').read_text().splitlines()]
        indices = entry['source_row_indices']
        validate_selection(rows, indices)
        if len(rows) != entry['raw_frames'] or len(indices) != entry['frames']:
            raise ValueError('Prepared frame count mismatch')
        records[str(source)] = entry
    return manifest, records


def prepare(sources, out):
    from export_act_dataset import check_harvest

    sources = [Path(p).resolve() for p in sources]
    if not sources or len(sources) != len(set(sources)):
        raise ValueError('Sources must be nonempty and unique')
    out = Path(out).resolve()
    if any(out.is_relative_to(source) for source in sources):
        raise ValueError('Prepared output must be separate from raw source directories')
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'incomplete', 'policy': POLICY, 'episodes': [],
                'formal_training_approved': False, 'raw_recordings_modified': False,
                'images_copied': False, 'image_root': 'per-episode source directory',
                'timeline': '20Hz edited training time; original source time retained separately',
                'scope': 'Stationary wait trimming only; final behavior review and split pending.'}
    atomic_json(out / 'manifest.json', manifest)
    totals = Counter()
    for episode_index, source in enumerate(sources):
        hashes = {n: sha256(source / n) for n in RAW_FILES if (source / n).exists()}
        quality = audit_recording(source, write_report=False)
        if not quality['eligible_for_training_export_review']:
            raise ValueError(f'Source not eligible: {source}')
        report = json.loads((source / 'report.json').read_text())
        meta = json.loads((source / 'recording.json').read_text())
        check_harvest(dict(report, basket_mount=meta['basket_mount']))
        if not np.isclose(meta['control_dt'], .05):
            raise ValueError('Expected 20Hz raw data')
        rows = [json.loads(line) for line in (source / 'trajectory.jsonl').read_text().splitlines()]
        indices, cuts = trim_waits(rows)
        name = f'episode-{episode_index:03d}-seed-{report["seed"]:03d}.jsonl'
        with (out / name).open('w') as f:
            for index, original in enumerate(indices):
                row = rows[original]
                edited = {'frame_index': index, 'timestamp': index / 20,
                          'source_row_index': original, 'source_frame': row['frame'],
                          'source_timestamp': row['timestamp'],
                          'observation.state': row['state'] + row['base_pose'],
                          'action': row['action'] + row['base_command'],
                          'images': row['images'], 'phase': row['phase'], 'stage': row['stage'],
                          'is_model_wait': row['is_model_wait'], 'is_recovery': row['is_recovery']}
                f.write(json.dumps(edited, ensure_ascii=False) + '\n')
        # Read back all rows: images/state/action must refer to the same source tick.
        edited_rows = [json.loads(line) for line in (out / name).read_text().splitlines()]
        for i, (edited, original) in enumerate(zip(edited_rows, indices)):
            raw = rows[original]
            assert edited['timestamp'] == i / 20 and edited['source_frame'] == raw['frame']
            assert edited['source_timestamp'] == raw['timestamp']
            assert edited['images'] == raw['images']
            assert edited['observation.state'] == raw['state'] + raw['base_pose']
            assert edited['action'] == raw['action'] + raw['base_command']
        assert all(sha256(source / n) == h for n, h in hashes.items())
        entry = {'episode_index': episode_index, 'seed': report['seed'], 'source': str(source),
                 'scene_version': report.get('scene_version'), 'file': name,
                 'file_sha256': sha256(out / name), 'source_files_sha256': hashes,
                 'raw_frames': len(rows), 'frames': len(indices),
                 'removed_frames': len(rows) - len(indices), 'source_row_indices': indices,
                 'cuts': cuts, 'raw_phase_counts': dict(Counter(r['phase'] for r in rows)),
                 'phase_counts': dict(Counter(rows[i]['phase'] for i in indices)),
                 'quality': quality, 'independent_harvest_passed': True,
                 'all_non_wait_frames_retained': True, 'state_action_image_alignment_passed': True}
        manifest['episodes'].append(entry)
        totals.update(raw_frames=len(rows), frames=len(indices), removed_frames=len(rows) - len(indices),
                      decoded_images=quality['decoded_images'], cuts=len(cuts))
        atomic_json(out / 'manifest.json', manifest)
        print(f'[{episode_index+1}/{len(sources)}] seed={report["seed"]}: {len(rows)} -> {len(indices)} frames', flush=True)
    manifest.update(status='verified', totals=dict(totals), episode_count=len(sources))
    atomic_json(out / 'manifest.json', manifest)
    try:
        load_prepared(out / 'manifest.json')
    except BaseException:
        manifest['status'] = 'verification_failed'
        atomic_json(out / 'manifest.json', manifest)
        raise
    print(json.dumps(manifest['totals']), flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources-json', type=Path, required=True, help='JSON list of approved-for-review raw source paths')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.sources_json.read_text()), args.out)
