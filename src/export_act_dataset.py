"""Export audited episodes to local LeRobot v3; no upload or model API."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from recording_quality import audit_recording, atomic_json

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
LABELS = ('frame', 'timestamp', 'phase', 'stage', 'is_model_wait', 'is_recovery')
TASK = 'Pick one ripe strawberry by its short stalk, place it in the fixed basket, and return.'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_split(seeds, validation_seeds):
    """Keep every episode of a seed on one side, including repeated seeds."""
    held = set(validation_seeds)
    if not held.issubset(set(seeds)):
        raise ValueError('Validation seeds must be present in the sources')
    train = [i for i, seed in enumerate(seeds) if seed not in held]
    validation = [i for i, seed in enumerate(seeds) if seed in held]
    if not train:
        raise ValueError('At least one training episode is required')
    return {'train': train, 'validation': validation,
            'train_seeds': sorted(set(seeds) - held), 'validation_seeds': sorted(held)}


def check_harvest(report):
    """Use offline acceptance fields, never model finish alone."""
    if not report.get('success') or report.get('detached_fruit_count') != 1:
        raise ValueError('Independent single-fruit harvest acceptance failed')
    fruit = report.get('ripe_fruit_in_basket', [])
    if len(fruit) != 1 or not report.get('single_fruit_delivered'):
        raise ValueError('Exactly one ripe fruit must be delivered')
    if not report.get('placement_executed') or not report.get('returned_to_ready'):
        raise ValueError('Placement and return are required')
    if report.get('basket_mount') == 'chassis' and not report.get('delivery_base_parked'):
        raise ValueError('Base must remain parked during delivery')
    events = [e for e in report.get('events', []) if e.get('event') == 'detached']
    if len(events) != 1 or events[0].get('fruit') != fruit[0]:
        raise ValueError('Detachment event and delivered ripe fruit disagree')


def jpeg_episode_data(source, rows, cameras, episode_index):
    """Public save_episode buffer: embed existing JPEGs without add_frame's PNG staging."""
    source = Path(source).resolve()
    data = dict(size=len(rows), task=[TASK] * len(rows), episode_index=episode_index,
                index=[], task_index=[], frame_index=list(range(len(rows))),
                timestamp=[i / 20 for i in range(len(rows))])
    data['observation.state'] = [np.asarray(r['state'] + r['base_pose'], np.float32) for r in rows]
    data['action'] = [np.asarray(r['action'] + r['base_command'], np.float32) for r in rows]
    for camera in cameras:
        paths = []
        for row in rows:
            path = (source / row['images'][camera]).resolve()
            if not path.is_relative_to(source):
                raise ValueError('Image path outside recording')
            with Image.open(path) as im:
                if im.format != 'JPEG' or im.mode != 'RGB' or im.size != (480, 360):
                    raise ValueError(f'Expected original 480x360 RGB JPEG: {path}')
            paths.append(str(path))
        data[f'observation.images.{camera}'] = paths
    return data


def verify_embedded_jpegs(out, records, cameras):
    """Read Parquet directly: every image must contain the exact source bytes."""
    import pyarrow.parquet as pq

    expected = iter((record['source'], row) for record in records for row in record['rows'])
    image_count = byte_count = 0
    columns = ['index'] + [f'observation.images.{c}' for c in cameras]
    frame_count = 0
    for path in sorted((out / 'data').rglob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=columns):
            for actual in batch.to_pylist():
                source, row = next(expected)
                if actual['index'] != frame_count:
                    raise ValueError('Parquet frame ordering changed')
                for camera in cameras:
                    encoded = actual[f'observation.images.{camera}']['bytes']
                    original = (source / row['images'][camera]).read_bytes()
                    if encoded != original or not original.startswith(b'\xff\xd8'):
                        raise ValueError(f'Embedded JPEG differs from source at frame {frame_count}: {camera}')
                    image_count += 1
                    byte_count += len(original)
                frame_count += 1
    if next(expected, None) is not None:
        raise ValueError('Missing exported frames')
    return dict(images_checked=image_count, source_image_bytes=byte_count,
                embedded_image_bytes=byte_count, all_jpeg_bytes_matched=True)


def verify_export(out):
    """Resume verification without re-encoding an already finalized export."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.set_num_threads(4)
    out = Path(out).resolve()
    manifest = json.loads((out / 'export_manifest.json').read_text())
    atomic_json(out / 'EXPORT_INCOMPLETE.json', {'note': 'Verification in progress; do not train until it passes.'})
    repo, cameras, episodes = manifest['repo_id'], manifest['cameras'], manifest['sources']
    records = []
    for episode in episodes:
        source = Path(episode['source'])
        rows = [json.loads(line) for line in (source / 'trajectory.jsonl').read_text().splitlines()]
        if 'source_row_indices' in episode:
            from training_preparation import POLICY, validate_selection
            if manifest.get('preparation', {}).get('policy') != POLICY:
                raise ValueError('Unknown preparation policy')
            assert len(rows) == episode['raw_frames']
            validate_selection(rows, episode['source_row_indices'])
            rows = [rows[i] for i in episode['source_row_indices']]
        assert len(rows) == episode['frames']
        records.append({'source': source, 'rows': rows, 'hashes': episode['source_files_sha256']})
    features = {'observation.state', 'action'} | {f'observation.images.{c}' for c in cameras}
    loaded = LeRobotDataset(repo, root=out, delta_timestamps={'action': [j / 20 for j in range(10)]})
    rows = [row for r in records for row in r['rows']]
    expected_action = np.asarray([r['action'] + r['base_command'] for r in rows], np.float32)
    expected_state = np.asarray([r['state'] + r['base_pose'] for r in rows], np.float32)
    assert len(loaded) == len(rows)
    assert loaded.num_episodes == len(episodes)
    expected_labels = [{'episode_index': e['episode_index'], 'dataset_index': e['dataset_from_index'] + i,
                        **{k: row[k] for k in LABELS}}
                       for e, r in zip(episodes, records) for i, row in enumerate(r['rows'])]
    assert json.loads((out / 'source_alignment.json').read_text())['rows'] == expected_labels
    numeric = loaded.hf_dataset.select_columns(['action', 'observation.state', 'timestamp',
                                                'frame_index', 'episode_index', 'index']).with_format('numpy')
    np.testing.assert_array_equal(np.asarray(numeric['action']), expected_action)
    np.testing.assert_array_equal(np.asarray(numeric['observation.state']), expected_state)
    np.testing.assert_array_equal(np.asarray(numeric['index']), np.arange(len(rows)))
    frame_indices = np.asarray(numeric['frame_index'])
    episode_indices = np.asarray(numeric['episode_index'])
    timestamps = np.asarray(numeric['timestamp'])
    for episode in episodes:
        start, end = episode['dataset_from_index'], episode['dataset_to_index']
        np.testing.assert_array_equal(frame_indices[start:end], np.arange(end - start))
        np.testing.assert_array_equal(episode_indices[start:end], episode['episode_index'])
        np.testing.assert_array_equal(timestamps[start:end], (np.arange(end - start) / 20).astype(timestamps.dtype))
    allowed = set(features) | {'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'}
    assert set(loaded.features) <= allowed, set(loaded.features) - allowed
    boundary_checks = 0
    for episode, record in zip(episodes, records):
        start, end = episode['dataset_from_index'], episode['dataset_to_index']
        for index in range(max(start, end - 10), end):
            sample = loaded[index]
            indices = np.minimum(np.arange(index, index + 10), end - 1)
            np.testing.assert_array_equal(sample['action'].numpy(), expected_action[indices])
            np.testing.assert_array_equal(sample['action_is_pad'].numpy(), np.arange(index, index + 10) >= end)
            boundary_checks += 1
        for local_index in (0, (end - start) // 2, end - start - 1):
            sample = loaded[start + local_index]
            assert int(sample['episode_index']) == episode['episode_index']
            for camera in cameras:
                with Image.open(record['source'] / record['rows'][local_index]['images'][camera]) as im:
                    expected = np.asarray(im.convert('RGB'))
                actual = sample[f'observation.images.{camera}'].permute(1, 2, 0).numpy()
                np.testing.assert_array_equal(np.rint(actual * 255).astype(np.uint8), expected)
        for name, old_hash in record['hashes'].items():
            assert digest(record['source'] / name) == old_hash, f'Raw source changed: {name}'
    jpeg_check = (verify_embedded_jpegs(out, records, cameras)
                  if manifest.get('image_storage') == 'original_jpeg_bytes_in_parquet' else None)
    atomic_json(out / 'export_verification.json', {
        'passed': True, 'all_state_action_values_matched': True, 'frames': len(rows),
        'image_readback_samples': len(episodes) * 3 * len(cameras), 'image_pixels_matched': True,
        'episode_boundary_chunks_checked': boundary_checks, 'cross_episode_action_leakage': False,
        'policy_feature_allowlist_checked': True, 'source_key_files_unchanged': True,
        'all_stage_labels_matched': True,
        'all_timestamps_and_frame_indices_matched': True,
        'original_jpeg_verification': jpeg_check,
    })
    (out / 'EXPORT_INCOMPLETE.json').unlink(missing_ok=True)
    print(f"Verified {len(episodes)} episodes, {len(rows)} frames", flush=True)


def export(source, out, additional_sources=(), validation_seeds=(), *, curation_manifest=None):
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.set_num_threads(4)
    sources = [Path(s).resolve() for s in (source, *additional_sources)]
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if len(set(sources)) != len(sources):
        raise ValueError('Duplicate source directory')
    prepared = {}
    preparation = None
    if curation_manifest is not None:
        from training_preparation import load_prepared
        prepared_manifest, prepared = load_prepared(curation_manifest)
        if set(prepared) != {str(s) for s in sources}:
            raise ValueError('Preparation and export sources must match exactly')
        preparation = {'manifest': str(Path(curation_manifest).resolve()),
                       'manifest_sha256': digest(Path(curation_manifest)),
                       'policy': prepared_manifest['policy']}
    records = []
    cameras = None
    metadata_signature = None
    for directory in sources:
        quality = audit_recording(directory, write_report=False)
        if not quality['eligible_for_training_export_review']:
            raise ValueError(f'Source failed raw quality gate: {directory}')
        report = json.loads((directory / 'report.json').read_text())
        check_harvest(report)
        meta = json.loads((directory / 'recording.json').read_text())
        signature = {k: meta.get(k) for k in ('cameras', 'image_size', 'control_dt', 'joint_control_limits', 'schema_version', 'basket_mount')}
        if metadata_signature is not None and signature != metadata_signature:
            raise ValueError('Incompatible camera, control or recording layouts')
        metadata_signature = signature
        cameras = meta['cameras']
        if meta['image_size'] != [480, 360] or not np.isclose(meta['control_dt'], .05):
            raise ValueError('This exporter requires 480x360 images at 20 Hz')
        rows = [json.loads(line) for line in (directory / 'trajectory.jsonl').read_text().splitlines()]
        raw_frames = len(rows)
        selection = prepared.get(str(directory))
        if selection is not None:
            rows = [rows[i] for i in selection['source_row_indices']]
        hashes = {name: digest(directory / name) for name in ('trajectory.jsonl', 'report.json', 'recording.json', 'quality.json')}
        source_hashes = {p.name: digest(p) for p in sorted((directory / 'source').glob('*.py'))}
        records.append({'source': directory, 'rows': rows, 'report': report, 'quality': quality,
                        'hashes': hashes, 'source_code_sha256': source_hashes,
                        'selection': selection, 'raw_frames': raw_frames})
        print(f"audited seed={report['seed']}: {len(rows)} frames, {quality['decoded_images']} images", flush=True)
    split = seed_split([r['report']['seed'] for r in records], validation_seeds)
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (9,), 'names': JOINTS + ['base_x_m', 'base_y_m', 'base_yaw_rad']},
        'action': {'dtype': 'float32', 'shape': (8,), 'names': JOINTS + ['base_linear_m_s', 'base_angular_rad_s']},
    }
    features.update({f'observation.images.{c}': {'dtype': 'image', 'shape': (360, 480, 3), 'names': ['height', 'width', 'channels']} for c in cameras})
    repo = 'local/strawberry_greenhouse_sample_v1' if len(records) == 1 else 'local/strawberry_greenhouse_pilot_v1'
    labels, episodes = [], []
    try:
        ds = LeRobotDataset.create(repo, 20, features, root=out, robot_type='so101_mobile_sim', use_videos=False)
        atomic_json(out / 'EXPORT_INCOMPLETE.json', {'note': 'Export in progress; do not train until verification passes.'})
        for episode_index, record in enumerate(records):
            start = len(labels)
            data = jpeg_episode_data(record['source'], record['rows'], cameras, episode_index)
            ds.save_episode(episode_data=data)
            for i, row in enumerate(record['rows']):
                labels.append({'episode_index': episode_index, 'dataset_index': len(labels), **{k: row[k] for k in LABELS}})
            print(f'episode {episode_index}: {len(record["rows"])} frames, original JPEGs embedded', flush=True)
            report = record['report']
            episodes.append({
                'episode_index': episode_index, 'seed': report['seed'], 'source': str(record['source']),
                'dataset_from_index': start, 'dataset_to_index': len(labels), 'frames': len(record['rows']),
                'scene_version': report.get('scene_version'), 'source_files_sha256': record['hashes'],
                'source_code_sha256': record['source_code_sha256'], 'quality': record['quality'],
                'search_steps': report.get('search_steps'), 'model_calls': report.get('model_calls'),
                'independent_harvest_acceptance_passed': True,
            })
            if record['selection'] is not None:
                episodes[-1].update(raw_frames=record['raw_frames'],
                                    source_row_indices=record['selection']['source_row_indices'])
        ds.finalize()
        atomic_json(out / 'source_alignment.json', {'episodes': episodes, 'rows': labels})
        manifest = {
            'source': str(sources[0]), 'sources': episodes, 'repo_id': repo, 'episodes': len(episodes),
            'frames': len(labels), 'fps': 20, 'cameras': list(cameras), 'state_dimension': 9, 'action_dimension': 8,
            'image_storage': 'original_jpeg_bytes_in_parquet',
            'image_reencoded': False,
            'joint_units': 'radians, including gripper', 'base_state_units': ['m', 'm', 'rad'],
            'base_action_units': ['m/s', 'rad/s'], 'split': split,
            'wait_policy': ('Stationary model/decision waits shortened; edited time at 20Hz. '
                            'Original frames/times in source_alignment.json; all physical holds retained.'
                            if preparation else 'All steps preserved; no gaps introduced. Labels in source_alignment.json.'),
            'preparation': preparation,
            'excluded_privileged_fields': ['target', 'fruit_position', 'attached', 'bilateral_contact'],
            'formal_training_dataset': False,
            'scope': 'Curated pipeline pilot; source controllers may differ. No stability-rate or learned-policy claim.',
        }
        atomic_json(out / 'export_manifest.json', manifest)
        verify_export(out)
    except BaseException:
        if out.exists():
            atomic_json(out / 'EXPORT_INCOMPLETE.json', {'sources': [str(s) for s in sources], 'note': 'Do not train; export or verification did not finish'})
        raise
    print(json.dumps({'out': str(out), 'episodes': len(episodes), 'frames': len(labels), 'split': split}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--verify-only', action='store_true', help='Verify the existing out directory; source is ignored')
    p.add_argument('--additional-source', type=Path, action='append', default=[])
    p.add_argument('--validation-seed', type=int, action='append', default=[])
    p.add_argument('--prepared', action='store_true', help='source is a prepared directory with manifest.json; export its selected frames')
    a = p.parse_args()
    if a.verify_only:verify_export(a.out)
    elif a.prepared:
        from training_preparation import load_prepared
        if a.additional_source:
            p.error('--prepared already supplies all sources; do not use --additional-source')
        manifest_path = a.source / 'manifest.json'
        _, prepared_sources = load_prepared(manifest_path)
        sources = list(prepared_sources)
        export(sources[0], a.out, sources[1:], a.validation_seed, curation_manifest=manifest_path)
    else:export(a.source, a.out, a.additional_source, a.validation_seed)
