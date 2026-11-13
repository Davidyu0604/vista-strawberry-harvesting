"""Summarize closed-loop logs using an offline-only demonstrated TCP reference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(run, reference):
    rows = [json.loads(line) for line in (run / 'trajectory.jsonl').read_text().splitlines()]
    report = json.loads((run / 'report.json').read_text())
    point = np.asarray(reference['points']['visual_grasp']['tcp_position'])
    distances = np.linalg.norm(np.asarray([r['tcp_position'] for r in rows]) - point, axis=1)
    nearest = int(np.argmin(distances))
    contact_runs, longest, total, first = {}, {}, {}, {}
    contacts_logged = all('bilateral_contact_fruits_after_step' in r for r in rows)
    for row in rows:
        contacts = row.get('bilateral_contact_fruits_after_step', [])
        for fruit in set(contact_runs) | set(contacts):
            contact_runs[fruit] = contact_runs.get(fruit, 0) + 1 if fruit in contacts else 0
            longest[fruit] = max(longest.get(fruit, 0), contact_runs[fruit])
            total[fruit] = total.get(fruit, 0) + int(fruit in contacts)
            if fruit in contacts:
                first.setdefault(fruit, row['frame'])
    parity = {}
    source = Path(reference['source'])
    for camera, digest in rows[0].get('image_sha256', {}).items():
        parity[camera] = digest == hashlib.sha256((source / camera / '000000.jpg').read_bytes()).hexdigest()
    delta = np.asarray([r['predicted_action'] for r in rows]) - np.asarray([r['action'] for r in rows])
    return dict(run=str(run), report=report, nearest_tcp_distance_m=float(distances[nearest]),
        nearest_frame=rows[nearest]['frame'], nearest_tcp_offset_m=(np.asarray(rows[nearest]['tcp_position']) - point).tolist(),
        last_tcp_distance_m=float(distances[-1]), last_gripper_rad=rows[-1]['state'][5],
        min_gripper_rad=float(min(r['state'][5] for r in rows)),
        joint_rate_limited_steps=int(np.count_nonzero(np.max(np.abs(delta[:, :6]), axis=1) > 1e-7)),
        max_joint_rate_limit_delta_rad=float(np.abs(delta[:, :6]).max()),
        contacts_logged=contacts_logged,
        bilateral_contacts={str(k): dict(first_frame=first.get(k), total_steps=total[k], longest_run=longest[k]) for k in total},
        initial_jpeg_bytes_match=parity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    result = dict(scope='Offline reference only. Distance and bilateral contact alone are not harvest success.',
                  runs=[summarize(run, reference) for run in args.run])
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
