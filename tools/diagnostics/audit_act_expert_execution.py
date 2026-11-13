"""Offline execution audit of retained expert commands; never a policy rollout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / 'export_manifest.json').read_text())
    episode, = [e for e in manifest['sources'] if e['seed'] == args.seed]
    source = Path(episode['source'])
    rows = [json.loads(line) for line in (source / 'trajectory.jsonl').read_text().splitlines()]
    required = ['env.py', 'peduncle.py', 'scene.py', 'natural_assets.py', 'mobile_scene.py',
                'mounted_basket.py', 'chassis.py', 'scene_details.py']
    candidates = [ROOT / 'src', source / 'source', source.parent / 'source',
                  source.parent.parent / 'source']
    physics_source = next((directory for directory in candidates if all(
        (directory / name).is_file() and
        hashlib.sha256((directory / name).read_bytes()).hexdigest() == episode['source_code_sha256'][name]
        for name in required)), None)
    if physics_source is None:
        raise ValueError('No hash-matched physics source snapshot')
    from evaluate_act import harvest_outcome
    sys.path.insert(0, str(physics_source))
    import scene
    import natural_assets
    from peduncle import PeduncleEnv
    results = {}
    with tempfile.TemporaryDirectory(prefix='expert-audit-') as directory:
        scratch = Path(directory)
        (scratch / 'assets/natural').mkdir(parents=True)
        (scratch / 'assets/so101').symlink_to(ROOT / 'assets/so101', target_is_directory=True)
        scene.ROOT = scratch
        natural_assets.ASSETS = scratch / 'assets/natural'
        for name, indices in [('raw', range(len(rows))), ('retained', episode['source_row_indices'])]:
            env = PeduncleEnv(args.seed, single=False, fixed_base=False, mobile_layout=True,
                              mobile_layout_level='moderate', render_quality='fast', initial_arm_pose='search_stow')
            initial_roll = float(env.state()[4])
            max_state_error = max_tcp_error = 0.
            contact_count = longest_contact = contact_run = 0
            first_contact = None
            for step, index in enumerate(indices):
                row = rows[index]
                max_state_error = max(max_state_error, float(np.max(np.abs(env.state() - row['state']))))
                max_tcp_error = max(max_tcp_error, float(np.linalg.norm(env.data.site_xpos[env.tcp] - row['tcp_position'])))
                env.drive(*row['base_command'])
                env.step(row['action'])
                contact = bool(env.bilateral_contacts().any())
                contact_count += int(contact)
                contact_run = contact_run + 1 if contact else 0
                longest_contact = max(longest_contact, contact_run)
                if contact and first_contact is None:
                    first_contact = step
            results[name] = dict(steps=len(indices), max_joint_state_deviation_rad=max_state_error,
                max_tcp_deviation_m=max_tcp_error, bilateral_contact_steps=contact_count,
                longest_bilateral_contact_run=longest_contact, first_contact_step=first_contact,
                events=env.events, outcome=harvest_outcome(env, initial_roll))
            env.close()
            print(json.dumps({name: results[name]}), flush=True)
    args.out.write_text(json.dumps(dict(scope='Offline expert-command audit, not ACT success',
        physics_sources_verified=True, physics_source=str(physics_source),
        source=str(source), runs=results), indent=2))


if __name__ == '__main__':
    main()
