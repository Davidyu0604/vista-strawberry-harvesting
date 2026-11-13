"""Serial windowed fresh50 collection; five attempts per audited sub-batch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from export_act_dataset import check_harvest
from recording_quality import atomic_json, audit_recording

ROOT = Path(__file__).resolve().parent
FLAGS = [
    '--peduncle-test', '--viewer', '--record-images', '--allow-row-search',
    '--mobile-layout', '--mobile-layout-level', 'moderate',
    '--centered-grasp', '--visual-proposals', '--max-calls', '32',
]


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([*ROOT.glob('*.py'), *ROOT.glob('*.txt')])}


def inspect_episode(dest, seed, code, frozen):
    report = json.loads((dest / 'report.json').read_text())
    config = json.loads((dest / 'run-config.json').read_text())
    quality = audit_recording(dest, write_report=False)
    events = [json.loads(line) for line in (dest / 'vista-events.jsonl').read_text().splitlines()]
    calls = sum(event.get('kind') == 'request' for event in events)
    budget_exhausted=report.get('stop_reason')=='call_budget_exhausted'
    if code or report.get('error') or report.get('stop_reason') not in ('model_finish','call_budget_exhausted'):
        raise ValueError('Runtime error, interruption or non-model finish; stop collection')
    if (not quality['integrity_passed'] or not quality['recording_finalized']
            or (not budget_exhausted and not quality['episode_completed'])):
        raise ValueError('Raw recording integrity or completion check failed')
    if calls != report.get('model_calls') or not 0 < calls <= 32:
        raise ValueError('Model call count mismatch or budget exceeded')
    if budget_exhausted and calls != config['max_calls']:
        raise ValueError('Budget stop before configured call limit')
    if config.get('seed') != seed or config.get('initial_arm_pose') != 'search_stow':
        raise ValueError('Seed or initial arm pose differs from collection plan')
    if config.get('basket_mount') != 'chassis' or config.get('policy_cameras') != ['harvest_close', 'wrist']:
        raise ValueError('Expected chassis basket and two policy cameras')
    for flag in ('mobile_layout', 'centered_grasp', 'visual_proposals'):
        if config.get(flag) is not True:
            raise ValueError(f'Missing collection configuration: {flag}')
    if config.get('mobile_layout_level') != 'moderate' or config.get('fixed_base') is not False:
        raise ValueError('Expected moderate mobile layout with row search')
    snapshot = config.get('source_sha256', {})
    if not snapshot or any(frozen.get(name) != digest or
                           hashlib.sha256((dest / 'source' / name).read_bytes()).hexdigest() != digest
                           for name, digest in snapshot.items()):
        raise ValueError('Episode source snapshot differs from frozen source')
    if source_hashes() != frozen:
        raise ValueError('Source changed during collection')
    success = bool(report.get('success'))
    if success:
        check_harvest(dict(report, basket_mount=config['basket_mount']))
    return {'seed': seed, 'path': str(dest), 'model_calls': calls,
            'stop_reason':report['stop_reason'],'episode_completed':quality['episode_completed'],
            'harvest_success': success, 'quality': quality,
            'source_verified': True,
            'final_training_approval': ('blocked' if quality['behaviour_review_blocks_training']
                                        else 'pending_behavior_review')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start-seed', type=int, default=16)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.start_seed < 0 or not 1 <= args.episodes <= 10:
        parser.error('Choose a nonnegative start seed and 1–10 attempts')
    seeds = list(range(args.start_seed, args.start_seed + args.episodes))
    batches = [seeds[i:i + 5] for i in range(0, len(seeds), 5)]
    out = args.out.resolve()
    frozen = source_hashes()
    plan = {'seeds': seeds, 'sub_batches': batches, 'max_calls_per_episode': 32,
            'max_calls_per_sub_batch': 160, 'max_calls_total': 32 * len(seeds),
            'source_sha256': frozen, 'flags': FLAGS, 'automatic_retries': False,
            'note': 'Attempts, not guaranteed successful demonstrations. Preserve failures; '
                    'no automatic ACT export or final behavior approval.'}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'plan.json', plan)
    results = []
    status = {'planned_seeds': seeds, 'results': results, 'stop_reason': None,
              'final_training_approval': 'pending_behavior_review'}
    env = dict(os.environ, MUJOCO_GL='glfw', PYOPENGL_PLATFORM='glx',
               PYTHONPATH='', PYTHONNOUSERSITE='1')
    try:
        for batch_index, batch in enumerate(batches, 1):
            batch_dir = out / f'batch-{batch_index}'
            batch_dir.mkdir()
            batch_results = []
            for seed in batch:
                if source_hashes() != frozen:
                    raise ValueError('Source changed; no further episode started')
                dest = batch_dir / f'seed-{seed}'
                status['active_seed'] = seed
                atomic_json(out / 'collection-summary.json', status)
                print(f'\n[COLLECT] batch={batch_index} seed={seed} out={dest}', flush=True)
                command = [sys.executable, '-u', str(ROOT / 'vista.py'), *FLAGS,
                           '--seed', str(seed), '--out', str(dest)]
                # Inherit terminal output so the user sees every model/tool call.
                code = subprocess.run(command, env=env).returncode
                row = inspect_episode(dest, seed, code, frozen)
                results.append(row)
                batch_results.append(row)
                status['active_seed'] = None
                status['total_model_calls'] = sum(r['model_calls'] for r in results)
                status['harvest_successes'] = sum(r['harvest_success'] for r in results)
                atomic_json(out / 'collection-summary.json', status)
                print(f"[COLLECT] seed={seed} harvest={row['harvest_success']} "
                      f"integrity=passed calls={row['model_calls']}", flush=True)
            batch_audit = {'results': batch_results,
                           'model_calls': sum(r['model_calls'] for r in batch_results),
                           'harvest_successes': sum(r['harvest_success'] for r in batch_results),
                           'final_training_approval': 'pending_behavior_review'}
            if batch_audit['model_calls'] > 160:
                raise ValueError('Sub-batch call budget exceeded')
            atomic_json(batch_dir / 'collection-checks.json', batch_audit)
            print(f'[COLLECT] batch={batch_index} raw checks complete', flush=True)
    except (KeyboardInterrupt, Exception) as exc:
        status['stop_reason'] = f'{type(exc).__name__}: {exc}'
        atomic_json(out / 'collection-summary.json', status)
        print(f"[COLLECT] STOP {status['stop_reason']}; originals retained", flush=True)
        return 1
    status['completed_attempts'] = len(results)
    atomic_json(out / 'collection-summary.json', status)
    print(f'[COLLECT] DONE {out / "collection-summary.json"}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
