"""Fine-tune ACT on one training episode to diagnose closed-loop failures.

Probes reuse that episode: this measures memorization, never held-out accuracy.
Initialization and normalization come from an existing checkpoint; all new
optimizer updates use only the selected demonstration. Originals are read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from act_inference_worker import Preprocessor
from recording_quality import atomic_json
from train_act import save_checkpoint


def select_episode(manifest, seed):
    matches = [e for e in manifest['sources'] if e['seed'] == seed]
    if len(matches) != 1:
        raise ValueError('Expected exactly one episode for the requested seed')
    episode = matches[0]
    if episode['episode_index'] not in manifest['split']['train']:
        raise ValueError('Overfit diagnostic must not use a held-out episode')
    if episode['dataset_to_index'] - episode['dataset_from_index'] != episode['frames']:
        raise ValueError('Episode indices do not match frame count')
    return episode


def initialization_steps(checkpoint, seed, allow_diagnostic=False):
    """Allow explicit same-episode continuation, without claiming optimizer resume."""
    if not checkpoint.get('diagnostic', True):
        return 0
    if not allow_diagnostic:
        raise ValueError('Diagnostic initialization requires --allow-diagnostic-init')
    if (checkpoint.get('diagnostic_kind') != 'single_episode_finetune'
            or checkpoint.get('training_episode_seed') != seed):
        raise ValueError('Diagnostic initialization must use the same training episode')
    return checkpoint.get('total_single_episode_steps', checkpoint['step'])


def train(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this bounded training experiment')
    if min(args.steps, args.batch_size, args.eval_every) < 1:
        raise ValueError('Training limits must be positive')
    root, out, initial = args.dataset.resolve(), args.out.resolve(), args.init_checkpoint.resolve()
    manifest_path = root / 'export_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (root / 'EXPORT_INCOMPLETE.json').exists() or not json.loads((root / 'export_verification.json').read_text())['passed']:
        raise ValueError('Dataset must pass export verification')
    episode = select_episode(manifest, args.episode_seed)
    dataset_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    checkpoint = torch.load(initial, map_location='cpu', weights_only=True)
    if checkpoint['dataset_manifest_sha256'] != dataset_hash:
        raise ValueError('Initialization checkpoint belongs to another dataset')
    previous_steps = initialization_steps(checkpoint, args.episode_seed, args.allow_diagnostic_init)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = ACTConfig.from_pretrained(initial.parent / 'policy_config')
    config.device = 'cuda'
    config.pretrained_backbone_weights = None
    ds = LeRobotDataset(manifest['repo_id'], root=root,
                        delta_timestamps={'action': [j / manifest['fps'] for j in range(config.chunk_size)]})
    indices = list(range(episode['dataset_from_index'], episode['dataset_to_index']))
    last = ds[indices[-1]]
    expected_pad = torch.arange(config.chunk_size) > 0
    if not torch.equal(last['action_is_pad'], expected_pad):
        raise ValueError('Action chunks cross the episode boundary')
    policy = ACTPolicy(config).cuda()
    policy.load_state_dict(checkpoint['model'], strict=True)
    prep = Preprocessor(checkpoint['normalization'], 'cuda')
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=config.optimizer_lr,
                                  weight_decay=config.optimizer_weight_decay)

    def prepare(batch):
        result = {key: (batch[key].cuda() - prep.stats[key]['mean']) / prep.stats[key]['std']
                  for key in ('observation.state', 'action')}
        result['action_is_pad'] = batch['action_is_pad'].cuda()
        for key in config.image_features:
            image = F.interpolate(batch[key].cuda(), size=(240, 320), mode='bilinear', align_corners=False)
            result[key] = (image - prep.image_mean) / prep.image_std
        return result

    # Include every initial frame as well as evenly spaced frames across the episode.
    probe_indices = sorted(set(indices[:32]) | set(np.linspace(indices[0], indices[-1], 32, dtype=int).tolist()))
    loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size, shuffle=True, num_workers=0)
    probe_loader = DataLoader(Subset(ds, probe_indices), batch_size=args.batch_size, num_workers=0)

    @torch.inference_mode()
    def evaluate():
        policy.eval()
        total = first = start = hold = 0.
        count = first_count = start_count = 0
        for raw in probe_loader:
            batch = prepare(raw)
            predicted = policy.predict_action_chunk(batch)
            valid = ~batch['action_is_pad']
            errors = (predicted - batch['action']).abs()
            total += errors[valid].sum().item()
            count += int(valid.sum()) * 8
            first += errors[:, 0].sum().item()
            first_count += len(predicted) * 8
            initial_mask = raw['index'] < indices[0] + 32
            start += errors[initial_mask.cuda()][valid[initial_mask.cuda()]].sum().item()
            start_count += int(valid[initial_mask.cuda()].sum()) * 8
            constant = torch.cat((raw['observation.state'][:, :6], torch.zeros(len(predicted), 2)), dim=-1).cuda()
            constant = (constant - prep.stats['action']['mean']) / prep.stats['action']['std']
            hold += ((constant[:, None] - batch['action']).abs()[valid]).sum().item()
        return {'same_episode_chunk_l1': total / count, 'same_episode_first_action_l1': first / first_count,
                'start_chunk_l1': start / start_count, 'hold_position_chunk_l1': hold / count,
                'probe_frames': len(probe_indices), 'held_out_evaluation': False}

    out.mkdir(parents=True, exist_ok=False)
    config.save_pretrained(out / 'policy_config')
    provenance = {'diagnostic': True, 'diagnostic_kind': 'single_episode_finetune',
                  'training_episode_seed': args.episode_seed, 'episode_index': episode['episode_index'],
                  'frames': len(indices), 'dataset_from_index': indices[0], 'dataset_to_index': indices[-1] + 1,
                  'dataset_manifest_sha256': dataset_hash, 'initial_checkpoint': str(initial),
                  'initial_checkpoint_sha256': hashlib.sha256(initial.read_bytes()).hexdigest(),
                  'initial_checkpoint_step': checkpoint['step'], 'steps': args.steps, 'batch_size': args.batch_size,
                  'previous_single_episode_steps': previous_steps,
                  'seed': args.seed, 'lr': config.optimizer_lr, 'normalization': 'Inherited from initial 50-episode training',
                  'optimizer': 'Fresh AdamW', 'probe_indices': probe_indices, 'held_out_evaluation': False}
    atomic_json(out / 'run-config.json', provenance)
    atomic_json(out / 'normalization.json', checkpoint['normalization'])
    started = time.monotonic()
    best = float('inf')

    def save(name, step, probe):
        save_checkpoint(out / name, {'model': policy.state_dict(), 'step': step,
            'normalization': checkpoint['normalization'], 'dataset_manifest_sha256': dataset_hash,
            'diagnostic': True, 'diagnostic_kind': provenance['diagnostic_kind'],
            'training_episode_seed': args.episode_seed, 'initial_checkpoint_step': checkpoint['step'],
            'total_single_episode_steps': previous_steps + step,
            'same_episode_metrics': probe})

    iterator = iter(loader)
    step = 0
    try:
        initial_metrics = evaluate()
        atomic_json(out / 'initial-probe.json', initial_metrics)
        print(json.dumps({'step': 0, **initial_metrics}), flush=True)
        with (out / 'metrics.jsonl').open('w') as log:
            for step in range(1, args.steps + 1):
                try:
                    raw = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    raw = next(iterator)
                policy.train()
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = policy(prepare(raw))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                row = {'step': step, 'loss': loss.item(), 'grad_norm': norm.item(),
                       'elapsed_seconds': time.monotonic() - started, **metrics}
                if step % args.eval_every == 0 or step == args.steps:
                    probe = evaluate()
                    row.update(probe)
                    if probe['same_episode_chunk_l1'] < best:
                        best = probe['same_episode_chunk_l1']
                        save('best.pt', step, probe)
                    save('last.pt', step, probe)
                log.write(json.dumps(row) + '\n')
                log.flush()
                if step == 1 or step % 50 == 0 or 'same_episode_chunk_l1' in row:
                    print(json.dumps(row), flush=True)
        atomic_json(out / 'training_result.json', {'status': 'completed', 'diagnostic': True, 'steps': step,
                    'best_same_episode_chunk_l1': best, 'wall_seconds': time.monotonic() - started,
                    'held_out_evaluation': False, 'closed_loop_rollout_evaluated': False})
    except BaseException as exc:
        atomic_json(out / 'training_result.json', {'status': 'interrupted_or_failed', 'steps': step,
                    'diagnostic': True, 'error_type': type(exc).__name__})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--init-checkpoint', type=Path, required=True)
    parser.add_argument('--allow-diagnostic-init', action='store_true',
                        help='Continue same-episode weights with a fresh optimizer in a new directory')
    parser.add_argument('--episode-seed', type=int, default=46)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    train(parser.parse_args())
