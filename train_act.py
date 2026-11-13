"""Local ACT baseline training on the prepared 50/10 split; no upload or simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

from recording_quality import atomic_json


def training_indices(episodes, split):
    train, validation = set(split['train']), set(split['validation'])
    if len(train) != 50 or len(validation) != 10 or train & validation:
        raise ValueError('Expected 50 training and 10 validation episodes')
    if train | validation != set(range(len(episodes))):
        raise ValueError('Split must cover all episodes')
    train_seeds = {e['seed'] for e in episodes if e['episode_index'] in train}
    validation_seeds = {e['seed'] for e in episodes if e['episode_index'] in validation}
    if train_seeds & validation_seeds:
        raise ValueError('A seed occurs on both sides of the split')
    indices = [i for e in episodes if e['episode_index'] in train
               for i in range(e['dataset_from_index'], e['dataset_to_index'])]
    return indices


def normalization(numeric, indices):
    stats = {}
    for key in ('observation.state', 'action'):
        values = np.asarray(numeric[key], dtype=np.float32)[indices]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite training data')
        stats[key] = {'mean': values.mean(0).tolist(),
                      'std': np.maximum(values.std(0), 1e-3).tolist()}
    return stats


def save_checkpoint(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def train(args):
    torch.set_num_threads(4)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use run-act-train.sh and verify nvidia-smi; no CPU fallback for full training.')
    if args.device == 'cpu' and not args.diagnostic:
        raise ValueError('CPU is available only for an explicit diagnostic run')
    if min(args.steps, args.batch_size, args.eval_every, args.validation_frames_per_episode) < 1 or args.workers < 0:
        raise ValueError('Invalid training limits')
    device = torch.device(args.device)
    root, out = args.dataset.resolve(), args.out.resolve()
    if (root / 'EXPORT_INCOMPLETE.json').exists():
        raise ValueError('Dataset export is incomplete')
    verification = json.loads((root / 'export_verification.json').read_text())
    if not verification.get('passed'):
        raise ValueError('Dataset verification failed')
    manifest_path = root / 'export_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    dataset_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    indices = training_indices(manifest['sources'], manifest['split'])
    if args.resume:
        if not (out / 'last.pt').exists():
            raise ValueError('No last.pt to resume')
    else:
        out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ds = LeRobotDataset(manifest['repo_id'], root=root,
                        delta_timestamps={'action': [j / 20 for j in range(50)]})
    if len(ds) != manifest['frames'] or ds.num_episodes != 60:
        raise ValueError('Dataset shape differs from manifest')
    cameras = [f'observation.images.{c}' for c in manifest['cameras']]
    allowed = set(cameras) | {'observation.state', 'action', 'timestamp', 'frame_index',
                              'episode_index', 'index', 'task_index'}
    if set(ds.features) - allowed:
        raise ValueError('Unexpected policy features')
    # Confirm 50-step chunks cannot leak into the next episode, for all 60 ends.
    for episode in manifest['sources']:
        sample = ds[episode['dataset_to_index'] - 1]
        expected_pad = torch.arange(50) > 0
        if not torch.equal(sample['action_is_pad'], expected_pad):
            raise ValueError('Action chunk crossed an episode boundary')
        torch.testing.assert_close(sample['action'], sample['action'][0].expand(50, -1))
    numeric = ds.hf_dataset.select_columns(['observation.state', 'action']).with_format('numpy')
    stats = normalization(numeric, indices)
    tensors = {k: {name: torch.tensor(values, device=device) for name, values in v.items()}
               for k, v in stats.items()}
    image_mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    image_std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]

    def prepare_batch(batch):
        result = {k: (batch[k].to(device) - tensors[k]['mean']) / tensors[k]['std']
                  for k in ('observation.state', 'action')}
        result['action_is_pad'] = batch['action_is_pad'].to(device)
        for key in cameras:
            image = F.interpolate(batch[key].to(device), size=(240, 320), mode='bilinear', align_corners=False)
            result[key] = (image - image_mean) / image_std
        return result

    config = ACTConfig(
        input_features={'observation.state': PolicyFeature(FeatureType.STATE, (9,)),
                        **{c: PolicyFeature(FeatureType.VISUAL, (3, 240, 320)) for c in cameras}},
        output_features={'action': PolicyFeature(FeatureType.ACTION, (8,))},
        chunk_size=50, n_action_steps=1, device=str(device), push_to_hub=False,
        pretrained_backbone_weights=(None if args.resume else 'ResNet18_Weights.IMAGENET1K_V1'))
    policy = ACTPolicy(config).to(device)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=config.optimizer_lr,
                                  weight_decay=config.optimizer_weight_decay)
    start_step, best = 0, float('inf')
    if args.resume:
        checkpoint = torch.load(out / 'last.pt', map_location=device, weights_only=True)
        if checkpoint['dataset_manifest_sha256'] != dataset_hash:
            raise ValueError('Resume dataset changed')
        if checkpoint['diagnostic'] != args.diagnostic or checkpoint['normalization'] != stats:
            raise ValueError('Resume training scope or normalization changed')
        policy.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step, best = checkpoint['step'], checkpoint['best_validation_l1']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint['cuda_rng']])
    else:
        config.save_pretrained(out / 'policy_config')
        atomic_json(out / 'normalization.json', stats)
        atomic_json(out / 'normalization_provenance.json', {
            'train_episode_indices': manifest['split']['train'], 'frames_used': len(indices),
            'held_out_frames_used': 0, 'image_normalization': 'ImageNet fixed mean/std'})
        atomic_json(out / 'run-config.json', {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'dataset_manifest_sha256': dataset_hash, 'split': manifest['split'],
            'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
            'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
            'chunk_size': 50, 'image_resize': [240, 320], 'precision': 'float32',
            'pretrained_backbone': 'ResNet18_Weights.IMAGENET1K_V1',
            'lr': config.optimizer_lr, 'lr_backbone': config.optimizer_lr_backbone,
            'note': 'Behavior-cloning baseline. Validation imitation loss is not a closed-loop harvest success rate.'})
    loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == 'cuda',
                        persistent_workers=args.workers > 0, drop_last=False)
    validation_indices = [int(i) for e in manifest['sources']
                          if e['episode_index'] in manifest['split']['validation']
                          for i in np.linspace(e['dataset_from_index'], e['dataset_to_index'] - 1,
                                               min(args.validation_frames_per_episode, e['frames']), dtype=int)]
    validation_loader = DataLoader(Subset(ds, validation_indices), batch_size=args.batch_size, shuffle=False,
                                   num_workers=0)

    def evaluate():
        policy.eval()
        error, count = 0., 0
        with torch.inference_mode():
            for batch in validation_loader:
                batch = prepare_batch(batch)
                prediction = policy.predict_action_chunk(batch)
                mask = ~batch['action_is_pad'].unsqueeze(-1)
                error += ((prediction - batch['action']).abs() * mask).sum().item()
                count += int(mask.sum()) * 8
        score = error / count
        if not np.isfinite(score):
            raise RuntimeError('Nonfinite validation score')
        return score

    started = time.monotonic()
    step = start_step
    iterator = iter(loader)
    try:
        while step < args.steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = policy(prepare_batch(batch))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            row = {'step': step, 'loss': loss.item(), 'grad_norm': norm.item(),
                   'elapsed_seconds': time.monotonic() - started, **metrics}
            if step % args.eval_every == 0 or step == args.steps:
                row['validation_normalized_l1'] = evaluate()
                if row['validation_normalized_l1'] < best:
                    best = row['validation_normalized_l1']
                    save_checkpoint(out / 'best.pt', {'model': policy.state_dict(), 'step': step,
                        'normalization': stats, 'dataset_manifest_sha256': dataset_hash,
                        'validation_normalized_l1': best, 'diagnostic': args.diagnostic})
                save_checkpoint(out / 'last.pt', {'model': policy.state_dict(), 'optimizer': optimizer.state_dict(),
                    'step': step, 'best_validation_l1': best, 'normalization': stats,
                    'dataset_manifest_sha256': dataset_hash, 'diagnostic': args.diagnostic,
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []})
            with (out / 'metrics.jsonl').open('a') as log:
                log.write(json.dumps(row) + '\n')
            if step == start_step + 1 or step % 50 == 0 or 'validation_normalized_l1' in row:
                print(json.dumps(row), flush=True)
        atomic_json(out / 'training_result.json', {'status': 'completed', 'steps': step,
            'best_validation_normalized_l1': best, 'diagnostic': args.diagnostic,
            'elapsed_seconds_this_run': time.monotonic() - started,
            'max_cuda_memory_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else 0,
            'closed_loop_rollout_evaluated': False})
    except BaseException as error:
        atomic_json(out / 'training_result.json', {'status': 'interrupted_or_failed', 'steps': step,
            'error_type': type(error).__name__, 'resume_from': 'last.pt if present; last validated checkpoint only'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--validation-frames-per-episode', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--diagnostic', action='store_true', help='Mark a short setup test, not a trained policy')
    train(parser.parse_args())
