"""Persistent ACT inference worker; runs in the existing training environment."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

CAMERAS = ('harvest_close', 'wrist')


def configure_rollout(config, checkpoint, *, allow_diagnostic=False, n_action_steps=None):
    if checkpoint.get('diagnostic', True) and not allow_diagnostic:
        raise ValueError('Diagnostic checkpoint requires --allow-diagnostic')
    if n_action_steps is not None:
        if not 1 <= n_action_steps <= config.chunk_size:
            raise ValueError('n_action_steps must be between 1 and chunk_size')
        if config.temporal_ensemble_coeff is not None and n_action_steps != 1:
            raise ValueError('Temporal ensembling requires n_action_steps=1')
        config.n_action_steps = n_action_steps


class Preprocessor:
    """Identical state/image transforms and inverse action transform to train_act."""
    def __init__(self, stats, device):
        self.device = torch.device(device)
        self.stats = {key: {name: torch.tensor(value, device=self.device, dtype=torch.float32)
                           for name, value in values.items()} for key, values in stats.items()}
        for key, size in (('observation.state', 9), ('action', 8)):
            for name in ('mean', 'std'):
                value = self.stats[key][name]
                if value.shape != (size,) or not torch.isfinite(value).all():
                    raise ValueError('Invalid normalization statistics')
            if (self.stats[key]['std'] <= 0).any():
                raise ValueError('Normalization std must be positive')
        self.image_mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.image_std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]

    def prepare(self, state, images):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.shape != (9,) or not torch.isfinite(state).all() or set(images) != set(CAMERAS):
            raise ValueError('Expected finite 9D state and exactly two policy cameras')
        stats = self.stats['observation.state']
        batch = {'observation.state': ((state - stats['mean']) / stats['std'])[None]}
        for camera in CAMERAS:
            array = np.asarray(images[camera])
            if array.shape != (360, 480, 3) or array.dtype != np.uint8:
                raise ValueError('Expected recorded-size uint8 RGB images')
            image = torch.from_numpy(array.copy()).to(self.device).permute(2, 0, 1)[None].float() / 255.
            image = F.interpolate(image, size=(240, 320), mode='bilinear', align_corners=False)
            batch['observation.images.' + camera] = (image - self.image_mean) / self.image_std
        return batch

    def decode_action(self, action):
        if action.shape != (1, 8) or not torch.isfinite(action).all():
            raise ValueError('Policy returned invalid action')
        stats = self.stats['action']
        return (action[0] * stats['std'] + stats['mean']).cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--allow-diagnostic', action='store_true')
    parser.add_argument('--n-action-steps', type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in ACT environment')
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    config = ACTConfig.from_pretrained(args.checkpoint.parent / 'policy_config')
    config.device = args.device
    config.pretrained_backbone_weights = None  # All trained parameters come from checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    configure_rollout(config, checkpoint, allow_diagnostic=args.allow_diagnostic,
                      n_action_steps=args.n_action_steps)
    policy = ACTPolicy(config).to(args.device)
    policy.load_state_dict(checkpoint['model'], strict=True)
    policy.eval()
    policy.reset()
    prep = Preprocessor(checkpoint['normalization'], args.device)
    print(json.dumps({'ready': True, 'checkpoint_step': checkpoint['step'],
                      'dataset_manifest_sha256': checkpoint['dataset_manifest_sha256'],
                      'device': args.device, 'n_action_steps': config.n_action_steps,
                      'chunk_size': config.chunk_size,
                      'diagnostic': checkpoint.get('diagnostic', True),
                      'diagnostic_kind': checkpoint.get('diagnostic_kind'),
                      'total_single_episode_steps': checkpoint.get('total_single_episode_steps'),
                      'training_episode_seed': checkpoint.get('training_episode_seed')}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        images = {}
        for name, encoded in request['images'].items():
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                images[name] = np.asarray(image.convert('RGB'))
        with torch.inference_mode():
            action = policy.select_action(prep.prepare(request['state'], images))
            response = prep.decode_action(action)
        print(json.dumps({'action': response}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}), flush=True)
        raise
