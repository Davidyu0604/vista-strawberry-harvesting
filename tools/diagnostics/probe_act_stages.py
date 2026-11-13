"""Teacher-observation action errors by stage; diagnostic, not closed-loop success."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from act_inference_worker import Preprocessor


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, action='append', required=True)
    parser.add_argument('--seed', type=int, default=46)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((args.dataset / 'export_manifest.json').read_text())
    episode, = [e for e in manifest['sources'] if e['seed'] == args.seed]
    source = Path(episode['source'])
    raw = [json.loads(line) for line in (source / 'trajectory.jsonl').read_text().splitlines()]
    retained = [raw[i] for i in episode['source_row_indices']]
    groups = {'start': list(range(32))}
    for stage in ['pregrasp', 'approach', 'gripper', 'detach']:
        candidates = [i for i, row in enumerate(retained) if row['stage'] == stage]
        groups[stage] = [candidates[i] for i in np.linspace(0, len(candidates)-1, 16, dtype=int)]
    indices = sorted(set(i for group in groups.values() for i in group))
    ds = LeRobotDataset(manifest['repo_id'], root=args.dataset.resolve(),
        delta_timestamps={'action': [j / manifest['fps'] for j in range(50)]})
    results = []
    for path in args.checkpoint:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        config = ACTConfig.from_pretrained(path.parent / 'policy_config')
        config.device = args.device
        config.pretrained_backbone_weights = None
        policy = ACTPolicy(config).to(args.device)
        policy.load_state_dict(checkpoint['model'], strict=True)
        policy.eval()
        prep = Preprocessor(checkpoint['normalization'], args.device)
        records = {}
        loader = DataLoader(Subset(ds, [episode['dataset_from_index'] + i for i in indices]), batch_size=4)
        with torch.inference_mode():
            for batch in loader:
                prepared = {'observation.state': (batch['observation.state'].to(args.device) - prep.stats['observation.state']['mean']) / prep.stats['observation.state']['std']}
                for key in config.image_features:
                    resized = torch.nn.functional.interpolate(batch[key].to(args.device), size=(240, 320), mode='bilinear', align_corners=False)
                    prepared[key] = (resized - prep.image_mean) / prep.image_std
                norm = policy.predict_action_chunk(prepared)
                prediction = (norm * prep.stats['action']['std'] + prep.stats['action']['mean']).cpu().numpy()
                truth = batch['action'].numpy()
                scale = prep.stats['action']['std'].cpu().numpy()
                for j, global_index in enumerate(batch['index'].tolist()):
                    index = global_index - episode['dataset_from_index']
                    valid = ~batch['action_is_pad'][j].numpy()
                    error = np.abs(prediction[j] - truth[j])
                    records[index] = dict(dataset_frame=index, raw_frame=retained[index]['frame'],
                        stage=retained[index]['stage'], first_action=prediction[j, 0].tolist(),
                        expert_first_action=truth[j, 0].tolist(), first_abs_error=error[0].tolist(),
                        chunk_normalized_l1=float((error[valid] / scale).mean()),
                        executed25_normalized_l1=float((error[:25] / scale).mean()))
        metrics = {}
        for name, samples in groups.items():
            values = [records[i] for i in samples]
            metrics[name] = dict(frames=len(values),
                first_action_mae_per_dimension=np.mean([v['first_abs_error'] for v in values], axis=0).tolist(),
                chunk_normalized_l1=float(np.mean([v['chunk_normalized_l1'] for v in values])),
                executed25_normalized_l1=float(np.mean([v['executed25_normalized_l1'] for v in values])))
        result = dict(checkpoint=str(path), step=checkpoint['step'],
            total_single_episode_steps=checkpoint.get('total_single_episode_steps'), stages=metrics,
            samples=list(records.values()))
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'samples'}), flush=True)
        del policy, checkpoint
    args.out.write_text(json.dumps(dict(scope='Same-episode teacher observations; no held-out or closed-loop claim',
        policy_inputs='RGB images and joint/base state only; stages are used solely for offline grouping',
        results=results), indent=2))


if __name__ == '__main__':
    main()
