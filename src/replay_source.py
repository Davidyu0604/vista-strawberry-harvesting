"""Select hash-verified physics sources for a historical episode."""
import hashlib
import importlib
import inspect
from pathlib import Path
import sys
from project_paths import PROJECT_ROOT, SOURCE_DIR


def replay_environment(episode, config, report, *, asset_root=None):
    root = Path(asset_root) if asset_root is not None else PROJECT_ROOT
    required = ['peduncle.py', 'env.py', 'scene.py', 'chassis.py']
    required += [n for n in ('mounted_basket.py', 'natural_assets.py', 'scene_details.py', 'mobile_scene.py', 'project_paths.py')
                 if n in config['source_sha256']]
    candidates = [SOURCE_DIR, episode/'source', episode.parent/'source', episode.parent.parent/'source']
    def matches(directory):
        return all((directory/n).is_file() and
                   hashlib.sha256((directory/n).read_bytes()).hexdigest() == config['source_sha256'][n]
                   for n in required)
    selected = next((p for p in candidates if matches(p)), None)
    if selected is None:
        raise ValueError('No matching physics source snapshot; use the recorded replay video')
    # Run this entry point in its own process, before importing simulation modules.
    if any(n in sys.modules for n in ('peduncle', 'env', 'scene', 'chassis')):
        raise RuntimeError('Replay source selection must precede simulation imports')
    sys.path.insert(0, str(selected))
    assets = importlib.import_module('natural_assets')
    assets.ASSETS = root/'assets/natural'
    scene = importlib.import_module('scene')
    scene.ROOT = root
    cls = importlib.import_module('peduncle').PeduncleEnv
    kwargs = dict(seed=config['seed'], single=report['single_fruit_fixture'],
                  fixed_base=config.get('fixed_base', False), render_quality=config['quality'])
    if 'basket_mount' in inspect.signature(cls).parameters:
        kwargs['basket_mount'] = config.get('basket_mount', 'ground')
    if 'mobile_layout' in inspect.signature(cls).parameters:
        kwargs['mobile_layout'] = config.get('mobile_layout', False)
    if 'mobile_layout_level' in inspect.signature(cls).parameters:
        # Pre-profile recordings used the original wide-gap generator.
        kwargs['mobile_layout_level'] = config.get('mobile_layout_level', 'stress')
    if config.get('initial_arm_pose')=='search_stow':
        kwargs['initial_arm_pose']='search_stow'
    return cls(**kwargs), str(selected)
