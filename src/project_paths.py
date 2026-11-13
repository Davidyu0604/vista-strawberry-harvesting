"""Repository paths shared by source snapshots, entry points and developer tools."""
from pathlib import Path
import sys

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
PROMPT_DIR = PROJECT_ROOT / 'prompts'
DEVELOPMENT_DIRS = ('tests', 'tools/validation', 'tools/diagnostics')

for relative in DEVELOPMENT_DIRS:
    path = str(PROJECT_ROOT / relative)
    if path not in sys.path:
        sys.path.append(path)


def project_source(name):
    """Resolve current code/prompts; historical snapshots keep their original names."""
    for directory in (SOURCE_DIR, PROMPT_DIR, PROJECT_ROOT):
        path = directory / name
        if path.is_file():
            return path
    matches = [PROJECT_ROOT / directory / name for directory in DEVELOPMENT_DIRS
               if (PROJECT_ROOT / directory / name).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one project source for {name!r}')
    return matches[0]


def source_files():
    return sorted([*SOURCE_DIR.glob('*.py'), *PROMPT_DIR.glob('*.txt')])
