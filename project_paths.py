"""Paths for relocated development tools; online control does not import this."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
DEVELOPMENT_DIRS = ('tests', 'tools/validation', 'tools/diagnostics')

# Keep direct script execution and unittest discovery independent of cwd.
for relative in DEVELOPMENT_DIRS:
    path = str(PROJECT_ROOT / relative)
    if path not in sys.path:
        sys.path.append(path)


def project_source(name):
    """Resolve a source filename for offline hash reports after relocation."""
    path = PROJECT_ROOT / name
    if path.is_file():
        return path
    matches = [PROJECT_ROOT / directory / name for directory in DEVELOPMENT_DIRS
               if (PROJECT_ROOT / directory / name).is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one project source for {name!r}')
    return matches[0]
