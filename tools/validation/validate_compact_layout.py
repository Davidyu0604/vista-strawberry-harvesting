"""Compatibility entry for current layout audit.

The historical compact-v20 validator and generator are preserved under
reports/01_控制验证/mobile-compact-v1/source/. This entry audits the current
maturity profile instead; it does not reproduce the old compact scene.
"""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse
from pathlib import Path
from validate_mobile_short_search import validate

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print('Auditing current maturity layout; compact-v20 sources are archived.',flush=True)
    validate(args.out)
