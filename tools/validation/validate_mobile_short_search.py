"""Compatibility entry: audit the current moderate profile.

The v21 validator is archived in reports/01_控制验证/mobile-short-search-v1/source/.
"""

# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from project_paths import PROJECT_ROOT, project_source

import argparse
import tempfile
from pathlib import Path
from validate_maturity_layout import validate as validate_current

def validate(out):
    with tempfile.TemporaryDirectory(prefix='maturity-preview-') as preview:
        validate_current(out,Path(preview))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print('Auditing current maturity-layout profile; historical v21 sources remain archived.',flush=True)
    validate(a.out)
