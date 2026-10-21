"""Run the curated research implementations from one repository root."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ENTRYPOINTS = {
    'smaf': ('SMAF-Net/trainer.py',),
    'concat': ('SMAF-Net/trainer.py',),
    'missing': ('SMAF-Net/evaluate.py', 'structured'),
    'random-missing': ('SMAF-Net/evaluate.py', 'random'),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=ENTRYPOINTS)
    args, forwarded = parser.parse_known_args()
    script, *mode = ENTRYPOINTS[args.command]
    if '--config' not in forwarded and not any(x.startswith('--config=') for x in forwarded):
        parser.error('Pass --config Configs/<configuration>.yaml explicitly.')
    return subprocess.call([sys.executable, str(ROOT / script), *mode, *forwarded], cwd=ROOT)

if __name__ == '__main__':
    raise SystemExit(main())
