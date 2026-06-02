"""
CLI entrypoint for DPO.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from training.dpo_trainer import run_dpo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max_samples", type=int, default=None, help="Override number of training samples")
    args = parser.parse_args()
    run_dpo(args.resume, args.max_samples)
