"""
CLI entrypoint for Reward Model Training.
"""
import os
os.environ["PYTHONUTF8"] = "1"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from training.reward_trainer import run_rm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max_samples", type=int, default=None, help="Override number of training samples")
    args = parser.parse_args()
    run_rm(args.resume, args.max_samples)
