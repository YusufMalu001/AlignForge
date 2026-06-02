"""
CLI entrypoint for Evaluation (Generation + Judging + Metrics).
"""
import os
os.environ["PYTHONUTF8"] = "1"
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from evaluation.evaluator import run_evaluation
from evaluation.reward_eval import run_reward_eval
from evaluation.metrics import run_metrics

def run_all():
    print("Step 1: Running generation on evaluation prompts...")
    run_evaluation()
    print("Step 2: Scoring responses with Local Reward Model...")
    run_reward_eval()
    print("Step 3: Computing final metrics...")
    run_metrics()
    print("Evaluation pipeline complete!")

if __name__ == "__main__":
    run_all()
