"""
CLI entrypoint for Evaluation (Generation + Judging + Metrics).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from evaluation.evaluator import run_evaluation
from evaluation.judge import run_judge
from evaluation.metrics import run_metrics

def run_all():
    print("Step 1: Running generation on evaluation prompts...")
    run_evaluation()
    print("Step 2: Judging responses with LLM-as-a-judge...")
    run_judge()
    print("Step 3: Computing final metrics...")
    run_metrics()
    print("Evaluation pipeline complete!")

if __name__ == "__main__":
    run_all()
