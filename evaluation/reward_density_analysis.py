import json
import logging
import numpy as np
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_density_analysis(judgments_file: str, output_dir: str):
    logger.info("Running Reward Density Analysis...")
    path = Path(judgments_file)
    if not path.exists():
        logger.error(f"Could not find judgments file at {judgments_file}")
        return
        
    with open(path, "r") as f:
        judgments = json.load(f)
        
    rpt = []
    rps = []
    
    for j in judgments:
        words = j['dpo_response'].split()
        tokens = len(words) * 1.3 # rough token approximation
        sentences = max(1, len(j['dpo_response'].split('.')))
        reward = j['dpo_reward']
        
        rpt.append(reward / tokens if tokens > 0 else 0)
        rps.append(reward / sentences if sentences > 0 else 0)
        
    report = {
        "reward_per_token_mean": float(np.mean(rpt)),
        "reward_per_sentence_mean": float(np.mean(rps)),
        "reward_density_variance": float(np.var(rpt))
    }
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "reward_density_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Density complete. Reward/Token: {report['reward_per_token_mean']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments", type=str, required=True)
    parser.add_argument("--output", type=str, default="./reports")
    args = parser.parse_args()
    run_density_analysis(args.judgments, args.output)
