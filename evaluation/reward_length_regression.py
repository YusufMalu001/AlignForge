import json
import logging
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_regression(judgments_file: str, output_dir: str):
    logger.info("Running Reward-Length Regression Analysis...")
    path = Path(judgments_file)
    if not path.exists():
        logger.error(f"Could not find judgments file at {judgments_file}")
        return
        
    with open(path, "r") as f:
        judgments = json.load(f)
        
    lengths = []
    rewards = []
    
    for j in judgments:
        lengths.append(len(j['dpo_response'].split()))
        rewards.append(j['dpo_reward'])
        lengths.append(len(j['baseline_response'].split()))
        rewards.append(j['baseline_reward'])
        
    lengths = np.array(lengths)
    rewards = np.array(rewards)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(lengths, rewards)
    r_squared = r_value ** 2
    
    report = {
        "beta_0": float(intercept),
        "beta_1": float(slope),
        "r_squared": float(r_squared),
        "p_value": float(p_value),
        "reward_explained_by_length": float(r_squared * 100)
    }
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "reward_length_regression.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # Generate Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, rewards, alpha=0.6, edgecolors='k')
    plt.plot(lengths, intercept + slope * lengths, "r--", alpha=0.8, label=f"Regression: R²={r_squared:.4f}")
    plt.title("Reward vs. Length Regression")
    plt.xlabel("Response Length (words)")
    plt.ylabel("Reward Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = out_dir / "reward_vs_length_regression.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Regression complete. beta_1={slope:.4f}. R²={r_squared:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments", type=str, required=True)
    parser.add_argument("--output", type=str, default="./reports")
    args = parser.parse_args()
    run_regression(args.judgments, args.output)
