import json
import logging
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_reward_bias_analysis(judgments_file: str, output_dir: str):
    logger.info("Running Reward Bias Analysis (Length Correlation)...")
    
    path = Path(judgments_file)
    if not path.exists():
        logger.error(f"Could not find benchmark file at {judgments_file}")
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
        
    if not lengths:
        logger.warning("No data points found for analysis.")
        return
        
    pearson_r, pearson_p = stats.pearsonr(lengths, rewards)
    spearman_rho, spearman_p = stats.spearmanr(lengths, rewards)
    
    report = {
        "data_points": len(lengths),
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p_value": float(spearman_p),
        "length_bias_detected": bool(pearson_r > 0.4 and pearson_p < 0.05)
    }
    
    logger.info(f"Pearson r: {pearson_r:.4f} (p={pearson_p:.4f})")
    logger.info(f"Spearman rho: {spearman_rho:.4f} (p={spearman_p:.4f})")
    
    if report["length_bias_detected"]:
        logger.warning("WARNING: Strong positive correlation between length and reward detected. RM is biased.")
    else:
        logger.info("RM length bias is within acceptable bounds.")
        
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "reward_bias_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # Generate Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, rewards, alpha=0.6, edgecolors='k')
    
    # Add trendline
    z = np.polyfit(lengths, rewards, 1)
    p = np.poly1d(z)
    plt.plot(lengths, p(lengths), "r--", alpha=0.8, label=f"Trendline (r={pearson_r:.2f})")
    
    plt.title("Reward Bias Analysis: Reward vs. Response Length")
    plt.xlabel("Response Length (words)")
    plt.ylabel("Reward Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = out_dir / "reward_vs_length.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {plot_path}")
    plt.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments", type=str, required=True, help="Path to benchmark judgments JSON")
    parser.add_argument("--output", type=str, default="./results", help="Output directory")
    args = parser.parse_args()
    run_reward_bias_analysis(args.judgments, args.output)
