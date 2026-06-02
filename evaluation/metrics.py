"""
Computes various evaluation metrics based on judgments.
Includes win rate, average length, toxicity (simulated/placeholder), repetition, token efficiency.
"""
import json
from pathlib import Path
import logging
import re
from typing import List, Dict

from tracking.logger import load_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def compute_vocab_diversity(text: str) -> float:
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def compute_repetition_score(text: str) -> float:
    """Simple proxy for repetition: high ratio of repeated n-grams is bad."""
    words = text.split()
    if len(words) < 4:
        return 0.0
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    return 1.0 - (len(set(bigrams)) / len(bigrams))

def run_metrics():
    config = load_config()
    output_dir = Path(config['output_dir'])
    judgments_path = output_dir / "reward_eval_outputs.json"
    
    if not judgments_path.exists():
        logger.error(f"Judgments not found at {judgments_path}. Run reward_eval first.")
        return
        
    with open(judgments_path, "r") as f:
        judgments = json.load(f)
        
    total = len(judgments)
    if total == 0:
        return
        
    wins = sum(1 for j in judgments if j['winner'] == 'win')
    losses = sum(1 for j in judgments if j['winner'] == 'loss')
    ties = sum(1 for j in judgments if j['winner'] == 'tie')
    
    base_rewards = [j['baseline_reward'] for j in judgments]
    dpo_rewards = [j['dpo_reward'] for j in judgments]
    
    avg_base_reward = sum(base_rewards) / total
    avg_dpo_reward = sum(dpo_rewards) / total
    reward_gain = avg_dpo_reward - avg_base_reward
    
    base_lengths = [len(j['baseline_response'].split()) for j in judgments]
    dpo_lengths = [len(j['dpo_response'].split()) for j in judgments]
    
    base_diversity = [compute_vocab_diversity(j['baseline_response']) for j in judgments]
    dpo_diversity = [compute_vocab_diversity(j['dpo_response']) for j in judgments]
    
    base_rep = [compute_repetition_score(j['baseline_response']) for j in judgments]
    dpo_rep = [compute_repetition_score(j['dpo_response']) for j in judgments]
    
    metrics = {
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "tie_rate": ties / total,
        "avg_baseline_reward": avg_base_reward,
        "avg_dpo_reward": avg_dpo_reward,
        "reward_gain": reward_gain,
        "avg_length_baseline": sum(base_lengths) / total,
        "avg_length_dpo": sum(dpo_lengths) / total,
        "vocab_diversity_baseline": sum(base_diversity) / total,
        "vocab_diversity_dpo": sum(dpo_diversity) / total,
        "repetition_score_baseline": sum(base_rep) / total,
        "repetition_score_dpo": sum(dpo_rep) / total
    }
    
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    logger.info(f"Metrics saved to {output_dir / 'final_metrics.json'}")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

if __name__ == "__main__":
    run_metrics()
