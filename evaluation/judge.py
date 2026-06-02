"""
GPT-4o-mini as judge for win rate.
"""
import json
from pathlib import Path
import logging
from tqdm import tqdm

from tracking.logger import load_config
from models.reward_scorer import RewardScorer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_judge():
    config = load_config()
    output_dir = Path(config['output_dir'])
    eval_outputs_path = output_dir / "eval_outputs.json"
    
    if not eval_outputs_path.exists():
        logger.error(f"Eval outputs not found at {eval_outputs_path}. Run evaluator first.")
        return
        
    with open(eval_outputs_path, "r") as f:
        outputs = json.load(f)
        
    scorer = RewardScorer()
    judgments = []
    
    win_count = 0
    loss_count = 0
    tie_count = 0
    
    logger.info("Judging responses...")
    for item in tqdm(outputs):
        winner = scorer.score_pair(item['prompt'], item['baseline_response'], item['dpo_response'])
        
        if winner == 'B':
            decision = "win"
            win_count += 1
        elif winner == 'A':
            decision = "loss"
            loss_count += 1
        else:
            decision = "tie"
            tie_count += 1
            
        judgments.append({
            "prompt": item['prompt'],
            "baseline_response": item['baseline_response'],
            "dpo_response": item['dpo_response'],
            "winner": decision
        })
        
    with open(output_dir / "judgments.json", "w") as f:
        json.dump(judgments, f, indent=2)
        
    logger.info(f"Judgments saved to {output_dir / 'judgments.json'}")
    logger.info(f"Win: {win_count}, Loss: {loss_count}, Tie: {tie_count}")

if __name__ == "__main__":
    run_judge()
