"""
Local Reward Model as judge.
Scores baseline and DPO outputs to calculate win rate and reward gain.
"""
import json
from pathlib import Path
import logging
from tqdm import tqdm
import torch

from tracking.logger import load_config
from models.base_loader import get_reward_model_and_tokenizer, apply_dynamic_quantization

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def get_reward_score(model, tokenizer, prompt, response, max_length):
    # The prompt + response is the full text
    text = prompt + response
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    
    with torch.inference_mode():
        outputs = model(**inputs)
        # Sequence classification models with num_labels=1 return shape (batch_size, 1)
        score = outputs.logits[0].item()
    return score

def run_reward_eval():
    config = load_config()
    output_dir = Path(config['output_dir'])
    eval_outputs_path = output_dir / "eval_outputs.json"
    
    if not eval_outputs_path.exists():
        logger.error(f"Eval outputs not found at {eval_outputs_path}. Run evaluator first.")
        return
        
    with open(eval_outputs_path, "r") as f:
        outputs = json.load(f)
        
    rm_checkpoint = str(output_dir / "rm_checkpoint")
    if not Path(rm_checkpoint).exists():
        logger.error(f"RM checkpoint not found at {rm_checkpoint}. Run RM training first.")
        return
        
    model, tokenizer = get_reward_model_and_tokenizer(
        config['model_name'], 
        is_trainable=False,
        checkpoint_path=rm_checkpoint
    )
    # Apply int8 quantization for faster eval
    model = apply_dynamic_quantization(model)
    
    judgments = []
    win_count = 0
    loss_count = 0
    tie_count = 0
    
    logger.info("Scoring responses with Local Reward Model...")
    for item in tqdm(outputs):
        base_score = get_reward_score(model, tokenizer, item['prompt'], item['baseline_response'], config['max_length'])
        dpo_score = get_reward_score(model, tokenizer, item['prompt'], item['dpo_response'], config['max_length'])
        
        if dpo_score > base_score:
            decision = "win"
            win_count += 1
        elif dpo_score < base_score:
            decision = "loss"
            loss_count += 1
        else:
            decision = "tie"
            tie_count += 1
            
        judgments.append({
            "prompt": item['prompt'],
            "baseline_response": item['baseline_response'],
            "dpo_response": item['dpo_response'],
            "baseline_reward": base_score,
            "dpo_reward": dpo_score,
            "winner": decision
        })
        
    with open(output_dir / "reward_eval_outputs.json", "w") as f:
        json.dump(judgments, f, indent=2)
        
    logger.info(f"Judgments saved to {output_dir / 'reward_eval_outputs.json'}")
    logger.info(f"Win: {win_count}, Loss: {loss_count}, Tie: {tie_count}")

if __name__ == "__main__":
    run_reward_eval()
