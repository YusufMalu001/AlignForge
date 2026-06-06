"""
Stage 2: Reward Model Training with TRL.
"""
import argparse
from pathlib import Path
import logging
import numpy as np
import torch
from transformers import TrainingArguments
from trl import RewardTrainer, RewardConfig

from tracking.logger import init_tracking, load_config, finish_tracking
from data.loader import prepare_dataset
from data.formatting import UnifiedFormatter
from models.base_loader import get_reward_model_and_tokenizer
from utils.reproducibility import set_seed, snapshot_experiment
from utils.checkpoint_manager import CheckpointManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    """Computes custom metrics for pairwise ranking."""
    predictions, _ = eval_pred
    
    # predictions is a tuple/array depending on output
    # TRL RewardTrainer outputs logits for chosen and rejected
    if isinstance(predictions, tuple):
        # Trl usually returns a tuple of (chosen_logits, rejected_logits)
        # But depending on version, it might be an array of shape (batch, 2)
        pass
    
    # Standard format for RewardTrainer compute_metrics:
    # eval_pred.predictions usually contains a numpy array of shape (batch_size, 2) [chosen, rejected] or similar.
    # We will compute pairwise accuracy and margin manually
    # Let's assume standard HF formatting where predictions is (chosen_logits, rejected_logits)
    try:
        if isinstance(eval_pred.predictions, tuple):
            chosen_logits, rejected_logits = eval_pred.predictions
        else:
            # If shape is (batch, 2)
            chosen_logits = eval_pred.predictions[:, 0]
            rejected_logits = eval_pred.predictions[:, 1]
    except Exception as e:
        logger.error(f"Could not parse predictions for metrics: {e}")
        return {"accuracy": 0.0, "reward_margin": 0.0}

    accuracy = np.mean(chosen_logits > rejected_logits)
    margin = np.mean(chosen_logits - rejected_logits)
    variance = np.var(np.concatenate([chosen_logits, rejected_logits]))
    
    return {
        "pairwise_accuracy": accuracy,
        "reward_margin": margin,
        "reward_variance": float(variance)
    }

def preprocess_function(examples, tokenizer, max_length):
    formatter = UnifiedFormatter(tokenizer)
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
        "chosen_length": [],
        "rejected_length": [],
    }
    
    for p, c, r in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        # CORRECT IMPLEMENTATION: Tokenize prompt + response
        fmt = formatter.format_reward(p, c, r)
        
        tokenized_chosen = tokenizer(fmt["chosen_text"], truncation=True, max_length=max_length)
        tokenized_rejected = tokenizer(fmt["rejected_text"], truncation=True, max_length=max_length)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        
        # Count non-padding tokens
        new_examples["chosen_length"].append(sum(tokenized_chosen["attention_mask"]))
        new_examples["rejected_length"].append(sum(tokenized_rejected["attention_mask"]))

    return new_examples

import torch.nn as nn
import json

class PenalizedRewardTrainer(RewardTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Load beta_1 dynamically
        beta_1 = 0.0
        try:
            with open("reports/reward_length_regression.json", "r") as f:
                beta_1 = json.load(f).get("beta_1", 0.0)
        except Exception:
            pass
            
        # Get standard outputs
        outputs = model(
            input_ids=inputs.get("input_ids_chosen"),
            attention_mask=inputs.get("attention_mask_chosen"),
            return_dict=True,
        )
        chosen_rewards = outputs.logits
        
        outputs_rejected = model(
            input_ids=inputs.get("input_ids_rejected"),
            attention_mask=inputs.get("attention_mask_rejected"),
            return_dict=True,
        )
        rejected_rewards = outputs_rejected.logits
        
        margin = chosen_rewards - rejected_rewards
        
        # Calculate length bias penalty
        if "chosen_length" in inputs and "rejected_length" in inputs:
            chosen_len = inputs["chosen_length"]
            rejected_len = inputs["rejected_length"]
            length_bias = beta_1 * (chosen_len - rejected_len).to(margin.device, dtype=margin.dtype)
            margin = margin - length_bias
            
        loss = -nn.functional.logsigmoid(margin).mean()
        
        if return_outputs:
            # Reconstruct format TRL expects for metrics
            return loss, (chosen_rewards, rejected_rewards)
        return loss

def run_rm(resume_from_checkpoint: bool = False, max_samples: int = None):
    config = load_config()
    set_seed(config.get("seed", 42))
    snapshot_experiment(config, "reward_training")
    
    if max_samples is not None:
        config['num_train_samples'] = max_samples
        config['num_eval_samples'] = min(config.get('num_eval_samples', 50), max_samples)
        
    init_tracking(config['project_name'], "rm", config)
    
    train_ds, eval_ds = prepare_dataset(config)
    
    sft_checkpoint_path = Path(config['output_dir']) / "sft_checkpoint"
    ckpt_path = str(sft_checkpoint_path) if (sft_checkpoint_path / "adapter_config.json").exists() else None
        
    model, tokenizer = get_reward_model_and_tokenizer(
        config['model_name'], 
        lora_config=config,
        is_trainable=True,
        checkpoint_path=ckpt_path
    )
    
    train_ds = train_ds.map(lambda x: preprocess_function(x, tokenizer, config['max_length']), batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(lambda x: preprocess_function(x, tokenizer, config['max_length']), batched=True, remove_columns=eval_ds.column_names)
    
    # Check if we should use debiased pipeline (if beta_1 > 0 in reports)
    use_debiased = False
    try:
        with open("reports/reward_length_regression.json", "r") as f:
            beta_1 = json.load(f).get("beta_1", 0.0)
            if beta_1 > 0:
                use_debiased = True
    except Exception:
        pass
        
    output_dir_name = "debiased_rm_checkpoint" if use_debiased else "rm_checkpoint"
    output_dir = str(Path(config['output_dir']) / output_dir_name)
    ckpt_manager = CheckpointManager(config['output_dir'])
    
    training_args = RewardConfig(
        output_dir=output_dir,
        learning_rate=float(config['learning_rate']),
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        save_strategy=config['save_strategy'],
        save_steps=config['save_steps'],
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=config['save_steps'],
        seed=config['seed'],
        report_to="none",
        use_cpu=True,
        max_length=config['max_length'],
        center_rewards_coefficient=0.01
    )
    
    TrainerClass = PenalizedRewardTrainer if use_debiased else RewardTrainer
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )
    
    logger.info("Starting Reward Model training...")
    ckpt_manager.resume_training(trainer, output_dir, force_resume=resume_from_checkpoint)
    ckpt_manager.save_checkpoint(trainer, output_dir, "final_rm_checkpoint")
    finish_tracking()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max_samples", type=int, default=None, help="Override number of training samples")
    args = parser.parse_args()
    run_rm(args.resume, args.max_samples)
