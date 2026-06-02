"""
Stage 2: Direct Preference Optimization (DPO) with TRL.
"""
import argparse
from pathlib import Path
import logging
import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

from tracking.logger import init_tracking, load_config, finish_tracking
from data.loader import prepare_dataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_dpo(resume_from_checkpoint: bool = False, max_samples: int = None):
    config = load_config()
    if max_samples is not None:
        config['num_train_samples'] = max_samples
        config['num_eval_samples'] = min(config.get('num_eval_samples', 50), max_samples)
        
    init_tracking(config['project_name'], "dpo", config)
    
    train_ds, eval_ds = prepare_dataset(config)
    
    sft_checkpoint_path = str(Path(config['output_dir']) / "sft_checkpoint")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    
    if Path(sft_checkpoint_path).exists():
        logger.info(f"Merging SFT adapters from {sft_checkpoint_path}")
        base_model = PeftModel.from_pretrained(base_model, sft_checkpoint_path).merge_and_unload()
    else:
        logger.warning("No SFT checkpoint found! Using base model for DPO.")

    logger.info("Creating reference model...")
    ref_model = copy.deepcopy(base_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    logger.info("Applying LoRA for DPO policy model...")
    peft_config = LoraConfig(
        r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        lora_dropout=config.get('lora_dropout', 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    policy_model = get_peft_model(base_model, peft_config)
    policy_model.print_trainable_parameters()
    
    output_dir = Path(config['output_dir']) / "dpo_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = DPOConfig(
        output_dir=str(output_dir),
        beta=config['beta'],
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
        report_to="wandb",
        max_prompt_length=config['max_length'] // 2,
        max_length=config['max_length']
    )
    
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    logger.info("Starting DPO training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    logger.info(f"Saving final DPO checkpoint to {output_dir}")
    trainer.save_model(str(output_dir))
    finish_tracking()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    run_dpo(args.resume)
