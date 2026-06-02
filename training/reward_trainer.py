"""
Stage 2: Reward Model Training with TRL.
"""
import argparse
from pathlib import Path
import logging
from transformers import TrainingArguments
from trl import RewardTrainer, RewardConfig
from peft import PeftModel

from tracking.logger import init_tracking, load_config, finish_tracking
from data.loader import prepare_dataset
from models.base_loader import get_reward_model_and_tokenizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def preprocess_function(examples, tokenizer, max_length):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        # Tokenize chosen
        tokenized_chosen = tokenizer(
            chosen,
            truncation=True,
            max_length=max_length,
        )
        # Tokenize rejected
        tokenized_rejected = tokenizer(
            rejected,
            truncation=True,
            max_length=max_length,
        )

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

def run_rm(resume_from_checkpoint: bool = False, max_samples: int = None):
    config = load_config()
    if max_samples is not None:
        config['num_train_samples'] = max_samples
        config['num_eval_samples'] = min(config.get('num_eval_samples', 50), max_samples)
        
    init_tracking(config['project_name'], "rm", config)
    
    train_ds, eval_ds = prepare_dataset(config)
    
    # RM starts from SFT checkpoint if available
    sft_checkpoint_path = Path(config['output_dir']) / "sft_checkpoint"
    if not (sft_checkpoint_path / "adapter_config.json").exists():
        logger.warning(f"Valid SFT checkpoint (adapter_config.json) not found at {sft_checkpoint_path}. Starting RM from base model.")
        sft_checkpoint_path = None
    else:
        sft_checkpoint_path = str(sft_checkpoint_path)
        
    model, tokenizer = get_reward_model_and_tokenizer(
        config['model_name'], 
        lora_config=config,
        is_trainable=True,
        checkpoint_path=sft_checkpoint_path
    )
    
    # Preprocess dataset for RewardTrainer
    train_ds = train_ds.map(
        lambda x: preprocess_function(x, tokenizer, config['max_length']),
        batched=True,
        remove_columns=train_ds.column_names
    )
    eval_ds = eval_ds.map(
        lambda x: preprocess_function(x, tokenizer, config['max_length']),
        batched=True,
        remove_columns=eval_ds.column_names
    )
    
    output_dir = Path(config['output_dir']) / "rm_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = RewardConfig(
        output_dir=str(output_dir),
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
        fp16=False,
        bf16=False,
        max_length=config['max_length'],
    )
    
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    
    logger.info("Starting Reward Model training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    logger.info(f"Saving final RM checkpoint to {output_dir}")
    trainer.save_model(str(output_dir))
    finish_tracking()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max_samples", type=int, default=None, help="Override number of training samples")
    args = parser.parse_args()
    run_rm(args.resume, args.max_samples)
