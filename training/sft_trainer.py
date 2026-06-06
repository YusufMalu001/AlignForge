"""
Stage 1: Supervised Fine-Tuning (SFT) with TRL.
"""
import argparse
from pathlib import Path
import logging
from trl import SFTTrainer, SFTConfig

from tracking.logger import init_tracking, load_config, finish_tracking
from data.loader import prepare_dataset
from models.base_loader import get_model_and_tokenizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_sft(resume_from_checkpoint: bool = False, max_samples: int = None):
    config = load_config()
    if max_samples is not None:
        config['num_train_samples'] = max_samples
        config['num_eval_samples'] = min(config.get('num_eval_samples', 50), max_samples)
        
    init_tracking(config['project_name'], "sft", config)
    
    train_ds, eval_ds = prepare_dataset(config)
    model, tokenizer = get_model_and_tokenizer(
        config['model_name'], 
        lora_config=config,
        is_trainable=True
    )
    
    output_dir = Path(config['output_dir']) / "sft_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = SFTConfig( 
        output_dir=str(output_dir), 
        learning_rate=float(config['learning_rate']), 
        num_train_epochs=config['num_epochs'], 
        per_device_train_batch_size=config['batch_size'], 
        gradient_accumulation_steps=config['gradient_accumulation_steps'], 
        save_strategy=config['save_strategy'], save_steps=config['save_steps'], 
        logging_steps=10, eval_strategy="steps", eval_steps=config['save_steps'], 
        seed=config['seed'], 
        report_to="none", # disable wandb 
        fp16=False, 
        bf16=False,
        dataset_text_field="text",
        max_length=config['max_length']
        ) 

    trainer = SFTTrainer( 
        model=model, 
        train_dataset=train_ds, 
        eval_dataset=eval_ds, 
        processing_class=tokenizer, 
        args=training_args
        )
    
    logger.info("Starting SFT training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    logger.info(f"Saving final SFT checkpoint to {output_dir}")
    trainer.save_model(str(output_dir))
    finish_tracking()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    run_sft(args.resume)
