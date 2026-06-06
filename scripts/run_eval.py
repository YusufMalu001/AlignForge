import argparse
import logging
from tracking.logger import load_config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_multi_eval(models_list: list):
    logger.info(f"Running multi-model evaluation for: {models_list}")
    # Placeholder for the actual evaluation logic
    # Real implementation would load dataset, generate responses for each model,
    # evaluate pairwise with Judge, and output leaderboard + significance_report.json
    
    logger.info("Evaluation pipeline skeleton ready.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True, help="Comma separated list of models e.g., base,sft,dpo,orpo")
    args = parser.parse_args()
    
    models = [m.strip() for m in args.models.split(',')]
    run_multi_eval(models)
