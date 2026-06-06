import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DatasetAnalyzer:
    def analyze(self, dataset, tokenizer):
        logger.info("Analyzing dataset statistics...")
        prompt_lens = []
        chosen_lens = []
        rejected_lens = []
        
        for item in dataset:
            prompt_lens.append(len(tokenizer(item['prompt'])['input_ids']))
            chosen_lens.append(len(tokenizer(item['chosen'])['input_ids']))
            rejected_lens.append(len(tokenizer(item['rejected'])['input_ids']))
            
        stats = {
            "prompt_length": {
                "mean": np.mean(prompt_lens),
                "median": np.median(prompt_lens),
                "p95": np.percentile(prompt_lens, 95)
            },
            "chosen_length": {
                "mean": np.mean(chosen_lens),
                "median": np.median(chosen_lens),
                "p95": np.percentile(chosen_lens, 95)
            },
            "rejected_length": {
                "mean": np.mean(rejected_lens),
                "median": np.median(rejected_lens),
                "p95": np.percentile(rejected_lens, 95)
            }
        }
        
        for k, v in stats.items():
            logger.info(f"{k} -> Mean: {v['mean']:.1f}, Median: {v['median']:.1f}, P95: {v['p95']:.1f}")
            
        return stats
