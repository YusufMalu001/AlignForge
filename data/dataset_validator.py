import json
import csv
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DatasetValidator:
    def __init__(self, max_length: int):
        self.max_length = max_length
        self.errors = {
            "empty_prompts": 0,
            "empty_chosen": 0,
            "empty_rejected": 0,
            "malformed": 0,
            "duplicates": 0,
            "truncated": 0
        }
        self.seen_prompts = set()
        
    def validate(self, dataset, tokenizer):
        total = len(dataset)
        logger.info(f"Validating dataset of size {total}")
        
        for item in dataset:
            prompt = item.get('prompt', '').strip()
            chosen = item.get('chosen', '').strip()
            rejected = item.get('rejected', '').strip()
            
            if not prompt: self.errors["empty_prompts"] += 1
            if not chosen: self.errors["empty_chosen"] += 1
            if not rejected: self.errors["empty_rejected"] += 1
            
            if "\n\nHuman:" not in item.get('prompt', '') and "Human:" not in item.get('prompt', ''):
                self.errors["malformed"] += 1
                
            if prompt in self.seen_prompts:
                self.errors["duplicates"] += 1
            self.seen_prompts.add(prompt)
            
            # Rough check for truncation
            tokens = len(tokenizer(prompt + chosen)['input_ids'])
            if tokens > self.max_length:
                self.errors["truncated"] += 1

        self._export_reports(total)
        
        # Fail if truncation > 20% or empty prompts > 0
        if self.errors["empty_prompts"] > 0:
            raise ValueError(f"Validation Failed: {self.errors['empty_prompts']} empty prompts found.")
        if self.errors["truncated"] / total > 0.20:
            raise ValueError(f"Validation Failed: High truncation rate ({(self.errors['truncated']/total)*100:.1f}%).")
            
    def _export_reports(self, total: int):
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        report = {k: {"count": v, "percentage": round((v/total)*100, 2)} for k, v in self.errors.items()}
        report["total_samples"] = total
        
        with open(output_dir / "dataset_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        with open(output_dir / "dataset_report.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Issue", "Count", "Percentage"])
            for k, v in self.errors.items():
                writer.writerow([k, v, f"{(v/total)*100:.2f}%"])
        logger.info("Saved dataset validation reports to results/")
