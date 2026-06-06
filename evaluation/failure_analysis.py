import json
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class FailureAnalyzer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def detect_refusal(self, text: str) -> bool:
        refusal_phrases = ["i cannot", "i apologize", "as an ai", "i'm sorry", "i am sorry"]
        return any(phrase in text.lower() for phrase in refusal_phrases)
        
    def detect_repetition(self, text: str) -> bool:
        words = text.lower().split()
        if len(words) < 10: return False
        counts = Counter(words)
        # If any word makes up more than 15% of a long response (excluding stopwords usually, but this is naive)
        most_common = counts.most_common(1)[0]
        if most_common[1] / len(words) > 0.15 and len(most_common[0]) > 3:
            return True
        return False
        
    def detect_length_bias(self, base_text: str, tuned_text: str) -> bool:
        """Flags if the tuned model is just writing significantly longer responses (reward hacking)."""
        base_len = len(base_text.split())
        tuned_len = len(tuned_text.split())
        return tuned_len > (base_len * 1.5) and tuned_len > 50

    def analyze(self, judgments: list, output_dir: str):
        logger.info("Running Failure Analysis...")
        failures = {
            "refusals": 0,
            "repetitions": 0,
            "length_bias": 0,
            "flagged_samples": []
        }
        
        for j in judgments:
            dpo_resp = j['dpo_response']
            base_resp = j['baseline_response']
            
            refused = self.detect_refusal(dpo_resp)
            repeated = self.detect_repetition(dpo_resp)
            len_bias = self.detect_length_bias(base_resp, dpo_resp)
            
            if refused: failures["refusals"] += 1
            if repeated: failures["repetitions"] += 1
            if len_bias: failures["length_bias"] += 1
            
            if refused or repeated or len_bias:
                failures["flagged_samples"].append({
                    "prompt": j["prompt"][:100] + "...",
                    "issues": [
                        "refusal" if refused else None,
                        "repetition" if repeated else None,
                        "length_bias" if len_bias else None
                    ]
                })
                
        out_path = Path(output_dir) / "failure_report.json"
        with open(out_path, "w") as f:
            json.dump(failures, f, indent=4)
            
        logger.info(f"Failure Analysis complete. Saved to {out_path}")
