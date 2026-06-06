import json
import random
import os
from pathlib import Path

def generate_mock_judgments():
    os.makedirs("results/smoke_test", exist_ok=True)
    
    judgments = []
    for i in range(50): # 50 samples
        # Base/SFT responses (shorter, lower reward)
        base_len = random.randint(15, 30)
        base_reward = random.uniform(0.1, 1.2) + (base_len * 0.02)
        
        # DPO responses (longer, higher reward, hacking the RM)
        dpo_len = random.randint(30, 80)
        # Strong correlation injected!
        dpo_reward = random.uniform(1.0, 1.5) + (dpo_len * 0.05)
        
        judgments.append({
            "prompt": f"Prompt {i}",
            "baseline_response": "word " * base_len,
            "baseline_reward": base_reward,
            "dpo_response": "word " * dpo_len,
            "dpo_reward": dpo_reward,
            "orpo_response": "word " * int(dpo_len * 0.8),
            "orpo_reward": dpo_reward * 0.85
        })
        
    with open("results/smoke_test/judgments.json", "w") as f:
        json.dump(judgments, f, indent=4)
        
if __name__ == "__main__":
    generate_mock_judgments()
