import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class UnifiedFormatter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos = tokenizer.eos_token if tokenizer.eos_token else ""

    def format_sft(self, prompt: str, chosen: str) -> str:
        """SFT: prompt + chosen + EOS"""
        return f"{prompt} {chosen}{self.eos}"

    def format_reward(self, prompt: str, chosen: str, rejected: str):
        """Reward: prompt + chosen, prompt + rejected"""
        return {
            "chosen_text": f"{prompt} {chosen}{self.eos}",
            "rejected_text": f"{prompt} {rejected}{self.eos}"
        }

    def format_dpo(self, prompt: str, chosen: str, rejected: str):
        """DPO: TRL DPOTrainer expects prompt, chosen, rejected columns natively."""
        return {
            "prompt": prompt,
            "chosen": f" {chosen}{self.eos}",
            "rejected": f" {rejected}{self.eos}"
        }

    def format_eval(self, prompt: str) -> str:
        """Evaluation: prompt only"""
        return prompt
