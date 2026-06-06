import logging
from tracking.logger import load_config
from .local_judge import LocalJudge
from .openai_judge import OpenAIJudge
from .openai_compatible_judge import OpenAICompatibleJudge

logger = logging.getLogger(__name__)

class PairwiseJudge:
    def __init__(self):
        config = load_config()
        judge_cfg = config.get("judge", {})
        judge_type = judge_cfg.get("type", "local")
        
        if judge_type == "local":
            model = judge_cfg.get("local", {}).get("model", "Qwen/Qwen2-0.5B-Instruct")
            logger.info(f"Initializing Local Judge with model {model}")
            self.backend = LocalJudge(model)
        elif judge_type == "openai":
            model = judge_cfg.get("openai", {}).get("model", "gpt-4o-mini")
            logger.info(f"Initializing OpenAI Judge with model {model}")
            self.backend = OpenAIJudge(model)
        elif judge_type == "openai_compatible":
            base_url = judge_cfg.get("openai_compatible", {}).get("base_url")
            api_key = judge_cfg.get("openai_compatible", {}).get("api_key", "")
            logger.info(f"Initializing OpenAI Compatible Judge at {base_url}")
            self.backend = OpenAICompatibleJudge(base_url, api_key)
        else:
            raise ValueError(f"Unknown judge type: {judge_type}")

    def compare(self, prompt: str, base_resp: str, tuned_resp: str) -> str:
        """Compares base vs tuned. Returns 'win' (tuned wins), 'loss' (base wins), or 'tie'."""
        verdict = self.backend.evaluate_pairwise(prompt, base_resp, tuned_resp)
        if verdict == "A": return "loss"  # A is base
        if verdict == "B": return "win"   # B is tuned
        return "tie"
