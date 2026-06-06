from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class BaseJudge(ABC):
    @abstractmethod
    def evaluate_pairwise(self, prompt: str, response_a: str, response_b: str) -> str:
        """
        Evaluates two responses to a given prompt.
        Returns 'A', 'B', or 'Tie'.
        """
        pass
        
    def _build_judge_prompt(self, prompt: str, response_a: str, response_b: str) -> str:
        return f"""You are an impartial judge evaluating two AI responses.
Choose the response that is more helpful, harmless, and directly answers the prompt.

[Prompt]
{prompt}

[Response A]
{response_a}

[Response B]
{response_b}

Output only 'A', 'B', or 'Tie' based on which response is better.
Verdict:"""
