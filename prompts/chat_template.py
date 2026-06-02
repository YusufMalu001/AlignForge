"""
Centralized prompt templates for the AlignForge project.
Ensures consistency across SFT, DPO, and evaluation inference.
"""

def get_system_prompt() -> str:
    """Returns the default system prompt."""
    return "You are a helpful, harmless, and honest assistant."

def format_prompt(user_input: str) -> str:
    """
    Formats the user input with the system prompt and chat template.
    """
    system_prompt = get_system_prompt()
    
    return f"<|system|>\n{system_prompt}\n\n<|user|>\n{user_input}\n\n<|assistant|>\n"

def format_chat(user_input: str, response: str) -> str:
    """
    Formats a complete turn (user + assistant).
    """
    prompt = format_prompt(user_input)
    return f"{prompt}{response}"
