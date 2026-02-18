"""
Model Router - selects the best model for a given task based on keywords and configuration.
"""
import re
import logging
from ..config import load_config
from .model_manager import list_models, get_model_path

logger = logging.getLogger("neurostudio.router")

# Keyword patterns for task type detection
TASK_PATTERNS = {
    "coding": [
        r'\b(code|program|script|function|class|debug|fix bug|implement|refactor)\b',
        r'\b(python|javascript|typescript|rust|go|java|c\+\+|html|css)\b',
        r'\b(compile|build|test|deploy|git|npm|pip|cargo)\b',
        r'\b(api|endpoint|database|sql|query|schema)\b',
        r'\b(algorith|data structure|sort|search|regex)\b',
    ],
    "analysis": [
        r'\b(analy[sz]|explain|describe|summariz|review|evaluat|compar)\b',
        r'\b(data|statistics|metrics|chart|graph|trend|pattern)\b',
        r'\b(research|study|investigat|examin|assess)\b',
        r'\b(performance|benchmark|profil|optimiz)\b',
    ],
    "creative": [
        r'\b(write|story|poem|essay|article|blog|content)\b',
        r'\b(creative|imagin|fiction|narrative|dialog)\b',
        r'\b(email|letter|message|report|document)\b',
    ],
    "chat": [
        r'\b(hello|hi|hey|what is|who is|how do|tell me|explain)\b',
        r'\b(help|question|opinion|think|recommend)\b',
    ],
}


def detect_task_type(message: str) -> str:
    """Detect the type of task from the user's message."""
    message_lower = message.lower()
    scores = {}

    for task_type, patterns in TASK_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = re.findall(pattern, message_lower)
            score += len(matches)
        scores[task_type] = score

    if not scores or max(scores.values()) == 0:
        return "chat"

    return max(scores, key=scores.get)


def select_model(message: str) -> str | None:
    """
    Select the best model for the given message.
    Returns model path or None if no routing is configured.
    """
    config = load_config()
    router_config = config.get("router", {})

    if not router_config.get("enabled", False):
        return None

    task_type = detect_task_type(message)
    logger.info("Detected task type: %s", task_type)

    # Check if a model is assigned for this task type
    model_filename = router_config.get(task_type)
    if model_filename:
        model_path = get_model_path(model_filename)
        if model_path:
            logger.info("Router selected model: %s for task: %s", model_filename, task_type)
            return model_path
        else:
            logger.warning("Configured model not found: %s", model_filename)

    return None
