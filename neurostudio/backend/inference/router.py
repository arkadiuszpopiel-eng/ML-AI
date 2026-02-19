"""
Model Router - selects the best model for a given task based on semantic similarity.

Two scoring strategies:
1. Embedding-based (default) - uses TF-IDF vectors for semantic classification
2. Keyword fallback - regex patterns when TF-IDF is not available
"""
import re
import math
import logging
from collections import Counter
from ..config import load_config
from .model_manager import list_models, get_model_path

logger = logging.getLogger("neurostudio.router")

# ──── Task type definitions with representative phrases ────

TASK_PROFILES: dict[str, list[str]] = {
    "coding": [
        "write code", "implement function", "debug error", "fix bug",
        "refactor class", "python script", "javascript code", "html css",
        "compile build test", "git commit push", "npm pip install",
        "api endpoint rest", "database sql query", "algorithm sort search",
        "code review", "unit test", "deploy docker", "regex pattern",
        "variable function class method", "programming development software",
        "typescript rust go java cpp",
    ],
    "analysis": [
        "analyze data", "explain concept", "describe mechanism", "summarize text",
        "review evaluate compare", "statistics metrics chart graph",
        "research study investigate", "performance benchmark profiling",
        "data trend pattern insight", "examine assess report",
        "pros cons advantages disadvantages", "how does it work",
        "what is the difference between", "explain the architecture",
    ],
    "creative": [
        "write story", "compose poem", "draft essay article blog",
        "creative writing fiction narrative", "dialog dialogue screenplay",
        "email letter message report", "content marketing copy",
        "brainstorm ideas", "imagine scenario", "rewrite rephrase paraphrase",
        "translate text", "write in style of", "generate description",
    ],
    "chat": [
        "hello hi hey good morning", "how are you", "what is your name",
        "tell me about yourself", "help me with", "thank you thanks",
        "what do you think", "recommend suggest opinion",
        "can you please", "I need help", "simple question",
        "yes no maybe", "goodbye bye see you",
    ],
}

# Keyword patterns (fallback if embedding scoring is unavailable)
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


# ──── TF-IDF Semantic Scorer ────

class SemanticScorer:
    """Lightweight TF-IDF based intent classifier. No external dependencies."""

    def __init__(self):
        self._idf: dict[str, float] = {}
        self._profile_vectors: dict[str, dict[str, float]] = {}
        self._built = False

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r'[a-zA-Z\u0100-\u017F]+', text.lower())

    def _compute_tf(self, tokens: list[str]) -> dict[str, float]:
        """Term frequency (normalized)."""
        counts = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {t: c / total for t, c in counts.items()}

    def build(self, profiles: dict[str, list[str]]):
        """Build TF-IDF vectors for each task type profile."""
        # Collect all documents
        all_docs: list[list[str]] = []
        profile_tokens: dict[str, list[str]] = {}

        for task_type, phrases in profiles.items():
            combined = " ".join(phrases)
            tokens = self._tokenize(combined)
            profile_tokens[task_type] = tokens
            all_docs.append(tokens)

        # Compute IDF
        num_docs = len(all_docs)
        all_terms = set()
        for doc in all_docs:
            all_terms.update(doc)

        for term in all_terms:
            doc_freq = sum(1 for doc in all_docs if term in doc)
            self._idf[term] = math.log((num_docs + 1) / (doc_freq + 1)) + 1

        # Build profile vectors (TF-IDF)
        for task_type, tokens in profile_tokens.items():
            tf = self._compute_tf(tokens)
            self._profile_vectors[task_type] = {
                t: tf_val * self._idf.get(t, 1.0)
                for t, tf_val in tf.items()
            }

        self._built = True
        logger.info("SemanticScorer built with %d profiles, %d terms", len(profiles), len(all_terms))

    def score(self, message: str) -> dict[str, float]:
        """Score a message against all task type profiles using cosine similarity."""
        if not self._built:
            self.build(TASK_PROFILES)

        tokens = self._tokenize(message)
        if not tokens:
            return {t: 0.0 for t in self._profile_vectors}

        # Build message TF-IDF vector
        tf = self._compute_tf(tokens)
        msg_vec = {t: tf_val * self._idf.get(t, 1.0) for t, tf_val in tf.items()}

        # Cosine similarity with each profile
        scores = {}
        msg_magnitude = math.sqrt(sum(v * v for v in msg_vec.values())) or 1.0

        for task_type, profile_vec in self._profile_vectors.items():
            dot = sum(msg_vec.get(t, 0.0) * v for t, v in profile_vec.items())
            prof_magnitude = math.sqrt(sum(v * v for v in profile_vec.values())) or 1.0
            scores[task_type] = dot / (msg_magnitude * prof_magnitude)

        return scores


# Singleton scorer
_scorer = SemanticScorer()


def detect_task_type(message: str) -> str:
    """Detect the type of task from the user's message using semantic scoring."""
    scores = _scorer.score(message)

    best_type = max(scores, key=scores.get) if scores else "chat"
    best_score = scores.get(best_type, 0.0)

    # If semantic score is too low, fall back to keyword matching
    if best_score < 0.05:
        best_type = _detect_task_type_keywords(message)

    logger.debug("Task detection: %s (scores: %s)", best_type, {k: f"{v:.3f}" for k, v in scores.items()})
    return best_type


def get_task_scores(message: str) -> dict[str, float]:
    """Get all task type scores for a message. Used by the API for transparency."""
    return _scorer.score(message)


def _detect_task_type_keywords(message: str) -> str:
    """Fallback: detect task type using keyword patterns."""
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


def get_router_config() -> dict:
    """Get current router configuration with available models info."""
    config = load_config()
    router_config = config.get("router", {})
    models = list_models()

    return {
        "enabled": router_config.get("enabled", False),
        "assignments": {
            "coding": router_config.get("coding"),
            "analysis": router_config.get("analysis"),
            "creative": router_config.get("creative"),
            "chat": router_config.get("chat"),
        },
        "available_models": [m.to_dict() for m in models],
        "task_types": ["coding", "analysis", "creative", "chat"],
    }


def update_router_config(enabled: bool | None = None, assignments: dict | None = None) -> dict:
    """Update router configuration and save to config.yaml."""
    from ..config import save_config

    config = load_config()
    if "router" not in config:
        config["router"] = {}

    if enabled is not None:
        config["router"]["enabled"] = enabled

    if assignments:
        for task_type in ("coding", "analysis", "creative", "chat"):
            if task_type in assignments:
                config["router"][task_type] = assignments[task_type]

    save_config(config)
    logger.info("Router config updated: enabled=%s, assignments=%s", config["router"].get("enabled"), assignments)
    return get_router_config()
