"""Mock model for testing the framework without real LLM endpoints."""

import random
import time
from typing import Any, Dict
from models.base_model import BaseModel
from utils.tokenizer import count_tokens


REASONING_TEMPLATES = [
    "Let me think step by step. First, I analyze the problem. Then, I apply the relevant rules. Finally, I arrive at the answer: {answer}.",
    "Step 1: Understand the question. Step 2: Apply logic. Step 3: Compute the result. The answer is {answer}.",
    "Analyzing carefully: The key insight is to break down the problem. Using this approach, the result is {answer}.",
    "Given the information, I reason as follows: the solution requires systematic thinking. Therefore, the answer is {answer}.",
]

ANSWERS = ["42", "yes", "no", "the answer is correct", "invalid", "true", "false"]


class MockModel(BaseModel):
    """Deterministic mock model for framework testing."""

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        accuracy_level: float = 0.8,
        seed: int = 42,
        deterministic: bool = True,
    ):
        super().__init__(name=name, config=config, deterministic=deterministic)
        self.accuracy_level = accuracy_level
        self.seed = seed
        self._rng = random.Random(seed)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a mock response."""
        cache_key = self._cache_key(prompt, **kwargs)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if self.deterministic:
            rng = random.Random(hash(prompt) % (2**31))
        else:
            rng = self._rng

        answer = rng.choice(ANSWERS)
        template = rng.choice(REASONING_TEMPLATES)
        response = template.format(answer=answer)

        self._set_cached(cache_key, response)
        return response

    def generate_with_trace(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response with trace metadata."""
        start = time.time()
        response = self.generate(prompt, **kwargs)
        elapsed = time.time() - start

        token_count = count_tokens(response)
        steps = [s.strip() for s in response.split(".") if s.strip()]

        return {
            "response": response,
            "token_count": token_count,
            "latency": elapsed,
            "reasoning_steps": steps,
            "model": self.name,
        }
