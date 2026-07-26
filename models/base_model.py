"""Abstract base class for all LLM models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import hashlib
import json
import threading


class BaseModel(ABC):
    """Abstract base class defining the interface for all LLM models."""

    def __init__(self, name: str, config: Dict[str, Any], deterministic: bool = True):
        self.name = name
        self.config = config
        self.deterministic = deterministic
        self._cache: Dict[str, str] = {}
        self._token_cache: Dict[str, int] = {}
        self._usage_local = threading.local()

    def _set_reported_output_tokens(self, value: Any) -> None:
        """Store provider-reported output tokens for the current worker."""
        try:
            count = int(value)
            self._usage_local.output_tokens = count if count >= 0 else None
        except (TypeError, ValueError):
            self._usage_local.output_tokens = None

    def _take_output_token_count(self, text: str) -> int:
        """Consume exact provider usage, falling back to shared BPE count."""
        from utils.tokenizer import count_tokens

        count = getattr(self._usage_local, "output_tokens", None)
        self._usage_local.output_tokens = None
        return count if count is not None else count_tokens(text)

    def _cache_key(self, prompt: str, **kwargs) -> str:
        # Only called when deterministic=True — safe to hash
        payload = json.dumps({"prompt": prompt, **kwargs}, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()

    def _get_cached(self, key: Optional[str]) -> Optional[str]:
        if key is None or not self.deterministic:
            return None
        value = self._cache.get(key)
        if value is not None:
            self._set_reported_output_tokens(self._token_cache.get(key))
        return value

    def _set_cached(self, key: Optional[str], value: str) -> None:
        if key is not None and self.deterministic:
            self._cache[key] = value
            count = getattr(self._usage_local, "output_tokens", None)
            if count is not None:
                self._token_cache[key] = count

    def _maybe_cache_key(self, prompt: str, **kwargs) -> Optional[str]:
        """Return cache key only if caching is active (deterministic mode)."""
        if not self.deterministic:
            return None          # skip md5 computation entirely
        return self._cache_key(prompt, **kwargs)

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt."""
        pass

    @abstractmethod
    def generate_with_trace(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response along with reasoning trace metadata."""
        pass

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for a batch of prompts."""
        return [self.generate(p, **kwargs) for p in prompts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
