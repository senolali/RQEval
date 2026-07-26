"""API-based model implementation supporting OpenAI-style endpoints."""

import time
from typing import Any, Dict, Optional
from models.base_model import BaseModel
from utils.tokenizer import count_tokens


class APIModel(BaseModel):
    """OpenAI-style API model wrapper with retry, timeout, and rate-limit handling."""

    def __init__(
        self,
        name: str,
        api_key: str,
        model_id: str,
        config: Dict[str, Any],
        base_url: Optional[str] = None,
        deterministic: bool = True,
        max_retries: int = 3,
        timeout: int = 60,
        max_tokens: int = 512,
        request_delay: float = 1.0,   # conservative default
    ):
        super().__init__(name=name, config=config, deterministic=deterministic)
        self.api_key       = api_key
        self.model_id      = model_id
        self.base_url      = base_url
        self.max_retries   = max_retries
        self.timeout       = timeout
        self.max_tokens    = max_tokens
        self.request_delay = request_delay
        self._last_call_ts = 0.0

    def _build_client(self):
        """Build OpenAI client."""
        try:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            return OpenAI(**kwargs)
        except ImportError:
            raise ImportError("openai library is required for APIModel.")

    def _pace(self):
        remaining = self.request_delay - (time.time() - self._last_call_ts)
        if remaining > 0:
            time.sleep(remaining)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text via API with retry and rate-limit handling."""
        cache_key = self._cache_key(prompt, **kwargs)
        cached    = self._get_cached(cache_key)
        if cached is not None:
            return cached

        client  = self._build_client()
        backoff = 60

        for attempt in range(self.max_retries):
            self._pace()
            try:
                self._last_call_ts = time.time()
                params = {
                    "model":      self.model_id,
                    "messages":   [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                }
                if self.deterministic:
                    params["temperature"] = 0.0

                response = client.chat.completions.create(**params)
                result   = response.choices[0].message.content
                self._set_reported_output_tokens(
                    getattr(getattr(response, "usage", None),
                            "completion_tokens", None)
                )
                self._set_cached(cache_key, result)
                return result

            except Exception as e:
                error_str = str(e).lower()
                if "rate_limit" in error_str or "429" in error_str:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 300)
                elif "timeout" in error_str:
                    time.sleep(5)
                elif attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"API call failed after {self.max_retries} retries. Last error: {e}"
                    )

        raise RuntimeError(f"API call failed after {self.max_retries} retries.")

    def generate_with_trace(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response with trace metadata."""
        start    = time.time()
        response = self.generate(prompt, **kwargs)
        elapsed  = time.time() - start
        steps    = [s.strip() for s in response.split(".") if s.strip()]
        return {
            "response":        response,
            "token_count":     self._take_output_token_count(response),
            "latency":         elapsed,
            "reasoning_steps": steps,
            "model":           self.name,
        }

    def _build_client(self):
        """Build OpenAI client."""
        try:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            return OpenAI(**kwargs)
        except ImportError:
            raise ImportError("openai library is required for APIModel.")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text via API with retry and rate-limit handling."""
        cache_key = self._cache_key(prompt, **kwargs)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        client = self._build_client()
        attempt = 0
        backoff = 2

        while attempt < self.max_retries:
            try:
                params = {
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                }
                if self.deterministic:
                    params["temperature"] = 0.0

                response = client.chat.completions.create(**params)
                result = response.choices[0].message.content
                self._set_reported_output_tokens(
                    getattr(getattr(response, "usage", None),
                            "completion_tokens", None)
                )
                self._set_cached(cache_key, result)
                return result

            except Exception as e:
                error_str = str(e).lower()
                if "rate_limit" in error_str or "429" in error_str:
                    time.sleep(backoff)
                    backoff *= 2
                elif "timeout" in error_str:
                    time.sleep(1)
                else:
                    raise
                attempt += 1

        raise RuntimeError(f"API call failed after {self.max_retries} retries.")

    def generate_with_trace(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response with trace metadata."""
        start = time.time()
        response = self.generate(prompt, **kwargs)
        elapsed = time.time() - start

        token_count = self._take_output_token_count(response)
        steps = [s.strip() for s in response.split(".") if s.strip()]

        return {
            "response": response,
            "token_count": token_count,
            "latency": elapsed,
            "reasoning_steps": steps,
            "model": self.name,
        }
