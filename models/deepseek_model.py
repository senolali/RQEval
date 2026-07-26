"""DeepSeek API model wrapper (OpenAI-compatible endpoint).

Rate-limit strategy: proactive _pace() before every call prevents 429s.
On exhaustion raises RuntimeError — evaluator skips/flags the item instead
of silently returning '' and polluting metrics.

DeepSeek API limit: ~60 RPM (free tier)
"""

import time
import logging
from typing import Any, Dict
from models.base_model import BaseModel
from utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)

_TIER_RPM: Dict[str, int] = {
    "free":  60,
    "paid": 600,
}


class DeepSeekModel(BaseModel):
    """DeepSeek Chat API wrapper using OpenAI-compatible interface."""

    BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        name: str,
        api_key: str,
        model_id: str,
        config: Dict[str, Any],
        base_url: str = None,
        deterministic: bool = True,
        temperature: float = None,
        max_retries: int = 3,
        timeout: int = 60,
        max_tokens: int = 512,
        request_delay: float = None,   # None → auto from tier
        tier: str = "free",
    ):
        self._client       = None
        super().__init__(name=name, config=config, deterministic=deterministic)
        self._temperature  = temperature
        self.api_key       = api_key
        self.model_id      = model_id
        self.base_url      = base_url or self.BASE_URL
        self.max_retries   = max_retries
        self.timeout       = timeout
        self.max_tokens    = max_tokens
        self.tier          = tier
        self._last_call_ts = 0.0

        rpm = _TIER_RPM.get(tier, _TIER_RPM["free"])
        self.request_delay = request_delay if request_delay is not None \
                             else (60.0 / rpm) * 1.1
        logger.info(
            f"[{name}] tier={tier} | request_delay={self.request_delay:.2f}s | "
            f"~{60/self.request_delay:.0f} RPM effective"
        )

    def _build_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            return self._client
        except ImportError:
            raise ImportError("Run: pip install openai")

    def _pace(self):
        remaining = self.request_delay - (time.time() - self._last_call_ts)
        if remaining > 0:
            time.sleep(remaining)

    def generate(self, prompt: str, **kwargs) -> str:
        cache_key = self._maybe_cache_key(prompt, **kwargs)
        cached    = self._get_cached(cache_key)
        if cached is not None:
            return cached

        client  = self._build_client()
        backoff = 60

        for attempt in range(self.max_retries):
            self._pace()
            try:
                self._last_call_ts = time.time()
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self._temperature if self._temperature is not None
                                else (0.0 if self.deterministic else 0.7),
                )
                result = response.choices[0].message.content.strip()
                self._set_reported_output_tokens(
                    getattr(getattr(response, "usage", None),
                            "completion_tokens", None)
                )
                self._set_cached(cache_key, result)
                return result

            except Exception as e:
                err = str(e).lower()
                if "rate_limit" in err or "429" in err:
                    logger.warning(
                        f"[{self.name}] Rate limit (attempt {attempt+1}/{self.max_retries}) "
                        f"— waiting {backoff}s."
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 300)
                elif attempt < self.max_retries - 1:
                    logger.warning(f"[{self.name}] Transient error (attempt {attempt+1}): {e} — retrying in 5s...")
                    time.sleep(5)
                else:
                    raise RuntimeError(
                        f"[{self.name}] All {self.max_retries} retries failed. Last error: {e}"
                    )

        raise RuntimeError(f"[{self.name}] generate() exited retry loop unexpectedly.")

    def generate_with_trace(self, prompt: str, **kwargs) -> Dict[str, Any]:
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
