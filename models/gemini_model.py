"""Google Gemini API model wrapper — uses google-genai SDK (v1+).

Rate-limit strategy
-------------------
Instead of hammering the API and reacting to 429s with exponential backoff
(which was causing 62s wasted per call and empty responses), we proactively
pace requests using a minimum inter-request delay derived from the API tier's
RPM limit.

  tier='free'  →  15 RPM  →  4.4 s/request  (975 items × 6 calls ≈  6.5 h)
  tier='paid'  → 500 RPM  →  0.13 s/request (975 items × 6 calls ≈ ~12 min)

Set tier='paid' in your config if you have a billing account.
"""

import time
import logging
from typing import Any, Dict, Optional
from models.base_model import BaseModel
from utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)

# Conservative safe RPM per tier (Gemini 2.0 Flash)
_TIER_RPM: Dict[str, int] = {
    "free":  15,   # Free tier hard limit
    "paid": 500,   # Pay-as-you-go — conservative; actual limit is higher
}


class GeminiModel(BaseModel):
    """Google Gemini API wrapper using the new google-genai SDK."""

    def __init__(
        self,
        name: str,
        api_key: str,
        model_id: str,
        config: Dict[str, Any],
        deterministic: bool = True,
        temperature: float = None,
        max_retries: int = 3,          # reduced: proactive pacing makes retries rare
        timeout: int = 60,
        max_tokens: int = 512,
        request_delay: float = None,   # None → auto-computed from tier
        tier: str = "paid",            # "free" or "paid"
    ):
        super().__init__(name=name, config=config, deterministic=deterministic)
        self._temperature  = temperature
        self.api_key       = api_key
        self.model_id      = model_id
        self.max_retries   = max_retries
        self.timeout       = timeout
        self.max_tokens    = max_tokens
        self.tier          = tier
        self._client       = None
        self._last_call_ts = 0.0   # wall-clock time of last API call

        # Auto-compute safe delay from tier RPM + 10% safety margin.
        # This is the PRIMARY defence against rate limits.
        rpm = _TIER_RPM.get(tier, _TIER_RPM["free"])
        self.request_delay = request_delay if request_delay is not None \
                             else (60.0 / rpm) * 1.1

        logger.info(
            f"[{name}] tier={tier} | "
            f"request_delay={self.request_delay:.2f}s | "
            f"effective rate ~{60/self.request_delay:.0f} RPM"
        )

    def _build_client(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            return self._client
        except ImportError:
            raise ImportError(
                "Run: pip install google-genai  &&  "
                "pip uninstall google-generativeai -y"
            )

    def _generation_config(self):
        try:
            from google.genai import types
            return types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                temperature=self._temperature if self._temperature is not None
                            else (0.0 if self.deterministic else 0.7),
            )
        except ImportError:
            return None

    def _pace(self):
        """Block until self.request_delay seconds have passed since last call.
        This proactive pacing is the primary defence against rate limits."""
        elapsed = time.time() - self._last_call_ts
        remaining = self.request_delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def generate(self, prompt: str, **kwargs) -> str:
        # In-memory cache: skip API call for identical prompts (deterministic mode)
        cache_key = self._maybe_cache_key(prompt, **kwargs)
        cached    = self._get_cached(cache_key)
        if cached is not None:
            return cached

        client  = self._build_client()
        config  = self._generation_config()
        backoff = 60   # only used when proactive pacing still wasn't enough

        for attempt in range(self.max_retries):
            # Proactive pacing before EVERY call
            self._pace()

            try:
                self._last_call_ts = time.time()
                response = client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=config,
                )
                result = response.text.strip()
                usage = getattr(response, "usage_metadata", None)
                self._set_reported_output_tokens(
                    getattr(usage, "candidates_token_count", None)
                    or getattr(usage, "output_token_count", None)
                )
                self._set_cached(cache_key, result)
                return result

            except Exception as e:
                err = str(e).lower()

                if "quota" in err or "429" in err or "resource_exhausted" in err:
                    # Rate limit despite proactive pacing — back off longer
                    logger.warning(
                        f"[{self.name}] Rate limit hit (attempt {attempt+1}/"
                        f"{self.max_retries}) — waiting {backoff}s. "
                        f"Consider upgrading tier or increasing request_delay."
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 300)   # cap at 5 min

                elif "billing" in err or "permission" in err:
                    # Unrecoverable — raise so the caller knows immediately
                    raise RuntimeError(f"[{self.name}] Billing/permission error: {e}")

                elif attempt < self.max_retries - 1:
                    logger.warning(
                        f"[{self.name}] Transient error (attempt {attempt+1}): "
                        f"{e} — retrying in 5s..."
                    )
                    time.sleep(5)

                else:
                    # All retries exhausted — raise so evaluator can skip/flag
                    raise RuntimeError(
                        f"[{self.name}] All {self.max_retries} retries failed. "
                        f"Last error: {e}"
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
