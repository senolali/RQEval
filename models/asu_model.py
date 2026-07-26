"""ASU CreateAI gateway model wrapper (REST /query endpoint).

Talks to the ASU AI Acceleration CreateAI platform:
    POST {base_url}/query
    Authorization: Bearer <CREATEAI token>

Payload follows the CreateAI Query endpoint spec:
    endpoint="query", action="query", request_source="override_params",
    model_provider=..., model_name=..., query=<prompt>,
    model_params={temperature, system_prompt, max_tokens},
    response_format={"type": "json"}

Response is parsed from data["response"]["response"] (with fallbacks for
minor gateway format variations).

Rate-limit strategy (ASU default: 750,000 tokens/min per project, service
token). At <=1024-token responses this is far above what the framework
generates, so the practical control is a per-request pacing delay
(request_delay) plus exponential backoff on HTTP 429, per ASU's
"Best Practices" doc. Pacing is thread-safe (evaluator may parallelize).

On retry exhaustion raises RuntimeError — evaluator skips/flags the item
instead of silently returning '' and polluting metrics (same contract as
the other API wrappers).
"""

import time
import json
import logging
import threading
from typing import Any, Dict, Optional
from models.base_model import BaseModel
from utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


class ASUCreateAIModel(BaseModel):
    """ASU CreateAI /query REST endpoint wrapper."""

    DEFAULT_BASE_URL = "https://api-main-poc.aiml.asu.edu"

    def __init__(
        self,
        name: str,
        api_key: str,
        model_name: str,                 # CreateAI model key, e.g. "gpt4o_mini"
        model_provider: str,             # e.g. "openai", "aws", "gcp-deepmind"
        config: Dict[str, Any],
        base_url: Optional[str] = None,
        deterministic: bool = True,
        temperature: Optional[float] = None,
        max_retries: int = 5,
        timeout: int = 120,
        max_tokens: int = 512,
        request_delay: float = 1.0,
        system_prompt: str = "You are a helpful assistant.",
    ):
        super().__init__(name=name, config=config, deterministic=deterministic)
        self._temperature   = temperature
        self.api_key        = api_key
        self.model_name     = model_name
        self.model_provider = model_provider
        self.base_url       = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.max_retries    = max_retries
        self.timeout        = timeout
        self.max_tokens     = max_tokens
        self.request_delay  = request_delay
        self.system_prompt  = system_prompt
        self._last_call_ts  = 0.0
        self._pace_lock     = threading.Lock()

        logger.info(
            f"[{name}] ASU CreateAI | provider={model_provider} "
            f"model={model_name} | base_url={self.base_url} | "
            f"request_delay={request_delay:.2f}s"
        )

    # ------------------------------------------------------------------
    def _pace(self):
        """Thread-safe proactive pacing to stay under the gateway limits."""
        with self._pace_lock:
            now = time.time()
            remaining = self.request_delay - (now - self._last_call_ts)
            if remaining > 0:
                time.sleep(remaining)
            self._last_call_ts = time.time()

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        temperature = (
            self._temperature if self._temperature is not None
            else (0.0 if self.deterministic else 0.7)
        )
        model_params: Dict[str, Any] = {
            "temperature": float(temperature),
            "max_tokens": int(self.max_tokens),
            # ALWAYS send a non-empty system_prompt: without one (or with ""),
            # the CreateAI gateway injects its own 800-3400-token default
            # template, which would contaminate the evaluation. A short neutral
            # prompt suppresses the injection (verified 2026-07: input tokens
            # drop 797->29 for gpt4o_mini, 3426->75 for geminiflash2_5).
            "system_prompt": self.system_prompt or "You are a helpful assistant.",
        }

        return {
            "endpoint": "query",
            "action": "query",
            # Service token: allow per-request parameter overrides.
            "request_source": "override_params",
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "query": prompt,
            "model_params": model_params,
            # Keep the gateway "neutral": plain LLM call, no RAG, no history,
            # no prompt enhancement — so results match a direct API call.
            "enable_search": False,
            "enable_history": False,
        }

    @staticmethod
    def _extract_text(data: Any) -> str:
        """Extract the model text from CreateAI response variants."""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            inner = data.get("response", data)
            if isinstance(inner, str):
                return inner
            if isinstance(inner, dict):
                txt = inner.get("response")
                if isinstance(txt, str):
                    return txt
                # Some deployments return {"response": {"content": "..."}}
                txt = inner.get("content") or inner.get("text")
                if isinstance(txt, str):
                    return txt
        raise ValueError(f"Unexpected CreateAI response format: {str(data)[:300]}")

    @staticmethod
    def _extract_output_tokens(data: Any) -> Optional[int]:
        """Find provider/gateway output-token usage in response variants."""
        output_keys = {
            "completion_tokens",
            "output_tokens",
            "output_token_count",
            "candidates_token_count",
        }
        if isinstance(data, dict):
            for key, value in data.items():
                if key in output_keys:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        pass
            for value in data.values():
                found = ASUCreateAIModel._extract_output_tokens(value)
                if found is not None:
                    return found
        elif isinstance(data, list):
            for value in data:
                found = ASUCreateAIModel._extract_output_tokens(value)
                if found is not None:
                    return found
        return None

    # ------------------------------------------------------------------
    def generate(self, prompt: str, **kwargs) -> str:
        cache_key = self._maybe_cache_key(prompt, **kwargs)
        cached    = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            import requests
        except ImportError:
            raise ImportError("Run: pip install requests")

        url     = f"{self.base_url}/query"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(prompt)
        backoff = 30

        for attempt in range(self.max_retries):
            self._pace()
            try:
                resp = requests.post(url, headers=headers, json=payload,
                                     timeout=self.timeout)

                if resp.status_code == 429:
                    logger.warning(
                        f"[{self.name}] 429 rate limit "
                        f"(attempt {attempt+1}/{self.max_retries}) — "
                        f"waiting {backoff}s."
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 300)
                    continue

                if resp.status_code in (401, 403):
                    raise RuntimeError(
                        f"[{self.name}] Auth error {resp.status_code}: "
                        f"check ASU_CREATEAI_TOKEN. Body: {resp.text[:300]}"
                    )

                resp.raise_for_status()
                try:
                    data = resp.json()
                except json.JSONDecodeError:
                    raise ValueError(f"Non-JSON body: {resp.text[:300]}")

                result = self._extract_text(data).strip()
                if not result:
                    raise ValueError("Empty response text from gateway.")
                self._set_reported_output_tokens(
                    self._extract_output_tokens(data)
                )
                self._set_cached(cache_key, result)
                return result

            except RuntimeError:
                raise                       # auth errors: fail fast
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"[{self.name}] Transient error "
                        f"(attempt {attempt+1}/{self.max_retries}): {e} — "
                        f"retrying in 5s..."
                    )
                    time.sleep(5)
                else:
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
