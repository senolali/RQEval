"""
Local HuggingFace model implementation.

Supports:
- Automatic 4-bit quantization for 7B+ models (requires bitsandbytes)
- float32 / float16 for smaller models
- Sequential loading: model is released from RAM after evaluation
- CPU and CUDA support
- Deterministic generation
- Response caching
"""

import gc
import time
import logging
from typing import Any, Dict, List, Optional

from models.base_model import BaseModel
from utils.tokenizer import count_tokens

logger = logging.getLogger(__name__)


# Prompt templates per model family
PROMPT_TEMPLATES = {
    "phi":      "Instruct: {prompt}\nOutput:",
    "qwen":     "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    "mistral":  "[INST] {prompt} [/INST]",
    "llama":    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
    "default":  "{prompt}",
}


def _detect_family(model_id: str) -> str:
    mid = model_id.lower()
    if "phi" in mid:        return "phi"
    if "qwen" in mid:       return "qwen"
    if "mistral" in mid:    return "mistral"
    if "llama" in mid:      return "llama"
    return "default"


def _format_prompt(model_id: str, prompt: str) -> str:
    family = _detect_family(model_id)
    return PROMPT_TEMPLATES[family].format(prompt=prompt)


class LocalModel(BaseModel):
    """
    HuggingFace local model wrapper.

    RAM strategy:
      - Small models  (<=3B): float32, CPU
      - Medium models (<=7B): 4-bit quantization via bitsandbytes, CPU/CUDA
      - Large models  (>7B) : 4-bit quantization, requires CUDA or large RAM

    The model is lazy-loaded on first generate() call and can be explicitly
    unloaded with release() to free RAM before loading the next model.
    """

    SMALL_MODEL_THRESHOLD  = 3_000_000_000   # 3B params
    MEDIUM_MODEL_THRESHOLD = 9_000_000_000   # 9B params

    def __init__(
        self,
        name: str,
        model_id: str,
        config: Dict[str, Any],
        device: str = "cpu",
        max_new_tokens: int = 256,
        use_4bit: Optional[bool] = None,   # None = auto-decide
        use_8bit: bool = False,
        trust_remote_code: bool = True,
        deterministic: bool = True,
        low_cpu_mem_usage: bool = True,
        temperature: float = 0.7,          # comes from config, used in generate()
    ):
        super().__init__(name=name, config=config, deterministic=deterministic)
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.use_4bit = use_4bit        # None = auto
        self.temperature = temperature
        self.use_8bit = use_8bit
        self.trust_remote_code = trust_remote_code
        self.low_cpu_mem_usage = low_cpu_mem_usage

        self._model = None
        self._tokenizer = None
        self._family = _detect_family(model_id)
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading / unloading
    # ------------------------------------------------------------------

    def _should_quantize_4bit(self) -> bool:
        """Auto-decide 4-bit quantization based on model size hint."""
        if self.use_4bit is not None:
            return self.use_4bit
        # Heuristic: 7B+ models → quantize
        name_lower = self.model_id.lower()
        for size_hint in ["7b", "8b", "13b", "14b", "70b"]:
            if size_hint in name_lower:
                return True
        return False

    def _load_model(self) -> None:
        """Lazy-load tokenizer and model with appropriate precision."""
        if self._loaded:
            return

        logger.info(f"[{self.name}] Loading from HuggingFace: {self.model_id}")

        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )
            import torch
        except ImportError as e:
            raise ImportError(
                f"Missing dependency: {e}\n"
                "Run: pip install transformers torch accelerate"
            )

        # --- Tokenizer ---
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            padding_side="left",
        )
        if self.deterministic:
            self._tokenizer.padding_side = "left"

        # Ensure pad token exists
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # --- Quantization config ---
        quantize_4bit = self._should_quantize_4bit()

        cuda_ok = torch.cuda.is_available() and self.device == "cuda"

        # ── 4-bit quantization (bitsandbytes) ─────────────────────────────
        # KRITIK: device_map ve low_cpu_mem_usage KULLANMA.
        # Bunlar tied-weight modellerde (Qwen2, LLaMA vb.) modeli once "meta"
        # tensoru olarak olusturuyor, sonra gercek cihaza tasiyamiyor.
        # Cozum: from_pretrained ile direkt yukle, .to() ile tasi.
        # ──────────────────────────────────────────────────────────────────
        if quantize_4bit and cuda_ok:
            try:
                import bitsandbytes  # noqa
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info(f"[{self.name}] 4-bit quant, loading to CPU first...")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=bnb_config,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=False,
                )
                logger.info(f"[{self.name}] 4-bit model loaded OK.")
            except Exception as e:
                logger.warning(f"[{self.name}] 4-bit failed ({e}), falling back to float16 CUDA")
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=self.trust_remote_code,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=False,
                ).to("cuda")
        elif cuda_ok:
            logger.info(f"[{self.name}] float16 on CUDA")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=False,
            ).to("cuda")
        else:
            logger.info(f"[{self.name}] float32 on CPU")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
            )

        self._model.eval()
        self._loaded = True
        logger.info(f"[{self.name}] Model ready.")

    def release(self) -> None:
        """Unload model from RAM/VRAM to free memory for the next model."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        self._cache.clear()

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info(f"[{self.name}] Model released from memory.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs) -> str:
        cache_key = self._cache_key(prompt, **kwargs)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if not self._loaded:
            self._load_model()

        import torch

        formatted = _format_prompt(self.model_id, prompt)

        inputs = self._tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        )

        # Move inputs to model device
        try:
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        if self.deterministic:
            # do_sample=False + temperature → NaN/inf hatasi
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[0][input_len:]
        response = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        self._set_cached(cache_key, response)
        return response

    def generate_with_trace(self, prompt: str, **kwargs) -> Dict[str, Any]:
        start = time.time()
        response = self.generate(prompt, **kwargs)
        elapsed = time.time() - start

        token_count = count_tokens(response, hf_tokenizer=self._tokenizer)
        # Split on sentence boundaries for reasoning steps
        steps = [s.strip() for s in response.replace("\n", ". ").split(".") if s.strip()]

        return {
            "response": response,
            "token_count": token_count,
            "latency": elapsed,
            "reasoning_steps": steps if steps else [response],
            "model": self.name,
        }

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate for multiple prompts, loading model once."""
        if not self._loaded:
            self._load_model()
        return [self.generate(p, **kwargs) for p in prompts]
