"""
GSM8K Dataset Loader
--------------------
Grade School Math 8K — mathematical reasoning benchmark.
Source: openai/gsm8k (HuggingFace datasets)

Answer format: numerical (extracted from "#### <number>" pattern)
Reasoning type: multi-step arithmetic word problems
"""

import re
import random
import logging
from typing import Any, Dict, List, Optional

from llm_datasets.base_dataset import BaseDataset
from llm_datasets.perturbation import generate_perturbations

logger = logging.getLogger(__name__)


def _extract_gsm8k_answer(solution_text: str) -> str:
    """Extract final numerical answer from GSM8K solution string."""
    # GSM8K answers follow the pattern: #### <number>
    match = re.search(r"####\s*([\d,\.\-]+)", solution_text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: last number in text
    numbers = re.findall(r"[\d,]+\.?\d*", solution_text)
    if numbers:
        return numbers[-1].replace(",", "")
    return solution_text.strip()


class GSM8KDataset(BaseDataset):
    """
    GSM8K mathematical reasoning dataset.

    Loads from HuggingFace: openai/gsm8k
    Each item contains:
        - question: math word problem
        - answer:   numerical answer string
        - type:     "reasoning"
        - perturbations: semantic-preserving perturbations for
                        robustness testing (see llm_datasets/perturbation.py)
        - full_solution: step-by-step solution (for coherence evaluation)
    """

    HF_DATASET_ID = "openai/gsm8k"
    HF_SUBSET     = "main"
    HF_SPLIT      = "test"

    def __init__(
        self,
        name: str = "gsm8k",
        config: Dict[str, Any] = None,
        num_samples: int = 250,
        seed: int = 42,
        split: str = "test",
    ):
        super().__init__(name=name, config=config or {}, seed=seed)
        self.num_samples = num_samples
        self.split       = split

    def load(self) -> None:
        """Download and prepare GSM8K samples."""
        logger.info(f"[GSM8K] Loading {self.num_samples} samples from HuggingFace...")

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Run: pip install datasets")

        hf_dataset = load_dataset(
            self.HF_DATASET_ID,
            self.HF_SUBSET,
            split=self.split,
        )

        # Reproducible shuffle + sample
        rng = random.Random(self.seed)
        indices = list(range(len(hf_dataset)))
        rng.shuffle(indices)
        selected = indices[: self.num_samples]

        self._data = []
        for idx, raw_idx in enumerate(selected):
            row = hf_dataset[raw_idx]
            question = row["question"].strip()
            answer   = _extract_gsm8k_answer(row["answer"])

            self._data.append({
                "id":             f"gsm8k_{idx:04d}",
                "question":       question,
                "answer":         answer,
                "full_solution":  row["answer"],
                "type":           "reasoning",
                "dataset":        "gsm8k",
                "perturbations":  generate_perturbations(question, n=3),
            })

        logger.info(f"[GSM8K] Loaded {len(self._data)} items.")
