"""
StrategyQA Dataset Loader
--------------------------
Multi-step commonsense reasoning with binary (yes/no) answers.
Source: ChilleD/StrategyQA (HuggingFace datasets), test split.

Answer format: "yes" or "no"
Reasoning type: implicit multi-step commonsense
"""

import random
import logging
from typing import Any, Dict, List

from llm_datasets.base_dataset import BaseDataset
from llm_datasets.perturbation import generate_perturbations

logger = logging.getLogger(__name__)


class StrategyQADataset(BaseDataset):
    """
    StrategyQA commonsense reasoning dataset.

    Loads from HuggingFace: ChilleD/StrategyQA, split="test" (687 items).
    Each item contains:
        - question:      yes/no question
        - answer:        "yes" or "no"
        - type:          "reasoning"
        - facts:         supporting facts list (for coherence eval)
        - perturbations: semantic-preserving perturbations (see
                         llm_datasets/perturbation.py)
    """

    HF_DATASET_ID = "ChilleD/StrategyQA"
    HF_SPLIT      = "test"

    def __init__(
        self,
        name: str = "strategyqa",
        config: Dict[str, Any] = None,
        num_samples: int = 250,
        seed: int = 42,
        split: str = "test",
    ):
        super().__init__(name=name, config=config or {}, seed=seed)
        self.num_samples = num_samples
        self.split       = split

    def load(self) -> None:
        logger.info(f"[StrategyQA] Loading {self.num_samples} samples from HuggingFace "
                    f"({self.HF_DATASET_ID}, split={self.split!r})...")

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Run: pip install datasets")

        try:
            hf_dataset = load_dataset(self.HF_DATASET_ID, split=self.split)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load StrategyQA from {self.HF_DATASET_ID} "
                f"(split={self.split!r}): {e}\n"
                f"If the dataset's config/subset name has changed, try:\n"
                f'  load_dataset("{self.HF_DATASET_ID}", name="strategyQA", split="{self.split}")'
            )

        rng = random.Random(self.seed)
        indices = list(range(len(hf_dataset)))
        rng.shuffle(indices)
        selected = indices[: self.num_samples]

        self._data = []
        for idx, raw_idx in enumerate(selected):
            row      = hf_dataset[raw_idx]
            question = row["question"].strip()

            # answer field may be bool or string depending on version
            raw_ans = row.get("answer", row.get("gold", ""))
            if isinstance(raw_ans, bool):
                answer = "yes" if raw_ans else "no"
            else:
                answer = str(raw_ans).lower().strip()
                if answer not in ("yes", "no"):
                    answer = "yes" if answer in ("true", "1") else "no"

            facts = row.get("facts", [])

            self._data.append({
                "id":             f"strategyqa_{idx:04d}",
                "question":       question,
                "answer":         answer,
                "facts":          facts,
                "type":           "reasoning",
                "dataset":        "strategyqa",
                "perturbations":  generate_perturbations(question, n=3),
            })

        logger.info(f"[StrategyQA] Loaded {len(self._data)} items.")
