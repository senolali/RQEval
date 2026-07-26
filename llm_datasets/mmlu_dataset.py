"""
MMLU Dataset Loader
--------------------
Massive Multitask Language Understanding — 57-subject multiple choice benchmark.
Source: cais/mmlu (HuggingFace datasets)

Answer format: "A", "B", "C", or "D"
Reasoning type: knowledge + reasoning across diverse domains
"""

import random
import logging
from typing import Any, Dict, List, Optional

from llm_datasets.base_dataset import BaseDataset
from llm_datasets.perturbation import generate_perturbations

logger = logging.getLogger(__name__)

# Curated MMLU subjects relevant to reasoning quality evaluation
DEFAULT_SUBJECTS = [
    "logical_fallacies",
    "formal_logic",
    "abstract_algebra",
    "elementary_mathematics",
    "high_school_mathematics",
    "college_mathematics",
    "high_school_statistics",
    "conceptual_physics",
    "philosophy",
]

CHOICE_LABELS = ["A", "B", "C", "D"]


def _format_question(question: str, choices: List[str]) -> str:
    """Format MMLU question with labeled choices."""
    lines = [question]
    for label, choice in zip(CHOICE_LABELS, choices):
        lines.append(f"{label}. {choice}")
    lines.append("Answer with A, B, C, or D.")
    return "\n".join(lines)


def _make_perturbations(raw_question: str, choices: List[str], n: int = 3) -> List[str]:
    """Perturb the question stem (WordNet synonym substitution,
    dependency-parse reordering, back-translation -- see
    llm_datasets/perturbation.py) and reformat each variant with the
    original, unperturbed choices. Choices are intentionally left
    unperturbed: rephrasing an answer option risks invalidating the
    correct-answer mapping."""
    perturbed_stems = generate_perturbations(raw_question, n=n)
    return [_format_question(stem, choices) for stem in perturbed_stems]


class MMLUDataset(BaseDataset):
    """
    MMLU multiple-choice reasoning dataset.

    Loads from HuggingFace: cais/mmlu
    Samples evenly across DEFAULT_SUBJECTS for diverse coverage.

    Each item contains:
        - question:      formatted question with A/B/C/D choices
        - answer:        "A", "B", "C", or "D"
        - subject:       MMLU subject name
        - type:          "reasoning"
        - perturbations: semantic-preserving perturbations of the
                        question stem, reformatted with the original
                        choices (see llm_datasets/perturbation.py)
    """

    HF_DATASET_ID = "cais/mmlu"
    HF_SPLIT      = "test"

    def __init__(
        self,
        name: str = "mmlu",
        config: Dict[str, Any] = None,
        num_samples: int = 250,
        seed: int = 42,
        subjects: Optional[List[str]] = None,
        split: str = "test",
    ):
        super().__init__(name=name, config=config or {}, seed=seed)
        self.num_samples = num_samples
        self.subjects    = subjects or DEFAULT_SUBJECTS
        self.split       = split

    def load(self) -> None:
        logger.info(f"[MMLU] Loading {self.num_samples} samples across {len(self.subjects)} subjects...")

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Run: pip install datasets")

        rng             = random.Random(self.seed)
        per_subject     = max(1, self.num_samples // len(self.subjects))
        all_items       = []

        for subject in self.subjects:
            try:
                hf_dataset = load_dataset(
                    self.HF_DATASET_ID,
                    subject,
                    split=self.split,
                )
            except Exception as e:
                logger.warning(f"[MMLU] Could not load subject '{subject}': {e}")
                continue

            indices = list(range(len(hf_dataset)))
            rng.shuffle(indices)
            selected = indices[:per_subject]

            for raw_idx in selected:
                row     = hf_dataset[raw_idx]
                choices = row["choices"]
                answer_idx = int(row["answer"])
                answer_label = CHOICE_LABELS[answer_idx]

                formatted_q = _format_question(row["question"], choices)

                all_items.append({
                    "question":       formatted_q,
                    "answer":         answer_label,
                    "raw_question":   row["question"],
                    "choices":        choices,
                    "subject":        subject,
                    "type":           "reasoning",
                    "dataset":        "mmlu",
                    "perturbations":  _make_perturbations(row["question"], choices, n=3),
                })

        # Final shuffle + trim to exact num_samples
        rng.shuffle(all_items)
        all_items = all_items[: self.num_samples]

        self._data = [
            {**item, "id": f"mmlu_{i:04d}"}
            for i, item in enumerate(all_items)
        ]

        logger.info(f"[MMLU] Loaded {len(self._data)} items from {len(self.subjects)} subjects.")
