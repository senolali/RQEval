"""Correctness (CQ) using type-aware canonical answer matching."""

from typing import Any, Dict, List

from metrics.answer_extraction import is_correct


class AccuracyMetric:
    """Compute CQ with the paper's shared canonical matching protocol.

    The matching implementation is shared with CS and RS.  In particular,
    single-letter and short numeric gold answers are never accepted through
    unrestricted substring containment.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def _is_correct(self, prediction: str, gold: str) -> bool:
        return is_correct(prediction, gold)

    def compute(self, predictions: List[str], gold_answers: List[str]) -> float:
        if not predictions:
            return 0.0
        scores = self.compute_per_instance(predictions, gold_answers)
        return sum(scores) / len(predictions)

    def compute_per_instance(
        self,
        predictions: List[str],
        gold_answers: List[str],
    ) -> List[float]:
        return [
            1.0 if self._is_correct(pred, gold) else 0.0
            for pred, gold in zip(predictions, gold_answers)
        ]
