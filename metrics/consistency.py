"""Answer-level consistency (CS) across a fixed K independent runs."""

from typing import Any, Dict, List, Optional

from metrics.answer_extraction import extract_answer


class ConsistencyMetric:
    """Pairwise canonical-answer agreement with a fixed K denominator."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        return str(answer).strip().lower()

    def compute_instance(
        self,
        answers: List[str],
        gold: Optional[str] = None,
    ) -> float:
        expected_k = int(self.config.get("consistency_runs", len(answers) or 3))
        if expected_k < 2:
            return 0.0

        fixed = list(answers[:expected_k])
        fixed.extend([""] * (expected_k - len(fixed)))

        normalized = []
        for i, answer in enumerate(fixed):
            if not str(answer).strip():
                # Each failed/missing run gets a unique label.  Two failures
                # therefore cannot create artificial agreement.
                normalized.append(f"__failed_run_{i}__")
            elif gold is not None:
                extracted = extract_answer(answer, gold)
                normalized.append(
                    extracted if extracted is not None else f"__unparsed_run_{i}__"
                )
            else:
                normalized.append(self._normalize_answer(answer))

        agree = 0
        total_pairs = 0
        for i in range(expected_k):
            for j in range(i + 1, expected_k):
                total_pairs += 1
                if normalized[i] == normalized[j]:
                    agree += 1
        return agree / total_pairs if total_pairs else 0.0

    def compute(
        self,
        all_answers: List[List[str]],
        gold_answers: Optional[List[str]] = None,
    ) -> float:
        if not all_answers:
            return 0.0
        golds = gold_answers if gold_answers is not None else [None] * len(all_answers)
        scores = [
            self.compute_instance(answers, gold)
            for answers, gold in zip(all_answers, golds)
        ]
        return sum(scores) / len(scores)

    def compute_per_instance(
        self,
        all_answers: List[List[str]],
        gold_answers: Optional[List[str]] = None,
    ) -> List[float]:
        golds = gold_answers if gold_answers is not None else [None] * len(all_answers)
        return [
            self.compute_instance(answers, gold)
            for answers, gold in zip(all_answers, golds)
        ]
