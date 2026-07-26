"""Robustness (RS) over a fixed number P of perturbations."""

from typing import Any, Dict, List

from metrics.accuracy import AccuracyMetric


class RobustnessMetric:
    """Compute the paper's |C|-conditioned P-average robustness score.

        RS = (1/|C|) * sum_{i in C} (1/P) * sum_p I[y^i(p) = y_i]

    where C is the set of items whose *original* (unperturbed) answer was
    correct. An item contributes only when its original answer is correct;
    items outside C are excluded from both the numerator and the
    denominator, not merely zeroed out — this is the "trivially high scores
    for consistently-wrong models" safeguard described in the paper's
    Methodology. Missing, invalid, or failed perturbation calls contribute
    zero while remaining in the fixed P denominator for items in C.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._accuracy = AccuracyMetric(config=config)

    def _is_correct(self, prediction: str, gold: str) -> bool:
        return self._accuracy._is_correct(prediction, gold)

    def _instance_score(
        self,
        original: str,
        perturbations: List[str],
        gold: str,
    ) -> float:
        if not self._is_correct(original, gold):
            return 0.0

        expected_p = int(self.config.get("robustness_perturbations", 3))
        if expected_p <= 0:
            return 0.0
        fixed = list(perturbations[:expected_p])
        fixed.extend([""] * (expected_p - len(fixed)))
        matches = sum(
            1
            for pred in fixed
            if str(pred).strip() and self._is_correct(pred, gold)
        )
        return matches / expected_p

    def compute(
        self,
        original_predictions: List[str],
        perturbed_predictions: List[List[str]],
        gold_answers: List[str],
    ) -> float:
        if not original_predictions:
            return 0.0
        scores = self.compute_per_instance(
            original_predictions, perturbed_predictions, gold_answers
        )
        # Average only over C = {i : original_predictions[i] is correct},
        # per the paper's RS formula (Section 3.2). Items outside C are
        # excluded from the denominator, not merely scored zero — otherwise
        # RS would be deflated by (|C|/N) for every model with CQ < 1,
        # rather than measuring perturbation-stability conditional on an
        # already-correct baseline.
        correct_scores = [
            score
            for score, original, gold in zip(scores, original_predictions, gold_answers)
            if self._is_correct(original, gold)
        ]
        if not correct_scores:
            return 0.0
        return sum(correct_scores) / len(correct_scores)

    def compute_per_instance(
        self,
        original_predictions: List[str],
        perturbed_predictions: List[List[str]],
        gold_answers: List[str],
    ) -> List[float]:
        scores = []
        for i, (original, gold) in enumerate(
            zip(original_predictions, gold_answers)
        ):
            perts = (
                perturbed_predictions[i]
                if i < len(perturbed_predictions)
                else []
            )
            scores.append(self._instance_score(original, perts, gold))
        return scores
