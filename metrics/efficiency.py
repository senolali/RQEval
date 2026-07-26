"""Efficiency metric: ES = harmonic mean of correctness and token conciseness."""

from typing import Any, Dict, List


class EfficiencyMetric:
    """Measures efficiency as harmonic mean of correctness and token conciseness.

    Formula:
        T'_i = (T_i - T_min) / (T_max - T_min)   [normalized token count]
        ES_i = 2 * CQ_i * (1 - T'_i) / (CQ_i + (1 - T'_i))
        ES = 1 - (1/N) * sum(ES_i)     [we invert so higher = better]

    Wait — from paper:
        ES = (1/N) * sum(harmonic_mean(CQ_i, 1 - T'_i))

    Higher score = correct AND concise.
    Returns a normalized score in [0, 1].
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def _harmonic(self, cq: float, conciseness: float) -> float:
        """Compute harmonic mean of correctness and conciseness."""
        denom = cq + conciseness
        if denom == 0:
            return 0.0
        return (2 * cq * conciseness) / denom

    def compute(
        self,
        correctness_flags: List[float],
        token_counts: List[int],
    ) -> float:
        """Compute efficiency score.

        Args:
            correctness_flags: Per-instance correctness (0 or 1).
            token_counts: Number of tokens in each response.

        Returns:
            Normalized efficiency score in [0, 1].
        """
        if not correctness_flags or not token_counts:
            return 0.0

        t_min = min(token_counts)
        t_max = max(token_counts)

        scores = []
        for cq, t in zip(correctness_flags, token_counts):
            if t_max == t_min:
                t_norm = 0.0
            else:
                t_norm = (t - t_min) / (t_max - t_min)
            conciseness = 1.0 - t_norm
            scores.append(self._harmonic(cq, conciseness))

        return sum(scores) / len(scores)

    def compute_per_instance(
        self,
        correctness_flags: List[float],
        token_counts: List[int],
    ) -> List[float]:
        """Return per-instance efficiency scores."""
        if not token_counts:
            return []

        t_min = min(token_counts)
        t_max = max(token_counts)

        results = []
        for cq, t in zip(correctness_flags, token_counts):
            if t_max == t_min:
                t_norm = 0.0
            else:
                t_norm = (t - t_min) / (t_max - t_min)
            conciseness = 1.0 - t_norm
            results.append(self._harmonic(cq, conciseness))
        return results
