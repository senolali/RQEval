"""Aggregation strategies for combining individual reasoning quality metrics."""

from typing import Any, Dict, Optional


METRIC_KEYS = ["correctness", "consistency", "robustness", "logical_coherence", "efficiency", "stability"]


class AggregationStrategy:
    """Aggregates multiple metric scores using configurable weighting strategies.

    Supports:
    - balanced (equal weights)
    - safety_priority
    - accuracy_priority
    - efficiency_priority
    - custom (user-defined weights)

    Raw metric scores are never overridden; aggregation only adds new keys.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._strategies = self._load_strategies()

    def _load_strategies(self) -> Dict[str, Dict[str, float]]:
        """Load strategies from config or use defaults."""
        defaults = {
            "balanced": {k: 1.0 / 6 for k in METRIC_KEYS},
            "safety_priority": {
                "correctness": 0.30,
                "consistency": 0.05,
                "robustness": 0.30,
                "logical_coherence": 0.25,
                "efficiency": 0.05,
                "stability": 0.05,
            },
            "accuracy_priority": {
                "correctness": 0.50,
                "consistency": 0.10,
                "robustness": 0.15,
                "logical_coherence": 0.15,
                "efficiency": 0.05,
                "stability": 0.05,
            },
            "efficiency_priority": {
                "correctness": 0.20,
                "consistency": 0.15,
                "robustness": 0.15,
                "logical_coherence": 0.10,
                "efficiency": 0.30,
                "stability": 0.10,
            },
        }

        # Override from config if present
        if "aggregation" in self.config and "strategies" in self.config["aggregation"]:
            for strategy_name, weights in self.config["aggregation"]["strategies"].items():
                defaults[strategy_name] = weights

        return defaults

    def _validate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1."""
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6 and total > 0:
            return {k: v / total for k, v in weights.items()}
        return weights

    def aggregate(
        self,
        raw_metrics: Dict[str, float],
        strategy: str = "balanced",
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute weighted aggregate score.

        Args:
            raw_metrics: Dictionary with metric scores, must contain METRIC_KEYS.
            strategy: One of 'balanced', 'safety_priority', 'accuracy_priority',
                      'efficiency_priority', or 'custom'.
            custom_weights: Required when strategy == 'custom'.

        Returns:
            Weighted aggregate score in [0, 1].
        """
        if strategy == "custom":
            if custom_weights is None:
                raise ValueError("custom_weights must be provided for 'custom' strategy.")
            weights = self._validate_weights(custom_weights)
        elif strategy in self._strategies:
            weights = self._validate_weights(self._strategies[strategy])
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {list(self._strategies.keys())}")

        score = sum(
            weights.get(key, 0.0) * raw_metrics.get(key, 0.0)
            for key in METRIC_KEYS
        )
        return min(max(score, 0.0), 1.0)

    def aggregate_all_strategies(
        self,
        raw_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute aggregate score for all strategies.

        Args:
            raw_metrics: Raw metric scores dict.

        Returns:
            Dictionary mapping strategy name -> aggregate score.
        """
        results = {}
        for strategy in self._strategies:
            results[strategy] = self.aggregate(raw_metrics, strategy=strategy)
        return results

    def get_strategy_weights(self, strategy: str) -> Dict[str, float]:
        """Return the weights for a given strategy."""
        return self._strategies.get(strategy, {})
