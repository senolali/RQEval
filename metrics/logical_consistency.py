"""Logical Coherence metric: LS = 1 - contradiction_rate in reasoning traces."""

from typing import Any, Dict, List, Optional


class LogicalConsistencyMetric:
    """Detects contradictions in reasoning traces using NLI.

    Formula: LS = 1 - (1/N) * sum((1/(n_i-1)) * sum(psi(s_j, s_{j+1})))
    where psi(s_j, s_{j+1}) = 1 if NLI(s_j, s_{j+1}) == 'contradiction', else 0.
    Returns a normalized score in [0, 1].

    Batch processing: all step-pairs across all traces are sent to the NLI
    pipeline in a single call to maximise GPU throughput and suppress the
    "use a dataset" warning from transformers.
    """

    def __init__(self, config: Dict[str, Any] = None, nli_model: Optional[str] = None):
        self.config = config or {}
        self.nli_model_name = nli_model or "cross-encoder/nli-deberta-v3-small"
        self._nli_pipeline = None

    def _load_nli(self):
        """Lazy-load the NLI pipeline on GPU if available, else CPU."""
        try:
            import torch
            from transformers import pipeline
            device = 0 if torch.cuda.is_available() else -1  # 0=GPU, -1=CPU
            self._nli_pipeline = pipeline(
                "text-classification",
                model=self.nli_model_name,
                top_k=None,
                device=device,
            )
        except Exception:
            self._nli_pipeline = None

    def _ensure_nli(self):
        """Load NLI pipeline on first use."""
        if self._nli_pipeline is None:
            try:
                self._load_nli()
            except Exception:
                self._nli_pipeline = None

    def _batch_nli(self, pairs: List[str]) -> List[str]:
        """Run NLI on a list of 'premise [SEP] hypothesis' strings in one batch.

        Returns a list of predicted labels (contradiction / entailment / neutral).
        Falls back to 'neutral' for any failed item.
        """
        if not pairs:
            return []
        try:
            results = self._nli_pipeline(
                pairs,
                truncation=True,
                max_length=512,
                batch_size=32,        # process up to 32 pairs per GPU call
            )
            labels = []
            for result in results:
                row = {r["label"].lower(): r["score"] for r in result}
                labels.append(max(row, key=row.get))
            return labels
        except Exception:
            return ["neutral"] * len(pairs)

    def _heuristic_contradiction(self, steps: List[str]) -> float:
        """Heuristic fallback when NLI is unavailable."""
        negations = {"not", "no", "never", "false", "incorrect", "wrong",
                     "cannot", "isn't", "aren't"}
        contradiction_count = 0
        total_pairs = 0
        for i in range(len(steps) - 1):
            total_pairs += 1
            words_a = set(steps[i].lower().split())
            words_b = set(steps[i + 1].lower().split())
            content_a = words_a - negations
            content_b = words_b - negations
            neg_in_a = bool(words_a & negations)
            neg_in_b = bool(words_b & negations)
            overlap = content_a & content_b
            if overlap and (neg_in_a != neg_in_b):
                contradiction_count += 1
        return (contradiction_count / total_pairs) if total_pairs else 0.0

    def compute(self, reasoning_traces: List[List[str]]) -> float:
        """Compute average logical coherence across all traces (batched).

        Args:
            reasoning_traces: List of reasoning step lists.

        Returns:
            Normalized coherence score in [0, 1].
        """
        if not reasoning_traces:
            return 1.0

        self._ensure_nli()

        if self._nli_pipeline is None:
            # Heuristic fallback — no NLI model available
            scores = [1.0 - self._heuristic_contradiction(steps)
                      for steps in reasoning_traces]
            return sum(scores) / len(scores)

        # ── Batch all step-pairs from all traces into a single NLI call ──
        # Build index: (trace_idx, pair_idx) → flat position in batch
        flat_inputs: List[str] = []
        index: List[tuple] = []   # (trace_idx, total_pairs_in_trace)

        per_trace_pairs: List[List[int]] = []  # flat positions per trace

        for t_idx, steps in enumerate(reasoning_traces):
            if len(steps) <= 1:
                per_trace_pairs.append([])
                continue
            positions = []
            for j in range(len(steps) - 1):
                flat_inputs.append(f"{steps[j]} [SEP] {steps[j + 1]}")
                positions.append(len(flat_inputs) - 1)
            per_trace_pairs.append(positions)

        # Single batched NLI call
        if flat_inputs:
            all_labels = self._batch_nli(flat_inputs)
        else:
            all_labels = []

        # Reconstruct per-trace scores
        scores = []
        for t_idx, steps in enumerate(reasoning_traces):
            positions = per_trace_pairs[t_idx]
            if not positions:
                scores.append(1.0)
                continue
            total_pairs = len(positions)
            contradictions = sum(
                1 for pos in positions
                if "contradiction" in all_labels[pos]
            )
            scores.append(1.0 - contradictions / total_pairs)

        return sum(scores) / len(scores)

    def compute_trace(self, steps: List[str]) -> float:
        """Compute logical coherence for a single trace (convenience wrapper)."""
        return self.compute([steps])
