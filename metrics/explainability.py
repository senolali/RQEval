"""Stability metric: SS = pairwise BERTScore similarity across reasoning traces."""

from typing import Any, Dict, List, Optional
import itertools


class ExplainabilityMetric:
    """Measures reasoning process stability across multiple runs using BERTScore.

    Formula:
        SS = (1/N) * sum((2/(K*(K-1))) * sum_pairs(Sim(T_i^k, T_i^l)))
    where Sim is BERTScore F1 between reasoning traces.

    All pairwise comparisons are batched into a single BERTScore call to
    avoid repeated model loading and maximise GPU throughput.
    """

    def __init__(self, config: Dict[str, Any] = None, bertscore_model: Optional[str] = None):
        self.config = config or {}
        self.bertscore_model = bertscore_model or "distilbert-base-uncased"
        self._bertscore_available = None

    def _check_bertscore(self) -> bool:
        if self._bertscore_available is not None:
            return self._bertscore_available
        try:
            import bert_score  # noqa
            self._bertscore_available = True
        except ImportError:
            self._bertscore_available = False
        return self._bertscore_available

    def _batch_bertscore(self, cands: List[str], refs: List[str]) -> List[float]:
        """Compute BERTScore F1 for all pairs in a single batched call."""
        results = [0.0] * len(cands)
        valid = [
            i for i, (cand, ref) in enumerate(zip(cands, refs))
            if str(cand).strip() and str(ref).strip()
        ]
        if not valid:
            return results
        try:
            import torch
            from bert_score import score as bert_score_fn
            device = "cuda" if torch.cuda.is_available() else "cpu"
            P, R, F = bert_score_fn(
                [cands[i] for i in valid],
                [refs[i] for i in valid],
                model_type=self.bertscore_model,
                lang="en",
                verbose=False,
                device=device,
                batch_size=64,
            )
            for i, value in zip(valid, F):
                results[i] = float(value)
            return results
        except Exception:
            for i in valid:
                results[i] = self._jaccard(cands[i], refs[i])
            return results

    def _jaccard(self, text_a: str, text_b: str) -> float:
        """Jaccard similarity fallback."""
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a and not tokens_b:
            return 0.0
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def compute(self, all_traces: List[List[str]]) -> float:
        """Compute average stability across N questions (fully batched).

        Args:
            all_traces: List of N items, each containing K trace strings.

        Returns:
            Normalized stability score in [0, 1].
        """
        if not all_traces:
            return 0.0

        expected_k = int(self.config.get("stability_runs", 3))
        fixed_all_traces = []
        for traces in all_traces:
            fixed = list(traces[:expected_k])
            fixed.extend([""] * (expected_k - len(fixed)))
            fixed_all_traces.append(fixed)
        all_traces = fixed_all_traces

        # ── Collect all pairwise comparisons across all questions ──
        # flat_cands[i], flat_refs[i] = one pair to compare
        # pair_map[q_idx] = list of flat indices belonging to question q
        flat_cands: List[str] = []
        flat_refs:  List[str] = []
        pair_map:   List[List[int]] = []

        for traces in all_traces:
            k = len(traces)
            if k <= 1:
                pair_map.append([])
                continue
            positions = []
            for a, b in itertools.combinations(range(k), 2):
                flat_cands.append(traces[a])
                flat_refs.append(traces[b])
                positions.append(len(flat_cands) - 1)
            pair_map.append(positions)

        # ── Single batched similarity call ──
        if flat_cands:
            if self._check_bertscore():
                all_sims = self._batch_bertscore(flat_cands, flat_refs)
            else:
                all_sims = [self._jaccard(a, b)
                            for a, b in zip(flat_cands, flat_refs)]
        else:
            all_sims = []

        # ── Reconstruct per-question scores ──
        scores = []
        for traces, positions in zip(all_traces, pair_map):
            k = len(traces)
            if k <= 1 or not positions:
                scores.append(0.0)
                continue
            coeff = 2.0 / (k * (k - 1))
            total_sim = sum(all_sims[p] for p in positions)
            scores.append(coeff * total_sim)

        return sum(scores) / len(scores)

    def compute_instance(self, traces: List[str]) -> float:
        """Compute stability for one question (convenience wrapper)."""
        return self.compute([traces])
