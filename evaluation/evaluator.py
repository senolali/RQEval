"""
Main evaluation pipeline orchestrator.

Supports:
- Multiple datasets evaluated separately + combined
- Sequential RAM management for local models
- Per-dataset metric breakdown (for publication tables)
"""

import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from llm_datasets.base_dataset import BaseDataset
from models.base_model import BaseModel
from metrics.accuracy import AccuracyMetric
from metrics.consistency import ConsistencyMetric
from metrics.robustness import RobustnessMetric
from metrics.logical_consistency import LogicalConsistencyMetric
from metrics.efficiency import EfficiencyMetric
from metrics.explainability import ExplainabilityMetric
from metrics.aggregation import AggregationStrategy
from utils.logger import get_logger
from utils.experiment_tracker import ExperimentTracker
from utils.checkpoint import ItemCheckpoint, item_key as ck_item_key
from utils.latex_export import export_latex

# Lazy import to avoid circular dependency
def _get_plotter(exp_dir: str):
    from visualization.radar_plot import RadarPlot
    return RadarPlot(output_dir=exp_dir)

logger = get_logger(__name__)


class Evaluator:
    """Full pipeline: load → generate → 6 metrics → aggregate → export."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = "outputs",
        experiment_id: Optional[str] = None,
    ):
        self.config        = config
        self.output_dir    = output_dir
        self.experiment_id = experiment_id or f"exp_{int(time.time())}"
        self.tracker       = ExperimentTracker(
            experiment_id=self.experiment_id,
            output_dir=output_dir,
        )

        m = config.get("metrics", {})
        self.consistency_runs         = m.get("consistency_runs", 3)
        self.stability_runs           = m.get("stability_runs", 3)
        self.robustness_perturbations = m.get("robustness_perturbations", 3)
        # Parallel workers for API models. Set 1 for local models.
        # Override in config.yaml: max_workers: 8
        self.max_workers = config.get("max_workers", 1)

        self.accuracy_metric    = AccuracyMetric(config=m)
        self.consistency_metric = ConsistencyMetric(config=m)
        self.robustness_metric  = RobustnessMetric(config=m)
        self.coherence_metric   = LogicalConsistencyMetric(
            config=m, nli_model=m.get("nli_model"))
        self.efficiency_metric  = EfficiencyMetric(config=m)
        self.stability_metric   = ExplainabilityMetric(
            config=m, bertscore_model=m.get("bertscore_model"))
        self.aggregation        = AggregationStrategy(config=config)

    def _release_metric_models(self) -> None:
        """Release NLI and BERTScore models from VRAM after each model evaluation.

        Called once per LLM model (after all per-dataset passes complete) so
        NLI/BERTScore stay alive across per-dataset breakdown — avoiding
        repeated reload cycles (~30-80 min saved per model).
        """
        try:
            import torch, gc
            # Release NLI pipeline
            if (hasattr(self.coherence_metric, '_nli_pipeline')
                    and self.coherence_metric._nli_pipeline is not None):
                del self.coherence_metric._nli_pipeline
                self.coherence_metric._nli_pipeline = None
                logger.info("  [Released NLI model from memory]")
            # Release BERTScore model if stored as attribute
            for attr in ('_model', '_bertscore_model', 'model'):
                if (hasattr(self.stability_metric, attr)
                        and getattr(self.stability_metric, attr) is not None):
                    setattr(self.stability_metric, attr, None)
                    logger.info("  [Released BERTScore model from memory]")
                    break
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Core metric computation (reusable for any item subset)
    # ------------------------------------------------------------------

    def _process_item(self, model, item, idx, total, log_lock):
        def log(msg):
            with log_lock:
                logger.info(msg)
        log(f"    [{idx+1}/{total}] {item.get('dataset','?')} | {item['question'][:55]}...")
        failed = False
        try:
            td = model.generate_with_trace(item["question"])
        except Exception as e:
            log(f"    [FAILED] Generation failed: {e}")
            failed = True
            td = {
                "response": "",
                "token_count": 0,
                "latency": 0.0,
                "reasoning_steps": [""],
            }
        log(f"      Gold   : {str(item['answer'])[:80]}")
        log(f"      Pred   : {str(td['response'])[:80]}")
        log(f"      Tokens : {td['token_count']} | Latency: {td['latency']:.2f}s")
        k_ans = [td["response"]]
        k_tr  = [" ".join(td["reasoning_steps"] or [""])]
        for k_i in range(self.consistency_runs - 1):
            if failed:
                continue
            try:
                td2 = model.generate_with_trace(item["question"])
                k_ans.append(td2["response"])
                k_tr.append(" ".join(td2["reasoning_steps"] or [""]))
                log(f"        [CS/SS run {k_i+2}/{self.consistency_runs}] Pred: {str(td2['response'])[:80]}")
            except Exception as e:
                log(f"        [CS/SS run {k_i+2}/{self.consistency_runs}] Failed: {e} - zero contribution")
        perts   = item.get("perturbations", [])[:self.robustness_perturbations]
        p_preds = []
        for p_i, p in enumerate(perts):
            if failed or not p or str(p).strip() == str(item["question"]).strip():
                p_preds.append("")
                continue
            try:
                p_resp = model.generate(p)
                p_preds.append(p_resp)
                log(f"        [RS perturb {p_i+1}/{len(perts)}] Pred: {str(p_resp)[:80]}")
            except Exception as e:
                log(f"        [RS perturb {p_i+1}/{len(perts)}] Failed: {e} - zero contribution")
        return {
            "prediction":      td["response"],
            "gold_answer":     item["answer"],
            "token_count":     max(int(td.get("token_count", 0)), 0),
            "reasoning_trace": td["reasoning_steps"] or [""],
            "k_answers":       k_ans,
            "k_traces":        k_tr,
            "perturbed_preds": p_preds,
            "dataset":         item.get("dataset", "unknown"),
            "question_id":     str(item.get("question_id", idx)),
            "generation_failed": failed,
        }

    def _get_checkpoint(self, model) -> Optional[ItemCheckpoint]:
        """Build the per-model checkpoint (stable path across reruns).

        Path: <output_dir>/checkpoints/<experiment_name>/<model>.jsonl
        Fingerprint covers everything that changes item content or run
        structure; a mismatch invalidates the old checkpoint.
        Disable via config:  experiment: { checkpoint: false }
        """
        exp_cfg = self.config.get("experiment", {})
        if exp_cfg.get("checkpoint", True) is False:
            return None
        import re
        exp_name = exp_cfg.get("name", "experiment")
        safe_exp = re.sub(r'[^A-Za-z0-9_\-]', '_', exp_name)
        safe_mod = re.sub(r'[^A-Za-z0-9_\-]', '_', model.name)
        path = os.path.join(self.output_dir, "checkpoints", safe_exp,
                            f"{safe_mod}.jsonl")
        scalar_model_params = {}
        for attr in (
            "model_id", "model_name", "model_provider", "base_url",
            "max_tokens", "max_new_tokens", "temperature", "_temperature",
            "deterministic", "system_prompt",
        ):
            value = getattr(model, attr, None)
            if isinstance(value, (str, int, float, bool)) or value is None:
                scalar_model_params[attr] = value
        fingerprint = {
            "protocol_version":          "paper-v8-fixed-denominators",
            "model":                    model.name,
            "model_params":             scalar_model_params,
            "seed":                     exp_cfg.get("seed"),
            "datasets":                 self.config.get("datasets", []),
            "consistency_runs":         self.consistency_runs,
            "stability_runs":           self.stability_runs,
            "robustness_perturbations": self.robustness_perturbations,
            "metrics":                  self.config.get("metrics", {}),
            "aggregation":              self.config.get("aggregation", {}),
        }
        return ItemCheckpoint(path, fingerprint, enabled=True)

    def _compute_metrics(self, model, items):
        """Run all 6 metrics using parallel workers.

        API models  : max_workers=8  →  ~8x speedup  (5h → ~40min)
        Local models: max_workers=1  →  sequential (avoids OOM)
        Set in config.yaml:  max_workers: 8
        """
        total    = len(items)
        # Local modeller: her zaman 1 worker, model onceden yukle
        is_local = hasattr(model, "_load_model")
        workers  = 1 if is_local else self.max_workers

        # Local model icin: evaluate baslamadan once modeli yukle
        # Boylece her _process_item cagrisi ayri yukleme yapmaz
        if is_local and not getattr(model, "_loaded", False):
            logger.info(f"  [Local model] Pre-loading {model.name}...")
            model._load_model()
            logger.info(f"  [Local model] {model.name} ready.")

        log_lock = threading.Lock()
        logger.info(f"  [Parallel workers: {workers}]  ({'local' if is_local else 'api'})")

        # ── Checkpoint / resume (item-level) ─────────────────────────
        ckpt = self._get_checkpoint(model)
        keys = [ck_item_key(item, idx) for idx, item in enumerate(items)]

        results_by_idx = {}
        pending = []
        for idx, item in enumerate(items):
            cached_entry = ckpt.get(keys[idx]) if ckpt is not None else None
            if cached_entry is not None:
                results_by_idx[idx] = cached_entry
            else:
                pending.append(idx)

        if ckpt is not None and len(pending) < total:
            logger.info(
                f"  [Checkpoint] {total - len(pending)}/{total} items restored "
                f"— running remaining {len(pending)}."
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._process_item, model, items[idx], idx, total, log_lock): idx
                for idx in pending
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    entry = future.result()
                    results_by_idx[idx] = entry
                    if ckpt is not None and entry is not None:
                        ckpt.add(keys[idx], entry)
                except Exception as e:
                    logger.error(f"    [Worker error idx={idx}]: {e}")
                    results_by_idx[idx] = None
        if ckpt is not None:
            ckpt.close()

        predictions=[]; gold_answers=[]; token_counts=[]; reasoning_traces=[]
        all_k_answers=[]; all_k_traces=[]; perturbed_preds=[]; cache={}

        for idx in range(total):
            entry = results_by_idx.get(idx)
            if entry is None:
                item = items[idx]
                entry = {
                    "prediction": "",
                    "gold_answer": item["answer"],
                    "token_count": 0,
                    "reasoning_trace": [""],
                    "k_answers": [""] * self.consistency_runs,
                    "k_traces": [""] * self.stability_runs,
                    "perturbed_preds": [""] * self.robustness_perturbations,
                    "dataset": item.get("dataset", "unknown"),
                    "question_id": str(item.get("question_id", idx)),
                    "generation_failed": True,
                }
            predictions.append(entry["prediction"])
            gold_answers.append(entry["gold_answer"])
            token_counts.append(entry["token_count"])
            reasoning_traces.append(entry["reasoning_trace"])
            all_k_answers.append(entry["k_answers"])
            all_k_traces.append(entry["k_traces"])
            perturbed_preds.append(entry["perturbed_preds"])
            cache[entry["question_id"]] = entry

        self._prediction_cache = cache

        correctness_flags = self.accuracy_metric.compute_per_instance(predictions, gold_answers)
        cq = sum(correctness_flags) / len(correctness_flags) if correctness_flags else 0.0
        cs = self.consistency_metric.compute(all_k_answers, gold_answers)
        rs = self.robustness_metric.compute(predictions, perturbed_preds, gold_answers)
        ls = self.coherence_metric.compute(reasoning_traces)
        es = self.efficiency_metric.compute(correctness_flags, token_counts)
        ss = self.stability_metric.compute(all_k_traces)

        return {
            "correctness":       round(cq, 4),
            "consistency":       round(cs, 4),
            "robustness":        round(rs, 4),
            "logical_coherence": round(ls, 4),
            "efficiency":        round(es, 4),
            "stability":         round(ss, 4),
        }


    # ------------------------------------------------------------------
    # Cached per-dataset breakdown (zero extra API calls)
    # ------------------------------------------------------------------

    def _compute_metrics_cached(
        self,
        model: BaseModel,
        all_items: List[Dict[str, Any]],
        subset: List[Dict[str, Any]],
        ds_name: str,
    ) -> Dict[str, float]:
        """Compute per-dataset metrics entirely from the prediction cache.

        Zero extra API/model calls — all data was collected during the overall
        pass in _compute_metrics and stored in self._prediction_cache.
        """
        cache = getattr(self, "_prediction_cache", {})

        predictions:      List[str]       = []
        gold_answers:     List[str]       = []
        token_counts:     List[int]       = []
        reasoning_traces: List[List[str]] = []
        all_k_answers:    List[List[str]] = []
        all_k_traces:     List[List[str]] = []
        perturbed_preds:  List[List[str]] = []

        for idx, item in enumerate(subset):
            key = str(item.get("question_id", all_items.index(item)
                               if item in all_items else idx))
            entry = cache.get(key)

            if entry is None:
                # Fallback: item was not cached (should not happen in normal flow)
                logger.warning(f"  [Cache miss] {key} — generating fresh")
                try:
                    td = model.generate_with_trace(item["question"])
                except Exception:
                    td = {"response": "", "token_count": 1,
                          "latency": 0.0, "reasoning_steps": [""], "model": model.name}
                entry = {
                    "prediction":      td["response"],
                    "gold_answer":     item["answer"],
                    "token_count":     max(td["token_count"], 1),
                    "reasoning_trace": td["reasoning_steps"] or [""],
                    "k_answers":       [td["response"]],
                    "k_traces":        [" ".join(td["reasoning_steps"] or [""])],
                    "perturbed_preds": [],
                }

            predictions.append(entry["prediction"])
            gold_answers.append(entry["gold_answer"])
            token_counts.append(entry["token_count"])
            reasoning_traces.append(entry["reasoning_trace"])
            all_k_answers.append(entry["k_answers"])
            all_k_traces.append(entry["k_traces"])
            perturbed_preds.append(entry["perturbed_preds"])

        correctness_flags = self.accuracy_metric.compute_per_instance(predictions, gold_answers)
        cq = sum(correctness_flags) / len(correctness_flags) if correctness_flags else 0.0
        cs = self.consistency_metric.compute(all_k_answers, gold_answers)
        rs = self.robustness_metric.compute(predictions, perturbed_preds, gold_answers)
        ls = self.coherence_metric.compute(reasoning_traces)
        es = self.efficiency_metric.compute(correctness_flags, token_counts)
        ss = self.stability_metric.compute(all_k_traces)

        return {
            "correctness":       round(cq, 4),
            "consistency":       round(cs, 4),
            "robustness":        round(rs, 4),
            "logical_coherence": round(ls, 4),
            "efficiency":        round(es, 4),
            "stability":         round(ss, 4),
        }

    # ------------------------------------------------------------------
    # Single model evaluation (overall + per-dataset breakdown)
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model: BaseModel,
        dataset: BaseDataset,
    ) -> Dict[str, Any]:
        """Evaluate one model on the full dataset + per-dataset breakdown."""
        logger.info(f"\n{'='*60}")
        logger.info(f"  Model : {model.name}")
        logger.info(f"  Items : {len(dataset)}")
        logger.info(f"{'='*60}")

        all_items = dataset.get_all()

        # Check datasets upfront
        ds_names: List[str] = []
        if hasattr(dataset, "dataset_names"):
            ds_names = dataset.dataset_names()
        elif all_items:
            ds_names = list({item.get("dataset", "unknown") for item in all_items})

        # --- Overall metrics (single pass — no double computation) ---
        logger.info("  [Overall metrics]")
        overall_raw = self._compute_metrics(model, all_items)
        overall_agg = self.aggregation.aggregate_all_strategies(overall_raw)

        logger.info(
            f"  → CQ={overall_raw['correctness']:.3f} "
            f"CS={overall_raw['consistency']:.3f} "
            f"RS={overall_raw['robustness']:.3f} "
            f"LS={overall_raw['logical_coherence']:.3f} "
            f"ES={overall_raw['efficiency']:.3f} "
            f"SS={overall_raw['stability']:.3f} "
            f"| Balanced={overall_agg.get('balanced',0):.3f}"
        )

        # --- Per-dataset breakdown (reuse cached predictions — NO extra API calls) ---
        per_dataset: Dict[str, Any] = {}

        if len(ds_names) > 1:
            # NLI and BERTScore models are already loaded from the overall pass.
            # We keep them alive across ALL per-dataset passes and release only once
            # at the end — avoiding repeated load/unload cycles (~30-80 min saved).
            logger.info("  [Per-dataset breakdown — NLI/BERTScore shared across datasets]")
            for ds_name in sorted(ds_names):
                subset = [item for item in all_items if item.get("dataset") == ds_name]
                if not subset:
                    continue
                logger.info(f"  [{ds_name}] {len(subset)} items — reading from cache (0 API calls)")
                subset_raw = self._compute_metrics_cached(model, all_items, subset, ds_name)
                subset_agg = self.aggregation.aggregate_all_strategies(subset_raw)
                per_dataset[ds_name] = {
                    "raw_metrics": subset_raw,
                    "aggregated":  subset_agg,
                    "num_samples": len(subset),
                }
                logger.info(
                    f"    → CQ={subset_raw['correctness']:.3f} "
                    f"CS={subset_raw['consistency']:.3f} "
                    f"RS={subset_raw['robustness']:.3f} "
                    f"LS={subset_raw['logical_coherence']:.3f} "
                    f"ES={subset_raw['efficiency']:.3f} "
                    f"SS={subset_raw['stability']:.3f} "
                    f"| Balanced={subset_agg.get('balanced',0):.3f}"
                )

        result = {
            "model":         model.name,
            "experiment_id": self.experiment_id,
            "raw_metrics":   overall_raw,
            "aggregated":    overall_agg,
            "per_dataset":   per_dataset,
            "metadata": {
                "num_samples":              len(all_items),
                "consistency_runs":         self.consistency_runs,
                "stability_runs":           self.stability_runs,
                "robustness_perturbations": self.robustness_perturbations,
                "datasets":                 ds_names,
            },
        }

        self.tracker.log_model_result(model.name, result)

        # ── Per-model immediate save: Excel + plots ─────────────
        self._save_incremental(result)

        # Release NLI/BERTScore from VRAM before loading next LLM
        self._release_metric_models()

        # Release local model RAM/VRAM
        if hasattr(model, "release"):
            model.release()

        return result

    # ------------------------------------------------------------------
    # Incremental per-model save
    # ------------------------------------------------------------------

    def _save_incremental(self, result: Dict[str, Any]) -> None:
        """Save Excel and plots after each model completes.

        Files are overwritten on each call so the outputs directory always
        contains the latest cumulative results — safe to inspect mid-run.
        """
        try:
            completed = self.tracker._model_results   # all results so far
            exp_dir   = self.tracker.exp_dir
            model_name = result["model"]

            # ── Excel (cumulative — all completed models) ────────
            self.tracker.export_excel(
                completed, "reasoning_quality_results.xlsx")

            # ── LaTeX tables (cumulative — all completed models) ──
            export_latex(
                completed, os.path.join(exp_dir, "reasoning_quality_tables.tex"))

            # ── Per-model radar (single model) ───────────────────
            plotter = _get_plotter(exp_dir)
            import re
            safe = re.sub(r'[^A-Za-z0-9_\-]', '_', model_name.replace(" ", "_"))
            safe = re.sub(r'_+', '_', safe).strip('_')
            plotter.plot(
                [result],
                filename=f"radar_{safe}.png",
                title=f"Reasoning Quality — {model_name}",
            )

            # ── Cumulative radar (all completed models) ──────────
            if len(completed) > 1:
                plotter.plot(
                    completed,
                    filename="radar_plot.png",
                    title="Multi-Dimensional Reasoning Quality",
                )

            # ── Bar comparison (needs ≥2 models) ─────────────────
            if len(completed) >= 2:
                plotter.plot_bar_comparison(
                    completed, "bar_comparison.png")

            saved = [
                f"reasoning_quality_results.xlsx",
                f"radar_{safe}.png",
                "reasoning_quality_tables.tex",
            ]
            if len(completed) >= 2:
                saved += ["radar_plot.png", "bar_comparison.png"]
            logger.info(f"  [Saved] {' · '.join(saved)}")

        except Exception as e:
            logger.warning(f"  [Incremental save failed] {e}")

    # ------------------------------------------------------------------
    # Multi-model pipeline
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        models: List[BaseModel],
        dataset: BaseDataset,
    ) -> List[Dict[str, Any]]:
        """Evaluate all models sequentially with RAM management."""
        logger.info(f"\nEvaluation pipeline")
        logger.info(f"  Models  : {[m.name for m in models]}")
        logger.info(f"  Samples : {len(dataset)}")

        results = []
        for i, model in enumerate(models):
            logger.info(f"\n[{i+1}/{len(models)}] {model.name}")
            result = self.evaluate_model(model, dataset)
            results.append(result)

        self.tracker.save_summary(results)
        return results
