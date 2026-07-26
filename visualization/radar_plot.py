"""Radar plot visualization for multi-dimensional reasoning quality metrics."""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


METRIC_LABELS = ["Correctness", "Consistency", "Robustness", "Coherence", "Efficiency", "Stability"]
METRIC_KEYS = ["correctness", "consistency", "robustness", "logical_coherence", "efficiency", "stability"]

COLORS = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#A8DADC",
    "#6A0572", "#3A86FF", "#8338EC", "#06D6A0",
]


class RadarPlot:
    """Generate publication-quality radar plots comparing model reasoning metrics."""

    def __init__(self, output_dir: str = "outputs", figsize: tuple = (10, 8)):
        self.output_dir = output_dir
        self.figsize = figsize
        os.makedirs(output_dir, exist_ok=True)

    def _get_scores(self, raw_metrics: Dict[str, float]) -> List[float]:
        return [raw_metrics.get(k, 0.0) for k in METRIC_KEYS]

    def plot(
        self,
        results: List[Dict[str, Any]],
        filename: str = "radar_plot.png",
        title: str = "Multi-Dimensional Reasoning Quality",
    ) -> str:
        """Generate and save a radar plot.

        Args:
            results: List of per-model result dicts (from Evaluator).
            filename: Output file name.
            title: Plot title.

        Returns:
            Path to saved plot.
        """
        n_metrics = len(METRIC_LABELS)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw={"polar": True})
        ax.set_facecolor("#F8F9FA")
        fig.patch.set_facecolor("white")

        # Gridlines styling
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")
        ax.yaxis.grid(True, color="lightgray", linestyle="--", linewidth=0.5)
        ax.xaxis.grid(True, color="lightgray", linestyle="--", linewidth=0.5)

        # Axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(METRIC_LABELS, fontsize=11, fontweight="bold")

        for i, result in enumerate(results):
            model_name = result["model"]
            scores = self._get_scores(result["raw_metrics"])
            scores += scores[:1]  # close polygon

            color = COLORS[i % len(COLORS)]
            ax.plot(angles, scores, "o-", linewidth=2, color=color, label=model_name, markersize=5)
            ax.fill(angles, scores, alpha=0.15, color=color)

        ax.set_title(title, size=14, fontweight="bold", pad=20)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.35, 1.15),
            frameon=True,
            framealpha=0.9,
            edgecolor="lightgray",
            fontsize=10,
        )

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        return output_path

    def plot_bar_comparison(
        self,
        results: List[Dict[str, Any]],
        filename: str = "bar_comparison.png",
        title: str = "Reasoning Quality Metrics by Model",
    ) -> str:
        """Generate grouped bar chart comparing models across metrics."""
        n_models = len(results)
        n_metrics = len(METRIC_KEYS)
        x = np.arange(n_metrics)
        width = 0.8 / max(n_models, 1)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_facecolor("#F8F9FA")
        fig.patch.set_facecolor("white")

        for i, result in enumerate(results):
            scores = [result["raw_metrics"].get(k, 0.0) for k in METRIC_KEYS]
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, scores, width,
                label=result["model"],
                color=COLORS[i % len(COLORS)],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, score in zip(bars, scores):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{score:.2f}",
                    ha="center", va="bottom", fontsize=7, color="gray",
                )

        ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score [0, 1]", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(METRIC_LABELS, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        return output_path
