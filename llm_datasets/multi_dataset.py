"""
Multi-Dataset Loader
---------------------
Combines multiple datasets (Synthetic + GSM8K + StrategyQA + MMLU)
into a single unified dataset for evaluation.

Tracks dataset origin per item so results can be reported per-dataset.
"""

import logging
from typing import Any, Dict, List, Optional

from llm_datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MultiDataset(BaseDataset):
    """
    Combines multiple BaseDataset instances into one.

    Items retain a 'dataset' field indicating their source.
    Supports per-dataset filtering via filter_by_dataset().
    """

    def __init__(
        self,
        name: str = "multi",
        config: Dict[str, Any] = None,
        seed: int = 42,
    ):
        super().__init__(name=name, config=config or {}, seed=seed)
        self._sub_datasets: List[BaseDataset] = []

    def add(self, dataset: BaseDataset) -> "MultiDataset":
        """Add a loaded dataset. Call dataset.load() before adding."""
        self._sub_datasets.append(dataset)
        return self

    def load(self) -> None:
        """Merge all sub-datasets into self._data."""
        self._data = []
        for ds in self._sub_datasets:
            for item in ds.get_all():
                # Ensure dataset tag is set
                item_copy = dict(item)
                if "dataset" not in item_copy:
                    item_copy["dataset"] = ds.name
                self._data.append(item_copy)
        logger.info(
            f"[MultiDataset] Combined {len(self._sub_datasets)} datasets "
            f"→ {len(self._data)} total items"
        )

    def filter_by_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Return items from a specific source dataset."""
        return [item for item in self._data if item.get("dataset") == dataset_name]

    def dataset_names(self) -> List[str]:
        """Return list of unique dataset names present."""
        return list({item.get("dataset", "unknown") for item in self._data})

    def summary(self) -> Dict[str, int]:
        """Return item count per dataset."""
        result: Dict[str, int] = {}
        for item in self._data:
            ds = item.get("dataset", "unknown")
            result[ds] = result.get(ds, 0) + 1
        return result
