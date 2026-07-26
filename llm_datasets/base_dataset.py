"""Abstract base class for all datasets."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional
import json


class BaseDataset(ABC):
    """Abstract dataset interface for plug-and-play dataset support."""

    def __init__(self, name: str, config: Dict[str, Any], seed: int = 42):
        self.name = name
        self.config = config
        self.seed = seed
        self._data: List[Dict[str, Any]] = []

    @abstractmethod
    def load(self) -> None:
        """Load or generate the dataset."""
        pass

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._data[idx]

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all dataset items."""
        return self._data

    def save_json(self, path: str) -> None:
        """Save dataset to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str, name: str = "json_dataset", seed: int = 42) -> "BaseDataset":
        """Load a dataset from a JSON file.

        Uses a trivial concrete subclass because BaseDataset is abstract
        (``load`` is already satisfied by reading the file here).
        """
        class _JsonDataset(cls if cls is not BaseDataset else BaseDataset):
            def load(self) -> None:  # data already loaded in from_json
                pass
        # Bypass abstractness cleanly via the concrete subclass:
        instance = _JsonDataset.__new__(_JsonDataset)
        instance.name = name
        instance.config = {}
        instance.seed = seed
        with open(path, "r", encoding="utf-8") as f:
            instance._data = json.load(f)
        return instance

    def filter_by_type(self, task_type: str) -> List[Dict[str, Any]]:
        """Filter items by task type."""
        return [item for item in self._data if item.get("type") == task_type]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, size={len(self._data)})"
