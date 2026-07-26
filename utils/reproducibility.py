"""Reproducibility utilities: seed control and deterministic mode."""

import os
import random


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except (ImportError, AttributeError):
        pass


def enable_deterministic_mode() -> None:
    """Enable deterministic computation across supported libraries."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (ImportError, AttributeError):
        pass


def get_reproducibility_info() -> dict:
    """Return current reproducibility configuration."""
    info = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "not set"),
    }
    try:
        import torch
        info["torch_deterministic"] = torch.are_deterministic_algorithms_enabled()
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["torch"] = "not installed"

    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except ImportError:
        pass

    return info
