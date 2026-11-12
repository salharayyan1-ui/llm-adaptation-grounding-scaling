from typing import Optional, Tuple
import os

def get_torch_device() -> Tuple[str, bool]:
    """Check device availability safely without hard crashes if torch is absent."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", True
        return "cpu", False
    except ImportError:
        return "cpu", False

def get_peak_memory_mb() -> float:
    """Return peak GPU memory allocated in MB, or 0.0 if CUDA is unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except (ImportError, Exception):
        pass
    return 0.0

def reset_peak_memory_stats() -> None:
    """Reset peak memory tracking."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, Exception):
        pass
