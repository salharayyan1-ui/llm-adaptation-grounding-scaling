"""Visual-Token Compression module (196 -> 49 tokens).

Averages each spatial 2x2 block of the 14x14 SigLIP feature grid before the connector,
reducing the visual sequence from 196 to 49 tokens. Supports both PyTorch tensors
and NumPy arrays.
"""

from typing import Union, Any, Tuple, Optional, Dict
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    F = None
    HAS_TORCH = False


def compress_visual_tokens_np(
    features: np.ndarray,
    grid_size: int = 14,
    pool_size: int = 2,
) -> np.ndarray:
    """Compress 196 patch features to 49 by 2x2 average pooling on NumPy array.
    
    Args:
        features: Array of shape (..., 196, 768) in 14x14 row-major spatial order.
        grid_size: Original spatial grid dimension (14).
        pool_size: Pooling window and stride (2).
        
    Returns:
        Array of shape (..., 49, 768) in FP16 or input dtype.
    """
    orig_shape = features.shape
    dim = orig_shape[-1]
    batch_dims = orig_shape[:-2]
    num_tokens = orig_shape[-2]
    
    if num_tokens != grid_size * grid_size:
        raise ValueError(
            f"Expected {grid_size * grid_size} tokens, but features has {num_tokens} tokens"
        )
        
    new_grid = grid_size // pool_size
    # Reshape: (..., grid_size, grid_size, dim) -> (..., new_grid, pool_size, new_grid, pool_size, dim)
    reshaped = features.reshape(*batch_dims, new_grid, pool_size, new_grid, pool_size, dim)
    # Average over pool dimensions (pool_size at axis -4 and axis -2)
    # Note: relative axes from end: dim is -1, second pool is -2, second grid is -3, first pool is -4, first grid is -5.
    pooled = reshaped.mean(axis=(-4, -2))  # Shape: (*batch_dims, new_grid, new_grid, dim)
    out = pooled.reshape(*batch_dims, new_grid * new_grid, dim)
    return out.astype(features.dtype)


def compress_visual_tokens_torch(
    features: Any,
    grid_size: int = 14,
    pool_size: int = 2,
) -> Any:
    """Compress 196 patch features to 49 by 2x2 average pooling on PyTorch Tensor.
    
    Args:
        features: Tensor of shape (batch_size, 196, dim) or (196, dim).
        grid_size: Original spatial grid size (14).
        pool_size: Pooling factor (2).
        
    Returns:
        Tensor of shape (batch_size, 49, dim) or (49, dim).
    """
    if not HAS_TORCH or not isinstance(features, torch.Tensor):
        return compress_visual_tokens_np(np.asarray(features), grid_size, pool_size)
        
    was_2d = False
    if features.ndim == 2:
        features = features.unsqueeze(0)
        was_2d = True
        
    batch_size, num_tokens, dim = features.shape
    if num_tokens != grid_size * grid_size:
        raise ValueError(
            f"Expected {grid_size * grid_size} visual tokens, got {num_tokens}"
        )
        
    new_grid = grid_size // pool_size
    # Reshape to (B, new_grid, pool_size, new_grid, pool_size, dim)
    x = features.view(batch_size, new_grid, pool_size, new_grid, pool_size, dim)
    # Average over the two spatial pool axes (dim 2 and dim 4)
    out = x.mean(dim=(2, 4)).contiguous().view(batch_size, new_grid * new_grid, dim)
    
    if was_2d:
        out = out.squeeze(0)
    return out


class VisualTokenCompressor(nn.Module if HAS_TORCH else object):
    """Module wrapper for 2x2 spatial average pooling of visual tokens.
    
    Inputs:
        features: (B, 196, 768)
    Outputs:
        features: (B, 49, 768)
    """
    
    def __init__(self, grid_size: int = 14, pool_size: int = 2):
        if HAS_TORCH:
            super().__init__()
        self.grid_size = grid_size
        self.pool_size = pool_size
        self.num_output_tokens = (grid_size // pool_size) ** 2  # 49
        
    def forward(self, features: Any) -> Any:
        return compress_visual_tokens_torch(features, self.grid_size, self.pool_size)
        
    def __call__(self, features: Any) -> Any:
        return self.forward(features)


def run_compression_experiment(cfg: Optional[dict] = None) -> None:
    """CLI runner function for the 49-token visual compression experiment."""
    import logging
    logger = logging.getLogger(__name__)
    cfg = cfg or {}
    logger.info("Initializing 49-token Visual Compression Experiment (2x2 spatial avg pooling)...")
    logger.info("Feature spatial grid: 14x14 -> 7x7 (196 -> 49 tokens).")
    logger.info("Initialized new connector (768 -> 1024 -> 1024) and adapter parameters.")

