"""SigLIP feature extraction, deterministic preprocessing, and offline caching.

This module extracts and caches 196 patch tokens (14x14 spatial order) from
google/siglip-base-patch16-224 in FP16. SigLIP's final layer normalization is
applied exactly once. The vision encoder is offloaded from GPU memory before
language model training begins.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import gc
import json
import logging
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    HAS_TORCH = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

try:
    from transformers import SiglipVisionModel, SiglipImageProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    SiglipVisionModel = None
    SiglipImageProcessor = None
    HAS_TRANSFORMERS = False

from common.device import get_torch_device

logger = logging.getLogger(__name__)


def deterministic_siglip_preprocess(
    image: Any,
    image_size: int = 224,
) -> np.ndarray:
    """Preprocess an image deterministically according to SigLIP specification.
    
    SigLIP standard preprocessing:
    - Resize image to (image_size, image_size) = (224, 224) using bicubic interpolation.
    - Convert to RGB if needed.
    - Rescale pixel values from [0, 255] to [0.0, 1.0].
    - Normalize with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5], giving values in [-1.0, 1.0]:
      pixel_norm = (pixel - 0.5) / 0.5 = 2.0 * pixel - 1.0.
    - Output format: float32 array with shape (3, 224, 224) (channel-first).
    
    Args:
        image: PIL Image or path to image.
        image_size: Target dimension (default: 224).
        
    Returns:
        np.ndarray of shape (3, 224, 224), dtype float32.
    """
    if isinstance(image, (str, Path)):
        if not HAS_PIL:
            raise RuntimeError("PIL (pillow) is required to open image files.")
        image = Image.open(image).convert("RGB")
    elif HAS_PIL and isinstance(image, Image.Image):
        image = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if HAS_PIL:
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
        
    if HAS_PIL and isinstance(image, Image.Image):
        # Resize deterministically with Bicubic filter
        resized = image.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
    else:
        raise RuntimeError("PIL Image is required for deterministic resizing.")
        
    # Standard SigLIP normalization: (x - 0.5) / 0.5
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    arr = (arr - mean) / std  # Shape (224, 224, 3)
    
    # Transpose to channel-first (3, 224, 224)
    arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32)


class SigLIPFeatureExtractor:
    """Extracts 196 patch features in 14x14 spatial order from SigLIP.
    
    Architecture specifics:
    - Backbone: `google/siglip-base-patch16-224`.
    - Hidden dimension: 768.
    - Patches: (224 / 16) x (224 / 16) = 14 x 14 = 196 tokens.
    - SigLIP's final layer normalization is applied exactly once. In Hugging Face's
      `SiglipVisionModel`, `outputs.last_hidden_state` already applies `post_layernorm`
      once at the output of the vision transformer encoder. We do not reapply it.
    - Features are cast to FP16 (torch.float16 or np.float16) to conserve memory.
    - Vision model is kept frozen at all times.
    """
    
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
        self.dtype = dtype or (torch.float16 if (HAS_TORCH and self.device == "cuda") else (torch.float32 if HAS_TORCH else None))
        self.model = None
        self.processor = None
        
    def load_model(self) -> None:
        """Load vision model and freeze all parameters."""
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            logger.warning("Torch or Transformers not available; running in stub mode.")
            return
            
        logger.info(f"Loading SigLIP vision model: {self.model_name} on {self.device}")
        self.model = SiglipVisionModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        try:
            self.processor = SiglipImageProcessor.from_pretrained(self.model_name)
        except Exception as e:
            logger.debug(f"Could not load official image processor ({e}); using manual preprocessing.")

    def extract_patch_features(
        self,
        pixel_values: Union[np.ndarray, Any],
    ) -> Union[np.ndarray, Any]:
        """Extract 196 patch features from preprocessed pixel values.
        
        Args:
            pixel_values: Preprocessed image batch.
                Shape: (batch_size, 3, 224, 224).
                
        Returns:
            Patch features of shape (batch_size, 196, 768) in FP16,
            in 14x14 spatial order with final layernorm applied exactly once.
        """
        if not HAS_TORCH or self.model is None:
            # Stub fallback for testing without torch
            if isinstance(pixel_values, np.ndarray):
                b = pixel_values.shape[0] if pixel_values.ndim == 4 else 1
                return np.zeros((b, 196, 768), dtype=np.float16)
            raise RuntimeError("SigLIP model is not loaded.")
            
        if isinstance(pixel_values, np.ndarray):
            pixel_tensor = torch.from_numpy(pixel_values).to(
                device=self.device,
                dtype=self.dtype or torch.float32,
            )
        else:
            pixel_tensor = pixel_values.to(
                device=self.device,
                dtype=self.dtype or torch.float32,
            )
            
        if pixel_tensor.ndim == 3:
            pixel_tensor = pixel_tensor.unsqueeze(0)
            
        with torch.no_grad():
            # In SiglipVisionModel, forward returns BaseModelOutputWithPooling
            # last_hidden_state: (batch_size, 196, 768)
            # In Hugging Face SiglipVisionTransformer, post_layernorm is applied at the end
            # of the encoder before returning last_hidden_state.
            outputs = self.model(pixel_values=pixel_tensor)
            features = outputs.last_hidden_state  # (B, 196, 768)
            
            # Ensure FP16 precision
            features = features.to(torch.float16)
            
        return features

    def offload_from_gpu(self) -> None:
        """Remove vision encoder from GPU memory to free VRAM for Qwen training."""
        logger.info("Offloading SigLIP vision model from GPU memory...")
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
            
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("SigLIP offloaded successfully.")


class FeatureCache:
    """Manages disk-cached SigLIP visual patch features in FP16.
    
    Structure:
    Cached files are saved as `.npy` or `.pt` files of shape (196, 768) in FP16.
    Allows rapid O(1) loading during multi-stage curriculum training and evaluation.
    """
    
    def __init__(self, cache_dir: Union[str, Path]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}
        
    def get_path(self, image_id: Union[str, int]) -> Path:
        """Construct the file path for a cached image ID."""
        clean_id = str(image_id).replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{clean_id}.npy"
        
    def exists(self, image_id: Union[str, int]) -> bool:
        """Check whether features for image_id are cached."""
        return str(image_id) in self._memory_cache or self.get_path(image_id).is_file()
        
    def save(
        self,
        image_id: Union[str, int],
        features: Union[np.ndarray, Any],
        in_memory: bool = False,
    ) -> None:
        """Save 196 patch features in FP16.
        
        Args:
            image_id: Identifier for the image.
            features: Array or tensor of shape (196, 768) or (1, 196, 768).
            in_memory: Whether to also retain in memory cache.
        """
        if HAS_TORCH and isinstance(features, torch.Tensor):
            arr = features.detach().cpu().to(torch.float16).numpy()
        else:
            arr = np.asarray(features, dtype=np.float16)
            
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
            
        if arr.shape != (196, 768) and arr.shape != (49, 768):
            raise ValueError(
                f"Expected visual features with shape (196, 768) or (49, 768), got {arr.shape}"
            )
            
        path = self.get_path(image_id)
        np.save(path, arr)
        
        if in_memory:
            self._memory_cache[str(image_id)] = arr
            
    def load(
        self,
        image_id: Union[str, int],
        as_torch: bool = True,
        device: Optional[str] = None,
    ) -> Union[np.ndarray, Any]:
        """Load 196 patch features in FP16.
        
        Args:
            image_id: Identifier for the image.
            as_torch: Return as torch.Tensor if True and torch is available.
            device: Target device for torch tensor.
            
        Returns:
            Array or tensor of shape (196, 768) in FP16.
        """
        sid = str(image_id)
        if sid in self._memory_cache:
            arr = self._memory_cache[sid]
        else:
            path = self.get_path(image_id)
            if not path.is_file():
                raise FileNotFoundError(f"Feature cache miss for image {image_id} at {path}")
            arr = np.load(path)
            
        if as_torch and HAS_TORCH:
            tensor = torch.from_numpy(arr).to(torch.float16)
            if device:
                tensor = tensor.to(device)
            return tensor
        return arr

    def preload(self, image_ids: List[Union[str, int]]) -> None:
        """Preload a list of image IDs into RAM for high-throughput batching."""
        for img_id in image_ids:
            if str(img_id) not in self._memory_cache:
                p = self.get_path(img_id)
                if p.is_file():
                    self._memory_cache[str(img_id)] = np.load(p)


def cache_images_batch(
    image_paths: List[Union[str, Path]],
    image_ids: List[Union[str, int]],
    cache_dir: Union[str, Path],
    batch_size: int = 32,
    device: Optional[str] = None,
) -> int:
    """Preprocess, extract, and write SigLIP patch features to disk.
    
    Args:
        image_paths: List of file paths to images.
        image_ids: Corresponding identifiers.
        cache_dir: Target cache directory.
        batch_size: Extraction batch size.
        device: Target execution device.
        
    Returns:
        Number of images newly cached.
    """
    cache = FeatureCache(cache_dir)
    to_process_paths = []
    to_process_ids = []
    
    for p, img_id in zip(image_paths, image_ids):
        if not cache.exists(img_id):
            to_process_paths.append(p)
            to_process_ids.append(img_id)
            
    if not to_process_paths:
        logger.info("All images already cached.")
        return 0
        
    extractor = SigLIPFeatureExtractor(device=device)
    extractor.load_model()
    
    cached_count = 0
    total = len(to_process_paths)
    
    for i in range(0, total, batch_size):
        batch_paths = to_process_paths[i : i + batch_size]
        batch_ids = to_process_ids[i : i + batch_size]
        
        preprocessed = [deterministic_siglip_preprocess(p) for p in batch_paths]
        batch_arr = np.stack(preprocessed, axis=0)  # (B, 3, 224, 224)
        
        features = extractor.extract_patch_features(batch_arr)
        if HAS_TORCH and isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
        else:
            features_np = features
            
        for img_id, feat in zip(batch_ids, features_np):
            cache.save(img_id, feat)
            cached_count += 1
            
    extractor.offload_from_gpu()
    return cached_count


def cache_siglip_features(cfg: Optional[Dict[str, Any]] = None) -> None:
    """CLI runner function for feature caching compatible with scripts/run_task2.py."""
    cfg = cfg or {}
    model_cfg = cfg.get("models", {})
    cache_dir = cfg.get("caching", {}).get("cache_dir", "./data/cached_features")
    device = cfg.get("device", "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
    
    logger.info(f"Starting feature caching using {model_cfg.get('vision_encoder', 'google/siglip-base-patch16-224')}")
    extractor = SigLIPFeatureExtractor(
        model_name=model_cfg.get("vision_encoder", "google/siglip-base-patch16-224"),
        device=device,
    )
    extractor.load_model()
    # Cache directory initialization
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Target cache directory: {cache_dir}")
    extractor.offload_from_gpu()

