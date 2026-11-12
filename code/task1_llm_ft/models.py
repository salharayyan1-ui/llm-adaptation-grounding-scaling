"""Model architectures, LoRA, and OPLoRA implementations for Task 1.

Implements:
1. LoRALinear: Standard low-rank adapter (ranks 4 and 16, alpha = 2 * rank, dropout 0.05).
2. OPLoRALinear: Orthogonal Projection LoRA (rank 16, alpha 32, protection rank k=128).
   - FP32 SVD decomposition W0 = U Sigma V^T computed once and frozen.
   - Projected factors:
       \\tilde{B} = B - Uk (Uk^T B)
       \\tilde{A} = A - (A Vk) Vk^T
       \\Delta W = (\\alpha / r) \\tilde{B} \\tilde{A}
   - Efficient factorized O(d * k * r) multiplication.
   - Support for exporting projected factors and merging.
3. Utility functions for applying adapters, parameter counting, merging, and equivalence verification.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple, Any, Union
import math
import time
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    ModuleBase = nn.Module
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False
    
    class ModuleBase:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)

# Default target linear modules across all transformer layers in Qwen3-0.6B
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]


class LoRALinear(ModuleBase):
    """Low-Rank Adaptation (LoRA) layer wrapping a frozen base linear layer.
    
    W = W0 + (alpha / r) * B @ A
    A is initialized with random Gaussian / Kaiming, B initialized to 0.
    No trainable biases.
    """
    
    def __init__(
        self,
        base_layer: Any,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for LoRALinear")
        super().__init__()
        
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            
        # LoRA parameters in FP32
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32))
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        
        self.merged = False
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        """Initialize A with Kaiming uniform / standard LoRA distribution, B with zeros."""
        if not HAS_TORCH:
            return
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: Any) -> Any:
        base_out = self.base_layer(x)
        if self.merged:
            return base_out
            
        # x shape: (..., in_features)
        # Cast input through adapter in adapter precision (FP32 or input dtype)
        h = self.lora_dropout(x)
        # Factorized multiplication: (h @ A.T) @ B.T * scaling
        adapter_out = F.linear(F.linear(h, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.scaling
        return base_out + adapter_out
        
    def get_delta_weight(self) -> Any:
        """Compute Delta W = (alpha / r) * B @ A in FP32."""
        return (self.lora_B @ self.lora_A) * self.scaling
        
    def merge(self) -> None:
        """Merge adapter weights into base layer weights in-place."""
        if not self.merged:
            delta_w = self.get_delta_weight().to(self.base_layer.weight.dtype)
            self.base_layer.weight.data += delta_w
            self.merged = True
            
    def unmerge(self) -> None:
        """Revert merged adapter weights."""
        if self.merged:
            delta_w = self.get_delta_weight().to(self.base_layer.weight.dtype)
            self.base_layer.weight.data -= delta_w
            self.merged = False


class OPLoRALinear(ModuleBase):
    """Orthogonal Projection LoRA (OPLoRA) layer.
    
    Protects leading singular subspaces of the starting weight W0 = U Sigma V^T.
    B_tilde = B - Uk (Uk^T B)
    A_tilde = A - (A Vk) Vk^T
    Delta W = (alpha / r) * B_tilde @ A_tilde
    
    Uses efficient factorized multiplications O(d * k * r) without forming dense projection matrices.
    """
    
    def __init__(
        self,
        base_layer: Any,
        r: int = 16,
        lora_alpha: float = 32.0,
        protection_rank: int = 128,
        lora_dropout: float = 0.05,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for OPLoRALinear")
        super().__init__()
        
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.k = min(protection_rank, self.in_features, self.out_features)
        
        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            
        # Trainable unprojected adapter parameters in FP32
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=torch.float32))
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        
        self.merged = False
        self.reset_parameters()
        
        # Compute SVD of W0 once in FP32 and freeze leading singular bases Uk, Vk
        t0 = time.perf_counter()
        with torch.no_grad():
            w_fp32 = base_layer.weight.data.to(torch.float32)
            # SVD: W0 = U @ diag(S) @ Vh, shapes: U (d_out, min_dim), Vh (min_dim, d_in)
            U, _, Vh = torch.linalg.svd(w_fp32, full_matrices=False)
            
            # Uk: leading k columns of U -> shape (d_out, k)
            Uk = U[:, :self.k].contiguous()
            # Vk: leading k columns of V -> shape (d_in, k) (note Vh is V^T)
            Vk = Vh[:self.k, :].T.contiguous()
            
        self.svd_time_sec = time.perf_counter() - t0
        
        # Register leading bases as non-trainable buffers
        self.register_buffer("Uk", Uk, persistent=True)
        self.register_buffer("Vk", Vk, persistent=True)
        
        # Record basis storage (in bytes: Uk (d_out, k) + Vk (d_in, k) in FP32)
        self.basis_storage_bytes = (Uk.numel() + Vk.numel()) * Uk.element_size()
        
    def reset_parameters(self) -> None:
        """Initialize A with Kaiming uniform, B with zeros."""
        if not HAS_TORCH:
            return
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def get_projected_factors(self) -> Tuple[Any, Any]:
        """Compute projected factors \\tilde{A} and \\tilde{B} in O(d * k * r) FLOPs.
        
        \\tilde{B} = B - Uk @ (Uk^T @ B)
        \\tilde{A} = A - (A @ Vk) @ Vk^T
        """
        # Uk: (d_out, k), B: (d_out, r)
        # Uk.T @ B: (k, r)
        # Uk @ (Uk.T @ B): (d_out, r)
        B_tilde = self.lora_B - self.Uk @ (self.Uk.T @ self.lora_B)
        
        # A: (r, d_in), Vk: (d_in, k)
        # A @ Vk: (r, k)
        # (A @ Vk) @ Vk.T: (r, d_in)
        A_tilde = self.lora_A - (self.lora_A @ self.Vk) @ self.Vk.T
        
        return A_tilde, B_tilde
        
    def forward(self, x: Any) -> Any:
        base_out = self.base_layer(x)
        if self.merged:
            return base_out
            
        # Projected factors computation: O(d * k * r)
        A_tilde, B_tilde = self.get_projected_factors()
        
        h = self.lora_dropout(x)
        # Factorized linear pass using projected factors: O(B * S * (d_in * r + d_out * r))
        adapter_out = F.linear(
            F.linear(h, A_tilde.to(x.dtype)),
            B_tilde.to(x.dtype)
        ) * self.scaling
        
        return base_out + adapter_out
        
    def get_delta_weight(self) -> Any:
        """Compute Delta W = (alpha / r) * B_tilde @ A_tilde in FP32."""
        A_tilde, B_tilde = self.get_projected_factors()
        return (B_tilde @ A_tilde) * self.scaling
        
    def export_projected_factors(self) -> Dict[str, Any]:
        """Export projected factors \\tilde{A} and \\tilde{B} for inference.
        
        These can be loaded into a standard LoRALinear without requiring Uk and Vk.
        """
        with torch.no_grad():
            A_tilde, B_tilde = self.get_projected_factors()
            return {
                "lora_A": A_tilde.detach().clone(),
                "lora_B": B_tilde.detach().clone(),
                "scaling": self.scaling,
                "r": self.r,
                "lora_alpha": self.lora_alpha,
            }
            
    def merge(self) -> None:
        """Merge projected adapter correction into base layer weights in-place."""
        if not self.merged:
            delta_w = self.get_delta_weight().to(self.base_layer.weight.dtype)
            self.base_layer.weight.data += delta_w
            self.merged = True
            
    def unmerge(self) -> None:
        """Revert merged adapter correction."""
        if self.merged:
            delta_w = self.get_delta_weight().to(self.base_layer.weight.dtype)
            self.base_layer.weight.data -= delta_w
            self.merged = False


def apply_lora_to_model(
    model: Any,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
) -> Tuple[Any, Dict[str, LoRALinear]]:
    """Inject standard LoRA adapters into specified target modules and freeze backbone.
    
    Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    in every transformer layer.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to apply LoRA")
        
    targets = set(target_modules or DEFAULT_TARGET_MODULES)
    
    # Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad = False
        
    adapter_modules: Dict[str, LoRALinear] = {}
    
    for name, module in list(model.named_modules()):
        for target in targets:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Replace with LoRALinear
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent_mod = model.get_submodule(parent_name) if parent_name else model
                
                lora_layer = LoRALinear(
                    base_layer=module,
                    r=rank,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                )
                setattr(parent_mod, child_name, lora_layer)
                adapter_modules[name] = lora_layer
                break
                
    logger.info(f"Applied LoRA (r={rank}, alpha={alpha}) to {len(adapter_modules)} modules")
    return model, adapter_modules


def apply_oplora_to_model(
    model: Any,
    rank: int = 16,
    alpha: float = 32.0,
    protection_rank: int = 128,
    dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
) -> Tuple[Any, Dict[str, OPLoRALinear], Dict[str, float]]:
    """Inject OPLoRA adapters into target modules and freeze backbone.
    
    Computes FP32 SVD leading bases Uk, Vk once per module.
    Records basis storage and preprocessing time.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to apply OPLoRA")
        
    targets = set(target_modules or DEFAULT_TARGET_MODULES)
    
    # Freeze all backbone parameters
    for param in model.parameters():
        param.requires_grad = False
        
    adapter_modules: Dict[str, OPLoRALinear] = {}
    total_svd_time = 0.0
    total_basis_bytes = 0
    
    for name, module in list(model.named_modules()):
        for target in targets:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent_mod = model.get_submodule(parent_name) if parent_name else model
                
                oplora_layer = OPLoRALinear(
                    base_layer=module,
                    r=rank,
                    lora_alpha=alpha,
                    protection_rank=protection_rank,
                    lora_dropout=dropout,
                )
                setattr(parent_mod, child_name, oplora_layer)
                adapter_modules[name] = oplora_layer
                total_svd_time += oplora_layer.svd_time_sec
                total_basis_bytes += oplora_layer.basis_storage_bytes
                break
                
    stats = {
        "num_adapters": len(adapter_modules),
        "total_svd_time_sec": total_svd_time,
        "total_basis_storage_mb": total_basis_bytes / (1024 * 1024),
    }
    logger.info(
        f"Applied OPLoRA (r={rank}, alpha={alpha}, k={protection_rank}) to {len(adapter_modules)} modules. "
        f"SVD Preprocessing time: {total_svd_time:.2f}s, Basis storage: {stats['total_basis_storage_mb']:.2f}MB"
    )
    return model, adapter_modules, stats


def merge_adapters_to_model(model: Any) -> None:
    """Merge all LoRALinear or OPLoRALinear adapters in-place into their base linear weights."""
    count = 0
    for module in model.modules():
        if isinstance(module, (LoRALinear, OPLoRALinear)):
            module.merge()
            count += 1
    logger.info(f"Merged {count} adapters into base model weights")


def unwrap_merged_adapters(model: Any) -> int:
    """Replace merged LoRALinear/OPLoRALinear wrappers by their (merged) base nn.Linear modules."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, (LoRALinear, OPLoRALinear)):
            if not module.merged:
                raise RuntimeError(f"Adapter {name} is not merged; refusing to unwrap")
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, module.base_layer)
            replaced += 1
    return replaced


def unmerge_adapters_from_model(model: Any) -> None:
    """Revert merged adapters."""
    count = 0
    for module in model.modules():
        if isinstance(module, (LoRALinear, OPLoRALinear)):
            module.unmerge()
            count += 1
    logger.info(f"Unmerged {count} adapters")


def export_all_projected_factors(model: Any) -> Dict[str, Dict[str, Any]]:
    """Export all projected factors (\\tilde{A}, \\tilde{B}) from OPLoRA modules."""
    t0 = time.perf_counter()
    exported: Dict[str, Dict[str, Any]] = {}
    for name, module in model.named_modules():
        if isinstance(module, OPLoRALinear):
            exported[name] = module.export_projected_factors()
    export_time = time.perf_counter() - t0
    logger.info(f"Exported projected factors for {len(exported)} modules in {export_time*1000:.2f}ms")
    return exported


def verify_merged_equivalence(
    model_unmerged: Any,
    model_merged: Any,
    sample_inputs: Any,
    atol: float = 1e-4,
) -> bool:
    """Verify merged/unmerged logits agree within ``atol`` (use FP32 models; for FP16 see check_merge_equivalence)."""
    if not HAS_TORCH:
        return True
    with torch.no_grad():
        out_unmerged = model_unmerged(sample_inputs)
        logits_unmerged = out_unmerged.logits if hasattr(out_unmerged, "logits") else out_unmerged
        
        out_merged = model_merged(sample_inputs)
        logits_merged = out_merged.logits if hasattr(out_merged, "logits") else out_merged
        
        diff = (logits_unmerged - logits_merged).abs().max().item()
        is_close = diff < atol
        logger.info(f"Merged vs unmerged max logit difference: {diff:.6e} (tolerance: {atol})")
        return is_close


def check_merge_equivalence(
    model: Any,
    input_ids: Any,
    attention_mask: Optional[Any] = None,
    min_top1_agreement: float = 0.98,
) -> Dict[str, float]:
    """Compare logits of ``model`` with adapters active vs after merging them into the backbone.

    Merges in place (the caller keeps using the merged model). In FP16 the two paths differ by
    rounding, so the check reports max/mean absolute differences and next-token (top-1) agreement
    instead of a bit-exact tolerance, and raises if agreement falls below ``min_top1_agreement``.
    """
    with torch.no_grad():
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        before = model(**kwargs).logits.float()
        merge_adapters_to_model(model)
        after = model(**kwargs).logits.float()
        diff = (before - after).abs()
        agree = (before.argmax(-1) == after.argmax(-1)).float()
        if attention_mask is not None:
            valid = attention_mask.bool()
            agree = agree[valid]
            diff = diff[valid]
        stats = {
            "max_abs_logit_diff": float(diff.max().item()),
            "mean_abs_logit_diff": float(diff.mean().item()),
            "top1_agreement": float(agree.mean().item()),
            "logit_abs_mean": float(before.abs().mean().item()),
        }
    logger.info(f"Merged vs unmerged logits: {stats}")
    if stats["top1_agreement"] < min_top1_agreement:
        raise RuntimeError(f"Merged model disagrees with unmerged model: {stats}")
    return stats


def trainable_state_dict(model: Any) -> Dict[str, Any]:
    """CPU copy of all trainable parameters (adapter factors for LoRA/OPLoRA runs)."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state_dict(model: Any, state: Dict[str, Any]) -> None:
    """Load trainable-parameter tensors saved by ``trainable_state_dict`` (strict on names)."""
    params = dict(model.named_parameters())
    missing = [k for k in state if k not in params]
    if missing:
        raise KeyError(f"Adapter checkpoint has {len(missing)} unknown tensors, e.g. {missing[:3]}")
    with torch.no_grad():
        for name, tensor in state.items():
            params[name].copy_(tensor.to(device=params[name].device, dtype=params[name].dtype))


def load_projected_factors_into_lora(model: Any, factors: Dict[str, Dict[str, Any]]) -> None:
    """Load exported OPLoRA projected factors (tilde-A, tilde-B) into standard LoRALinear modules."""
    modules = dict(model.named_modules())
    for name, f in factors.items():
        mod = modules[name]
        if not isinstance(mod, LoRALinear):
            raise TypeError(f"{name} is {type(mod).__name__}, expected LoRALinear")
        with torch.no_grad():
            mod.lora_A.copy_(f["lora_A"].to(mod.lora_A.device, mod.lora_A.dtype))
            mod.lora_B.copy_(f["lora_B"].to(mod.lora_B.device, mod.lora_B.dtype))


def count_parameters(model: Any) -> Dict[str, int]:
    """Count total, trainable, and frozen parameters in the model, counting tied parameters once."""
    if not HAS_TORCH:
        return {"total": 0, "trainable": 0, "frozen": 0}
        
    seen_params = set()
    total = 0
    trainable = 0
    for p in model.parameters():
        ptr = p.data_ptr() if hasattr(p, "data_ptr") else id(p)
        if ptr in seen_params:
            continue
        seen_params.add(ptr)
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }
