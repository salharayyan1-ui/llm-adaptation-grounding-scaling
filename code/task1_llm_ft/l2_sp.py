"""L2-SP (Starting-Point L2 Regularization) for Task 1: LLM Fine-Tuning.

Implements:
1. Reference parameter storage (\\theta_0) frozen at start of training.
2. Exact penalty calculation:
   \\Omega(\\theta, \\theta_0) = 0.5 * \\sum_{j=1}^P (\\theta_j - \\theta_{0,j})^2
   - Summed over ALL unique parameters (including biases and normalization layers).
   - Counting tied parameters once (verified by data_ptr).
   - Unaveraged penalty (do NOT divide by number of parameters).
   - Weight \\lambda = 10^-3.
3. Full-model displacement tracking:
   ||\\theta - \\theta_0||_2 = \\sqrt{\\sum_{j=1}^P (\\theta_j - \\theta_{0,j})^2}
   tracked at midpoint and final.
4. Overhead tracking: reference storage (MB) and device transfer time.
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
import time
import math
import logging

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class L2SPRegularizer:
    """Manages L2-SP starting-point regularization.
    
    Formula:
        \\Omega(\\theta, \\theta_0) = 0.5 * \\sum_{j=1}^P (\\theta_j - \\theta_{0,j})^2
        Loss_total = Loss_SFT + \\lambda * \\Omega(\\theta, \\theta_0)
        Displacement = ||\\theta - \\theta_0||_2 = \\sqrt{2 * \\Omega(\\theta, \\theta_0)}
    """
    
    def __init__(
        self,
        model: Any,
        lambda_reg: float = 1e-3,
        store_on_cpu: bool = True,
    ):
        """Initialize L2-SP regularizer by capturing \\theta_0 for all unique parameters.
        
        Args:
            model: PyTorch model to regularize.
            lambda_reg: Penalty multiplier \\lambda (default: 1e-3).
            store_on_cpu: If True, stores reference tensors on CPU to conserve GPU VRAM.
        """
        self.lambda_reg = lambda_reg
        self.store_on_cpu = store_on_cpu
        
        self.reference_params: Dict[str, Any] = {}
        self.param_names: List[str] = []
        self.total_elements = 0
        self.reference_storage_bytes = 0
        
        # Cumulative tracking of transfer and penalty compute time
        self.total_compute_time_sec = 0.0
        self.eval_count = 0
        
        if not HAS_TORCH:
            logger.warning("PyTorch not available; L2SPRegularizer initialized in dummy mode.")
            return
            
        t0 = time.perf_counter()
        seen_ptrs: Set[int] = set()
        
        for name, param in model.named_parameters():
            ptr = param.data_ptr()
            # Count tied parameters only once
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            
            # Store frozen copy of parameter
            target_device = "cpu" if store_on_cpu else param.device
            ref_tensor = param.detach().clone().to(device=target_device)
            ref_tensor.requires_grad = False
            
            self.reference_params[name] = ref_tensor
            self.param_names.append(name)
            self.total_elements += param.numel()
            self.reference_storage_bytes += ref_tensor.numel() * ref_tensor.element_size()
            
        init_time = time.perf_counter() - t0
        self.reference_storage_mb = self.reference_storage_bytes / (1024 * 1024)
        logger.info(
            f"L2-SP initialized: {len(self.reference_params)} unique parameter tensors "
            f"({self.total_elements:,} parameters). Reference storage: {self.reference_storage_mb:.2f} MB. "
            f"Capture time: {init_time*1000:.2f} ms. Lambda: {self.lambda_reg}"
        )
        
    def compute_penalty(self, model: Any) -> Tuple[Any, float]:
        """Compute the L2-SP proximity penalty \\Omega(\\theta, \\theta_0).
        
        Returns:
            (penalty_tensor, omega_value)
            penalty_tensor is a differentiable scalar PyTorch Tensor.
            omega_value is the float value of \\Omega.
        """
        if not HAS_TORCH:
            return 0.0, 0.0
            
        t0 = time.perf_counter()
        total_sq_diff = torch.tensor(0.0, device=next(model.parameters()).device, dtype=torch.float32)
        
        # Build map of current named parameters
        current_named_params = dict(model.named_parameters())
        
        for name in self.param_names:
            if name not in current_named_params:
                continue
            param = current_named_params[name]
            ref_param = self.reference_params[name]
            
            # Transfer ref to param device/dtype if needed
            if ref_param.device != param.device or ref_param.dtype != param.dtype:
                ref_on_dev = ref_param.to(device=param.device, dtype=param.dtype)
            else:
                ref_on_dev = ref_param
                
            diff = param - ref_on_dev
            total_sq_diff = total_sq_diff + torch.sum(diff.to(torch.float32) ** 2)
            
        penalty_tensor = 0.5 * total_sq_diff
        omega_val = float(penalty_tensor.item())
        
        compute_time = time.perf_counter() - t0
        self.total_compute_time_sec += compute_time
        self.eval_count += 1
        
        return penalty_tensor, omega_val
        
    def add_penalty_gradient(self, model: Any) -> float:
        """Add the gradient of lambda * Omega to ``p.grad`` once per optimizer update; return Omega.

        d(lambda * 0.5 * ||theta - theta_0||^2) / d theta = lambda * (theta - theta_0), so this is
        exactly equivalent to adding lambda * Omega to the loss, but costs one pass over the
        parameters per update instead of one autograd penalty graph per microbatch. Call after
        ``GradScaler.unscale_`` (gradients are unscaled) and before gradient clipping.
        """
        if not HAS_TORCH:
            return 0.0

        t0 = time.perf_counter()
        current = dict(model.named_parameters())
        omega = torch.zeros((), dtype=torch.float32, device=next(model.parameters()).device)
        with torch.no_grad():
            for name in self.param_names:
                p = current.get(name)
                if p is None:
                    continue
                ref = self.reference_params[name]
                if ref.device != p.device:
                    ref = ref.to(device=p.device, non_blocking=True)
                diff = p.detach().to(torch.float32) - ref.to(torch.float32)
                omega += 0.5 * torch.sum(diff * diff)
                if p.requires_grad:
                    pen = diff * self.lambda_reg
                    if p.grad is None:
                        p.grad = pen.to(p.dtype)
                    else:
                        p.grad.add_(pen.to(p.grad.dtype))
        omega_val = float(omega.item())
        self.total_compute_time_sec += time.perf_counter() - t0
        self.eval_count += 1
        return omega_val

    def compute_displacement(self, model: Any) -> float:
        """Compute full-model parameter displacement ||\\theta - \\theta_0||_2.
        
        Returns:
            Displacement as a float: \\sqrt{\\sum_{j=1}^P (\\theta_j - \\theta_{0,j})^2}
        """
        if not HAS_TORCH:
            return 0.0
            
        with torch.no_grad():
            _, omega_val = self.compute_penalty(model)
            displacement = math.sqrt(max(0.0, 2.0 * omega_val))
            return displacement
            
    def get_stats(self) -> Dict[str, Any]:
        """Return memory and overhead statistics."""
        avg_time_ms = (self.total_compute_time_sec / max(1, self.eval_count)) * 1000.0
        return {
            "reference_storage_mb": self.reference_storage_mb,
            "total_parameters": self.total_elements,
            "num_unique_tensors": len(self.reference_params),
            "lambda_reg": self.lambda_reg,
            "eval_count": self.eval_count,
            "total_overhead_sec": self.total_compute_time_sec,
            "avg_penalty_eval_ms": avg_time_ms,
        }


def compute_parameter_displacement(model: Any, reference_state_dict: Dict[str, Any]) -> float:
    """Compute ||\\theta - \\theta_0||_2 directly between model and a reference state dict."""
    if not HAS_TORCH:
        return 0.0
        
    with torch.no_grad():
        seen_ptrs: Set[int] = set()
        sum_sq = 0.0
        for name, param in model.named_parameters():
            ptr = param.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            
            if name in reference_state_dict:
                ref = reference_state_dict[name].to(device=param.device, dtype=param.dtype)
                diff = param - ref
                sum_sq += float(torch.sum(diff.to(torch.float32) ** 2).item())
                
        return math.sqrt(sum_sq)
