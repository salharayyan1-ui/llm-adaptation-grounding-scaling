"""Weight-space and activation diagnostics for Task 1: LLM Fine-Tuning.

Implements:
1. Weight-space diagnostics:
   - For all 5 final variants + Full FT midpoint on down_proj in layers 0, 13, 27:
     \\rho_k = ||Uk Uk^T \\Delta W||_F^2 / ||\\Delta W||_F^2
     \\eta = ||\\Delta W||_F / ||W_0||_F
   - Uses FP32 leading singular bases Uk (k=128) of starting weight W0.
2. Activation diagnostics:
   - Probe set: 32 GSM8K test questions + 32 HellaSwag validation contexts sampled with seed 6304.
   - Forward hooks caching starting model's down_proj inputs z at the last non-padding prompt token.
   - For all 6 checkpoints, computes:
     R(z) = ||\\Delta W z||_2 / (||W_0 z||_2 + 10^-8)
     in FP32 on math and non-math prompts.
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple, Any, Union
import os
import re
import json
import csv
import logging
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False

from common.seed import set_seed
from common.metrics import (
    compute_subspace_interference,
    compute_relative_update_magnitude,
    compute_activation_ratio,
)
from task1_llm_ft.models import LoRALinear, OPLoRALinear

logger = logging.getLogger(__name__)

DIAGNOSTIC_LAYERS = (0, 13, 27)
PROTECTION_RANK_K = 128


def sample_probe_set(
    gsm8k_questions: Sequence[Dict[str, Any]],
    hellaswag_contexts: Sequence[Dict[str, Any]],
    num_samples_per_task: int = 32,
    seed: int = 6304,
    output_path: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Sample 32 GSM8K test questions and 32 HellaSwag validation contexts using seed 6304.
    
    Persists example IDs and prompts so all models reuse the identical probe set.
    """
    rng = np.random.default_rng(seed)
    
    # GSM8K
    gsm8k_n = len(gsm8k_questions)
    gsm8k_idx = sorted(rng.choice(gsm8k_n, size=min(num_samples_per_task, gsm8k_n), replace=False).tolist())
    sampled_gsm8k = []
    for i in gsm8k_idx:
        item = gsm8k_questions[i]
        sampled_gsm8k.append({
            "id": item.get("id", i),
            "original_index": i,
            "task": "gsm8k",
            "question": item.get("question") or item.get("query") or str(item),
            "prompt": item.get("question") or item.get("query") or str(item),
        })
        
    # HellaSwag
    hella_n = len(hellaswag_contexts)
    hella_idx = sorted(rng.choice(hella_n, size=min(num_samples_per_task, hella_n), replace=False).tolist())
    sampled_hella = []
    for i in hella_idx:
        item = hellaswag_contexts[i]
        sampled_hella.append({
            "id": item.get("ind", item.get("id", i)),
            "original_index": i,
            "task": "hellaswag",
            "doc": {k: item[k] for k in ("activity_label", "ctx_a", "ctx_b") if k in item},
            "prompt": item.get("ctx") or item.get("context") or item.get("activity_label", "") + ": " + item.get("ctx_a", "") + " " + item.get("ctx_b", ""),
        })
        
    probe_set = {
        "seed": seed,
        "num_samples_per_task": num_samples_per_task,
        "gsm8k": sampled_gsm8k,
        "hellaswag": sampled_hella,
    }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(probe_set, f, indent=2)
        logger.info(f"Saved activation probe set to {output_path}")
        
    return probe_set


def extract_down_proj_layer(model: Any, layer_idx: int) -> Any:
    """Extract down_proj module from transformer layer_idx in Qwen architecture."""
    # Find layers list: model.model.layers or model.layers
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        
    if layers is None or layer_idx >= len(layers):
        raise ValueError(f"Could not locate transformer layer {layer_idx} in model")
        
    layer = layers[layer_idx]
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        return layer.mlp.down_proj
    elif hasattr(layer, "mlp") and hasattr(layer.mlp, "c_proj"):
        return layer.mlp.c_proj
    raise ValueError(f"Could not locate down_proj module in layer {layer_idx}")


def extract_weight_and_delta(
    base_module: Any,
    trained_module: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract W0 and Delta W as FP32 numpy arrays.
    
    Handles Full FT, LoRA, and OPLoRA layers.
    """
    if isinstance(base_module, (LoRALinear, OPLoRALinear)):
        w0_tensor = base_module.base_layer.weight.data
    else:
        w0_tensor = base_module.weight.data
    w0_np = w0_tensor.detach().cpu().to(torch.float32).numpy()
    
    if isinstance(trained_module, (LoRALinear, OPLoRALinear)):
        delta_w_tensor = trained_module.get_delta_weight()
        delta_w_np = delta_w_tensor.detach().cpu().to(torch.float32).numpy()
    else:
        w_trained_tensor = trained_module.weight.data
        delta_w_np = (w_trained_tensor - w0_tensor).detach().cpu().to(torch.float32).numpy()
        
    return w0_np, delta_w_np


def compute_weight_diagnostics(
    base_model: Any,
    trained_models: Dict[str, Any],
    layers: Sequence[int] = DIAGNOSTIC_LAYERS,
    k: int = PROTECTION_RANK_K,
) -> Dict[str, Any]:
    """Compute weight-space diagnostics (rho_k and eta) on down_proj in layers 0, 13, 27.
    
    Args:
        base_model: Starting model W0.
        trained_models: Dict mapping condition name (e.g. 'full_ft_midpoint', 'lora_16') to model.
        layers: Layer indices to examine.
        k: Protection rank k for SVD bases.
        
    Returns:
        Structured diagnostics dictionary with rho_k and eta per condition and layer.
    """
    results: Dict[str, Any] = {
        "k": k,
        "layers": list(layers),
        "conditions": {},
    }
    
    # Precompute SVD bases Uk for base model down_proj in target layers
    uk_bases: Dict[int, np.ndarray] = {}
    w0_matrices: Dict[int, np.ndarray] = {}
    
    for layer_idx in layers:
        base_mod = extract_down_proj_layer(base_model, layer_idx)
        if isinstance(base_mod, (LoRALinear, OPLoRALinear)):
            w0 = base_mod.base_layer.weight.data.detach().cpu().to(torch.float32).numpy()
        else:
            w0 = base_mod.weight.data.detach().cpu().to(torch.float32).numpy()
        w0_matrices[layer_idx] = w0
        
        # SVD: W0 = U Sigma V^T
        U, _, _ = np.linalg.svd(w0, full_matrices=False)
        uk_bases[layer_idx] = U[:, :k]
        
    for cond_name, trained_model in trained_models.items():
        cond_results: Dict[str, Any] = {}
        for layer_idx in layers:
            base_mod = extract_down_proj_layer(base_model, layer_idx)
            trained_mod = extract_down_proj_layer(trained_model, layer_idx)
            
            w0, delta_w = extract_weight_and_delta(base_mod, trained_mod)
            u_k = uk_bases[layer_idx]
            
            rho_k = compute_subspace_interference(delta_w, u_k)
            eta = compute_relative_update_magnitude(delta_w, w0)
            
            cond_results[f"layer_{layer_idx}"] = {
                "rho_k": rho_k,
                "eta": eta,
                "delta_w_fro": float(np.linalg.norm(delta_w, ord="fro")),
                "w0_fro": float(np.linalg.norm(w0, ord="fro")),
            }
            
        results["conditions"][cond_name] = cond_results
        
    return results


class ActivationProbeCacher:
    """Attaches forward hooks to starting model down_proj modules to cache prompt activations z."""
    
    def __init__(self, model: Any, layer_indices: Sequence[int] = DIAGNOSTIC_LAYERS):
        self.model = model
        self.layer_indices = layer_indices
        self.cached_activations: Dict[int, List[np.ndarray]] = {idx: [] for idx in layer_indices}
        self.hooks: List[Any] = []
        self._current_last_indices: List[int] = []
        
    def _create_hook(self, layer_idx: int):
        def hook_fn(module: Any, input_tensor: Tuple[Any, ...], output: Any):
            # Input to down_proj has shape (batch_size, seq_len, in_features)
            z_batch = input_tensor[0]
            if not isinstance(z_batch, torch.Tensor):
                return
            z_batch_fp32 = z_batch.detach().to(torch.float32).cpu()
            
            for b, last_idx in enumerate(self._current_last_indices):
                # Slice z at last non-padding prompt token
                z_token = z_batch_fp32[b, last_idx, :].numpy()
                self.cached_activations[layer_idx].append(z_token)
        return hook_fn
        
    def register(self) -> None:
        """Register hooks on target down_proj layers."""
        self.hooks.clear()
        for layer_idx in self.layer_indices:
            mod = extract_down_proj_layer(self.model, layer_idx)
            h = mod.register_forward_hook(self._create_hook(layer_idx))
            self.hooks.append(h)
            
    def remove(self) -> None:
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        
    def cache_prompts(
        self,
        prompts: Sequence[str],
        tokenizer: Any,
        device: str = "cpu",
        batch_size: int = 4,
    ) -> None:
        """Run prompts through model with hooks registered, caching activations z."""
        if not HAS_TORCH:
            return
            
        self.register()
        self.model.eval()
        
        try:
            with torch.no_grad():
                for start_idx in range(0, len(prompts), batch_size):
                    batch_texts = prompts[start_idx : start_idx + batch_size]
                    encoded = tokenizer(
                        batch_texts,
                        padding=True,
                        return_tensors="pt",
                    )
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded["attention_mask"].to(device)
                    
                    # Compute index of last non-padding token for each item
                    last_indices = (attention_mask.sum(dim=-1) - 1).tolist()
                    self._current_last_indices = last_indices
                    
                    # Forward pass
                    self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            self.remove()


def run_activation_diagnostics(
    base_model: Any,
    trained_models: Dict[str, Any],
    probe_set: Dict[str, List[Dict[str, Any]]],
    tokenizer: Any,
    device: str = "cpu",
    layers: Sequence[int] = DIAGNOSTIC_LAYERS,
) -> Dict[str, Any]:
    """Execute activation diagnostics R(z) = ||\\Delta W z||_2 / (||W_0 z||_2 + 10^-8).
    
    Evaluates on 32 GSM8K test questions and 32 HellaSwag validation contexts in FP32.
    """
    gsm8k_prompts = [item["prompt"] for item in probe_set["gsm8k"]]
    hella_prompts = [item["prompt"] for item in probe_set["hellaswag"]]
    
    # 1. Cache starting model down_proj activations z for both sets
    cacher_gsm8k = ActivationProbeCacher(base_model, layer_indices=layers)
    cacher_gsm8k.cache_prompts(gsm8k_prompts, tokenizer=tokenizer, device=device)
    
    cacher_hella = ActivationProbeCacher(base_model, layer_indices=layers)
    cacher_hella.cache_prompts(hella_prompts, tokenizer=tokenizer, device=device)
    
    results: Dict[str, Any] = {
        "layers": list(layers),
        "conditions": {},
    }
    
    # 2. For each trained model condition, compute R(z) using its local Delta W
    for cond_name, trained_model in trained_models.items():
        cond_report: Dict[str, Any] = {}
        for layer_idx in layers:
            base_mod = extract_down_proj_layer(base_model, layer_idx)
            trained_mod = extract_down_proj_layer(trained_model, layer_idx)
            w0, delta_w = extract_weight_and_delta(base_mod, trained_mod)
            
            # Math prompts (GSM8K)
            z_math = cacher_gsm8k.cached_activations[layer_idx]
            math_ratios = [compute_activation_ratio(delta_w, w0, z) for z in z_math]
            
            # Non-math prompts (HellaSwag)
            z_hella = cacher_hella.cached_activations[layer_idx]
            hella_ratios = [compute_activation_ratio(delta_w, w0, z) for z in z_hella]
            
            cond_report[f"layer_{layer_idx}"] = {
                "math_mean_r": float(np.mean(math_ratios)) if math_ratios else 0.0,
                "math_std_r": float(np.std(math_ratios)) if math_ratios else 0.0,
                "non_math_mean_r": float(np.mean(hella_ratios)) if hella_ratios else 0.0,
                "non_math_std_r": float(np.std(hella_ratios)) if hella_ratios else 0.0,
            }
            
        results["conditions"][cond_name] = cond_report
        
    return results


def hellaswag_query(doc: Dict[str, Any]) -> str:
    """lm-evaluation-harness HellaSwag prompt (stock completion format, no chat template)."""
    def preprocess(text: str) -> str:
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        return text.replace("  ", " ")

    return preprocess(doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"].capitalize())


def gsm8k_chat_prompt(tokenizer: Any, question: str) -> str:
    """GSM8K benchmark format: gsm8k_cot ``Q: ...\\nA:`` inside the non-thinking chat template."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Q: {question}\nA:"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def cache_probe_activations(
    base_model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    layers: Sequence[int],
    device: str,
) -> Dict[int, List[np.ndarray]]:
    """Cache the starting model's down_proj inputs z at the last prompt token (one prompt at a time,
    so there is no padding and the last position is always the last prompt token)."""
    cacher = ActivationProbeCacher(base_model, layer_indices=layers)
    cacher.register()
    base_model.eval()
    try:
        with torch.no_grad():
            for text in prompts:
                enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
                cacher._current_last_indices = [enc["input_ids"].shape[1] - 1]
                base_model(**enc)
    finally:
        cacher.remove()
    return cacher.cached_activations


def run_task1_diagnostics(cfg: Dict[str, Any], conditions: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Weight-space (rho_k, eta) and activation (R(z)) diagnostics for down_proj in layers 0, 13, 27."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to execute diagnostics")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from task1_llm_ft.checkpoints import load_down_proj_deltas

    model_cfg = cfg.get("model", {})
    base_out = cfg.get("checkpoints", {}).get("output_dir", "./outputs/task1")
    res_dir = cfg.get("results_dir", "./results/task1")
    os.makedirs(res_dir, exist_ok=True)

    model_name = model_cfg.get("name_or_path", "Qwen/Qwen3-0.6B")
    seed = model_cfg.get("seed", 6304)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = list(DIAGNOSTIC_LAYERS)
    k = PROTECTION_RANK_K
    conds = list(conditions or ["full_ft_midpoint", "full_ft_final", "full_ft_l2_sp_final",
                                "lora_4_final", "lora_16_final", "oplora_16_final"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Probe set: sampled once with seed 6304 BEFORE looking at any model result, saved and reused
    probe_path = os.path.join(res_dir, "probe_set_seed6304.json")
    if os.path.exists(probe_path):
        with open(probe_path, "r", encoding="utf-8") as f:
            probe_set = json.load(f)
    else:
        gsm8k_raw = load_dataset("openai/gsm8k", "main", split="test")
        hella_raw = load_dataset("Rowan/hellaswag", split="validation")
        probe_set = sample_probe_set(
            gsm8k_questions=[dict(x) for x in gsm8k_raw],
            hellaswag_contexts=[dict(x) for x in hella_raw],
            num_samples_per_task=32, seed=seed, output_path=probe_path)
    math_prompts = [gsm8k_chat_prompt(tokenizer, it["question"]) for it in probe_set["gsm8k"]]
    other_prompts = [hellaswag_query(it["doc"]) for it in probe_set["hellaswag"]]

    # Starting model in FP32: W0, SVD bases and cached activations z
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    w0, bases = {}, {}
    for li in layers:
        w = extract_down_proj_layer(base_model, li).weight.detach().cpu().to(torch.float32).numpy()
        U, _, Vh = np.linalg.svd(w, full_matrices=False)
        w0[li], bases[li] = w, (U[:, :k], Vh[:k, :].T)
    z_math = cache_probe_activations(base_model, tokenizer, math_prompts, layers, device)
    z_other = cache_probe_activations(base_model, tokenizer, other_prompts, layers, device)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results: Dict[str, Any] = {"k": k, "layers": layers, "probe_set": probe_path, "conditions": {}}
    for cond in conds:
        deltas = load_down_proj_deltas(cond, base_out, layers, w0)
        cond_res: Dict[str, Any] = {}
        for li in layers:
            dW, W0 = deltas[li], w0[li]
            Uk, Vk = bases[li]
            norm_sq = float(np.sum(dW ** 2))
            rho_out = compute_subspace_interference(dW, Uk)
            rho_in = float(np.sum((dW @ Vk) ** 2) / norm_sq) if norm_sq > 1e-12 else 0.0
            r_math = np.array([compute_activation_ratio(dW, W0, z) for z in z_math[li]])
            r_other = np.array([compute_activation_ratio(dW, W0, z) for z in z_other[li]])
            cond_res[f"layer_{li}"] = {
                "rho_k": rho_out,                 # fraction of ||dW||^2 inside span(U_k)  (OPLoRA constraint -> ~0)
                "rho_k_input_side": rho_in,       # fraction inside span(V_k) on the input side (OPLoRA constraint -> ~0)
                "eta": compute_relative_update_magnitude(dW, W0),
                "delta_w_fro": float(np.sqrt(norm_sq)),
                "w0_fro": float(np.linalg.norm(W0)),
                "math_mean_r": float(r_math.mean()), "math_std_r": float(r_math.std()), "math_median_r": float(np.median(r_math)),
                "non_math_mean_r": float(r_other.mean()), "non_math_std_r": float(r_other.std()),
                "non_math_median_r": float(np.median(r_other)),
                "math_to_non_math_ratio": float(r_math.mean() / max(r_other.mean(), 1e-12)),
            }
        results["conditions"][cond] = cond_res

    with open(os.path.join(res_dir, "diagnostics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved diagnostics_summary.json")
    return results
