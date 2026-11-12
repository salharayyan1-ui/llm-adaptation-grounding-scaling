"""Checkpoint -> evaluation-model plumbing shared by benchmarks, diagnostics and qualitative runs.

Conventions written by ``trainer.save_checkpoint``:
  Full FT / L2-SP : <out>/<variant>/checkpoint_step_<N>/model_state.pt (FP32 state dict)
  LoRA            : .../adapter_state.pt           (trainable factors only)
  OPLoRA          : .../adapter_state.pt + oplora_projected_factors.pt (exported tilde-A / tilde-B)
Evaluation always starts from a FRESH FP16 backbone; adapters are merged into it (OPLoRA through its
exported projected factors) and then unwrapped, so every condition is scored by a vanilla HF model.
"""

from typing import Any, Dict, Optional, Sequence
import os

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from task1_llm_ft.models import (
    apply_lora_to_model,
    check_merge_equivalence,
    load_projected_factors_into_lora,
    load_trainable_state_dict,
    merge_adapters_to_model,
    unwrap_merged_adapters,
)

CONDITION_SPECS: Dict[str, Dict[str, Any]] = {
    "baseline": {"variant": None, "step": 0, "kind": "base"},
    "full_ft_midpoint": {"variant": "full_ft", "step": 625, "kind": "full"},
    "full_ft_final": {"variant": "full_ft", "step": 1250, "kind": "full"},
    "full_ft_l2_sp_final": {"variant": "full_ft_l2_sp", "step": 1250, "kind": "full"},
    "lora_4_final": {"variant": "lora_4", "step": 1250, "kind": "lora", "rank": 4, "alpha": 8.0},
    "lora_16_final": {"variant": "lora_16", "step": 1250, "kind": "lora", "rank": 16, "alpha": 32.0},
    "oplora_16_final": {"variant": "oplora_16", "step": 1250, "kind": "oplora", "rank": 16, "alpha": 32.0},
}
ALL_CONDITIONS = list(CONDITION_SPECS)


def checkpoint_dir(base_out_dir: str, cond: str) -> Optional[str]:
    spec = CONDITION_SPECS[cond]
    if spec["variant"] is None:
        return None
    return os.path.join(base_out_dir, spec["variant"], f"checkpoint_step_{spec['step']}")


def _require(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint file: {path} (train this variant first)")
    return path


def load_condition_model(
    cond: str,
    base_out_dir: str,
    model_name: str,
    device: str,
    dtype: Any = None,
    merge_check_inputs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Return (model, info): FP16, eval mode, adapters merged and unwrapped, on ``device``."""
    from transformers import AutoModelForCausalLM

    dtype = dtype or torch.float16
    spec = CONDITION_SPECS[cond]
    info: Dict[str, Any] = {"condition": cond, "variant": spec["variant"], "update": spec["step"]}
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)  # fresh backbone copy

    ckpt = checkpoint_dir(base_out_dir, cond)
    kind = spec["kind"]
    if kind == "full":
        state = torch.load(_require(os.path.join(ckpt, "model_state.pt")), map_location="cpu", weights_only=True)
        model.load_state_dict(state)  # FP32 master weights are cast to the FP16 evaluation dtype
        info["weights"] = "model_state.pt"
    elif kind in ("lora", "oplora"):
        model, _ = apply_lora_to_model(model, rank=spec["rank"], alpha=spec["alpha"], dropout=0.0)
        if kind == "lora":
            load_trainable_state_dict(model, torch.load(
                _require(os.path.join(ckpt, "adapter_state.pt")), map_location="cpu", weights_only=True))
            info["weights"] = "adapter_state.pt"
        else:  # OPLoRA: inference uses the exported projected factors, never the raw A/B
            load_projected_factors_into_lora(model, torch.load(
                _require(os.path.join(ckpt, "oplora_projected_factors.pt")), map_location="cpu", weights_only=True))
            info["weights"] = "oplora_projected_factors.pt"

    model.to(device)
    model.eval()
    if kind in ("lora", "oplora"):
        if merge_check_inputs is not None:
            info["merge_check"] = check_merge_equivalence(model, **{k: v.to(device) for k, v in merge_check_inputs.items()})
        else:
            merge_adapters_to_model(model)
        info["unwrapped_modules"] = unwrap_merged_adapters(model)
    return model, info


def make_merge_check_batch(tokenizer: Any, texts: Sequence[str]) -> Dict[str, Any]:
    enc = tokenizer(list(texts), return_tensors="pt", padding=True)
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def load_down_proj_deltas(cond: str, base_out_dir: str, layers: Sequence[int], w0: Dict[int, Any]) -> Dict[int, Any]:
    """Delta-W of ``mlp.down_proj`` (FP32, numpy) for each requested layer, read straight from the
    checkpoint tensors (no model instantiation): full fine-tuning = W_trained - W0, LoRA =
    (alpha/r) B A, OPLoRA = (alpha/r) tilde-B tilde-A from the exported projected factors."""
    spec = CONDITION_SPECS[cond]
    ckpt = checkpoint_dir(base_out_dir, cond)
    kind = spec["kind"]
    out: Dict[int, Any] = {}
    if kind == "full":
        state = torch.load(_require(os.path.join(ckpt, "model_state.pt")), map_location="cpu", mmap=True, weights_only=True)
        for li in layers:
            w = state[f"model.layers.{li}.mlp.down_proj.weight"].to(torch.float32).numpy()
            out[li] = w - w0[li]
    elif kind == "lora":
        state = torch.load(_require(os.path.join(ckpt, "adapter_state.pt")), map_location="cpu", weights_only=True)
        scaling = spec["alpha"] / spec["rank"]
        for li in layers:
            a = state[f"model.layers.{li}.mlp.down_proj.lora_A"].to(torch.float32)
            b = state[f"model.layers.{li}.mlp.down_proj.lora_B"].to(torch.float32)
            out[li] = (scaling * (b @ a)).numpy()
    elif kind == "oplora":
        fac = torch.load(_require(os.path.join(ckpt, "oplora_projected_factors.pt")), map_location="cpu", weights_only=True)
        for li in layers:
            f = fac[f"model.layers.{li}.mlp.down_proj"]
            out[li] = (f["scaling"] * (f["lora_B"].to(torch.float32) @ f["lora_A"].to(torch.float32))).numpy()
    else:
        raise ValueError(f"No weight delta for condition {cond}")
    return out
