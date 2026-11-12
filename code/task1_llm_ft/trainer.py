"""Training engine for Task 1: LLM Fine-Tuning.

Implements the five variants (Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, OPLoRA-16) with the
assignment's exact shared recipe:

- 40,000 examples, 1,250 optimizer updates, microbatch 2 x accumulation 16 (effective batch 32)
- AdamW(0.9, 0.999, eps 1e-8), weight decay 0.01 (biases / <2-D params excluded), cosine decay,
  38 warmup updates, global-norm clipping 1.0 applied after unscaling
- FP16 autocast + dynamic GradScaler. Full FT / L2-SP keep FP32 master parameters; adapter variants
  keep a frozen FP16 backbone with FP32 adapters. Optimizer state is FP32 (parameters are FP32).
- Loss: summed next-token cross-entropy over assistant tokens + <|im_end|>, normalized by the number
  of valid tokens of the whole effective batch
- L2-SP: lambda * 0.5 * ||theta - theta_0||^2 over all unique parameters (tied counted once),
  applied as an exact gradient term once per update; Omega, lambda*Omega and total loss are logged
- Recovery checkpoints at updates 312, 625, 937, 1250 (weights / adapters, optimizer, GradScaler,
  RNG, data position) and a real ``resume_from`` path
"""

from typing import Dict, List, Optional, Sequence, Tuple, Any
from dataclasses import dataclass, asdict
import csv
import hashlib
import itertools
import json
import logging
import math
import os
import random
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

from common.seed import set_seed
from common.device import get_torch_device, get_peak_memory_mb, reset_peak_memory_stats
from task1_llm_ft.l2_sp import L2SPRegularizer, compute_parameter_displacement
from task1_llm_ft.models import (
    apply_lora_to_model,
    apply_oplora_to_model,
    count_parameters,
    export_all_projected_factors,
    trainable_state_dict,
)

logger = logging.getLogger(__name__)

FULL_FT_METHODS = ("full_ft", "l2_sp")


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for Task 1 fine-tuning."""
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "./outputs/task1"          # checkpoints (large, git-ignored)
    results_dir: str = "./results/task1"         # small machine-readable logs (committed)
    method: str = "full_ft"  # full_ft, l2_sp, lora_4, lora_16, oplora_16

    # Dataset and updates
    num_examples: int = 40000
    max_updates: int = 1250
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 16  # effective batch = 32
    max_seq_length: int = 2048

    # Optimizer and schedule
    learning_rate: float = 2e-5  # 2e-5 for full FT / L2-SP, 2e-4 for LoRA / OPLoRA
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    warmup_updates: int = 38
    max_grad_norm: float = 1.0

    # Regularization
    lambda_l2sp: float = 0.0  # 1e-3 for l2_sp

    # LoRA / OPLoRA parameters
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    oplora_protection_rank: int = 128

    # Precision & performance
    fp16: bool = True
    gradient_checkpointing: bool = True
    enable_thinking: bool = False
    selective_head: bool = True      # apply lm_head only at supervised positions (same loss, less memory)
    seed: int = 6304

    # Checkpoints
    checkpoint_updates: Tuple[int, ...] = (312, 625, 937, 1250)
    displacement_updates: Tuple[int, ...] = (625, 1250)
    save_optimizer_state: bool = True
    l2sp_reference_on_cpu: bool = False   # False: keep theta_0 on the GPU (fits 24 GB); True: CPU + transfers


def create_optimizer(model: Any, config: TrainingConfig) -> Any:
    """AdamW excluding biases and parameters with fewer than two dimensions from weight decay."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for optimizer creation")

    decay_params: List[Any] = []
    no_decay_params: List[Any] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name or "layernorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )


def get_cosine_schedule_with_warmup_lr(
    update_step: int,
    warmup_updates: int,
    total_updates: int,
    peak_lr: float,
    min_lr: float = 0.0,
) -> float:
    """Linear warmup followed by cosine decay (``update_step`` = number of completed updates)."""
    if update_step < warmup_updates:
        return peak_lr * float(update_step) / float(max(1, warmup_updates))
    progress = float(update_step - warmup_updates) / float(max(1, total_updates - warmup_updates))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine_decay


def set_optimizer_lr(optimizer: Any, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _file_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0


class Task1Trainer:
    """Complete trainer for Task 1 LLM fine-tuning experiments."""

    def __init__(
        self,
        config: TrainingConfig,
        model: Any,
        train_dataloader: Any,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.tokenizer = tokenizer
        self.adapters: Dict[str, Any] = {}
        self.oplora_stats: Dict[str, float] = {}

        set_seed(self.config.seed)
        self.device, self.has_cuda = get_torch_device()

        self.setup_time_sec = 0.0
        t0 = time.perf_counter()
        if HAS_TORCH:
            self._prepare_model()
        self.setup_time_sec = time.perf_counter() - t0

        # theta_0 for L2-SP (and displacement tracking for Full FT)
        self.regularizer: Optional[L2SPRegularizer] = None
        self.initial_params: Optional[Dict[str, Any]] = None
        if HAS_TORCH and (self.config.method == "l2_sp" or self.config.lambda_l2sp > 0.0):
            self.regularizer = L2SPRegularizer(
                self.model,
                lambda_reg=self.config.lambda_l2sp,
                store_on_cpu=self.config.l2sp_reference_on_cpu,
            )
        elif HAS_TORCH and self.config.method == "full_ft":
            self.initial_params = {n: p.detach().cpu().clone() for n, p in self.model.named_parameters()}

        self.param_counts = count_parameters(self.model) if HAS_TORCH else {}
        logger.info(f"Model parameters: {self.param_counts}")

        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)

        self.history: List[Dict[str, Any]] = []
        self.skipped_updates: List[int] = []
        self.checkpoint_records: List[Dict[str, Any]] = []
        self.checkpoint_time_sec = 0.0
        self._resume_state: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ model setup
    def _prepare_model(self) -> None:
        """Set precision, inject adapters, enable gradient checkpointing, move to the device."""
        method = self.config.method
        if method in FULL_FT_METHODS:
            self.model.float()  # FP32 master parameters; compute runs under FP16 autocast
            for p in self.model.parameters():
                p.requires_grad = True
            self.model.to(self.device)
        else:
            # Frozen FP16 backbone. Move first so OPLoRA's FP32 SVD runs on the GPU.
            self.model.to(self.device)
            if method == "lora_4":
                self.config.lora_rank, self.config.lora_alpha = 4, 8.0
                self.model, self.adapters = apply_lora_to_model(
                    self.model, rank=4, alpha=8.0, dropout=self.config.lora_dropout)
            elif method == "lora_16":
                self.config.lora_rank, self.config.lora_alpha = 16, 32.0
                self.model, self.adapters = apply_lora_to_model(
                    self.model, rank=16, alpha=32.0, dropout=self.config.lora_dropout)
            elif method == "oplora_16":
                self.config.lora_rank, self.config.lora_alpha = 16, 32.0
                self.model, self.adapters, self.oplora_stats = apply_oplora_to_model(
                    self.model, rank=16, alpha=32.0,
                    protection_rank=self.config.oplora_protection_rank, dropout=self.config.lora_dropout)
            else:
                raise ValueError(f"Unknown method: {method}")
            self.model.to(self.device)  # adapters (FP32) onto the device; backbone stays FP16

        if self.config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

    # ------------------------------------------------------------------ loss
    def _microbatch_loss(self, input_ids: Any, attention_mask: Any, labels: Any, denom: int) -> Any:
        """Summed token cross-entropy of one microbatch divided by the effective batch's valid tokens."""
        labels_shift = torch.full_like(labels, -100)
        labels_shift[:, :-1] = labels[:, 1:]

        inner = getattr(self.model, "model", None)
        head = getattr(self.model, "lm_head", None)
        if self.config.selective_head and inner is not None and head is not None:
            hidden = inner(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
            mask = labels_shift != -100
            if not bool(mask.any()):
                return hidden.sum() * 0.0
            logits = head(hidden[mask])
            loss_sum = F.cross_entropy(logits.float(), labels_shift[mask], reduction="sum")
        else:
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss_sum = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels_shift.view(-1), ignore_index=-100, reduction="sum")
        return loss_sum / denom

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.config.fp16 and self.has_cuda)

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, update_step: int, examples_seen: int, optimizer: Any, scaler: Any) -> str:
        t0 = time.perf_counter()
        ckpt_dir = os.path.join(self.config.output_dir, f"checkpoint_step_{update_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        record: Dict[str, Any] = {"update": update_step, "examples_seen": examples_seen}

        if self.config.method in FULL_FT_METHODS:
            weights_path = os.path.join(ckpt_dir, "model_state.pt")
            torch.save(self.model.state_dict(), weights_path)
            record["weights_file"] = "model_state.pt"
        else:
            weights_path = os.path.join(ckpt_dir, "adapter_state.pt")
            torch.save(trainable_state_dict(self.model), weights_path)  # adapter factors only
            record["weights_file"] = "adapter_state.pt"
        record["weights_mb"] = _file_mb(weights_path)

        if self.config.method == "oplora_16":
            t_exp = time.perf_counter()
            projected = export_all_projected_factors(self.model)
            proj_path = os.path.join(ckpt_dir, "oplora_projected_factors.pt")
            torch.save(projected, proj_path)
            record["projected_export_sec"] = time.perf_counter() - t_exp
            record["projected_factors_mb"] = _file_mb(proj_path)

        opt_path = os.path.join(ckpt_dir, "optimizer_state.pt")
        if self.config.save_optimizer_state:
            torch.save({
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "update_step": update_step,
                "examples_seen": examples_seen,
                "data_position_microbatches": update_step * self.config.gradient_accumulation_steps,
                "skipped_updates": self.skipped_updates,
                "history": self.history,
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                "config": asdict(self.config),
            }, opt_path)
        record["optimizer_mb"] = _file_mb(opt_path)
        record["total_mb"] = record["weights_mb"] + record["optimizer_mb"] + record.get("projected_factors_mb", 0.0)

        elapsed = time.perf_counter() - t0
        self.checkpoint_time_sec += elapsed
        record["save_sec"] = elapsed
        self.checkpoint_records.append(record)
        logger.info(f"Saved recovery checkpoint at update {update_step} to {ckpt_dir}: {record}")
        return ckpt_dir

    def load_checkpoint(self, ckpt_dir: str, optimizer: Any, scaler: Any) -> int:
        """Restore weights / adapters, optimizer, GradScaler and RNG; returns the completed update count."""
        method = self.config.method
        if method in FULL_FT_METHODS:
            state = torch.load(os.path.join(ckpt_dir, "model_state.pt"), map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
        else:
            from task1_llm_ft.models import load_trainable_state_dict
            load_trainable_state_dict(self.model, torch.load(
                os.path.join(ckpt_dir, "adapter_state.pt"), map_location="cpu", weights_only=True))
        ck = torch.load(os.path.join(ckpt_dir, "optimizer_state.pt"), map_location="cpu", weights_only=False)
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        if ck["rng"].get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        self.history = ck["history"]
        self.skipped_updates = ck["skipped_updates"]
        self._resume_state = ck
        logger.info(f"Resumed from {ckpt_dir} at update {ck['update_step']}")
        return int(ck["update_step"])

    # ------------------------------------------------------------------ training
    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Execute fine-tuning over 1,250 updates (40,000 examples)."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for training")

        cfg = self.config
        self.model.train()
        reset_peak_memory_stats()

        optimizer = create_optimizer(self.model, cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and self.has_cuda)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        accum_steps = cfg.gradient_accumulation_steps

        update_step = 0
        examples_seen = 0
        if resume_from:
            update_step = self.load_checkpoint(resume_from, optimizer, scaler)
            examples_seen = int(self._resume_state["examples_seen"])

        logger.info(
            f"Starting run: method={cfg.method}, updates={cfg.max_updates}, microbatch={cfg.micro_batch_size}, "
            f"accum={accum_steps}, lr={cfg.learning_rate}, lambda={cfg.lambda_l2sp}, start_update={update_step}")

        data_iter = iter(self.train_dataloader)
        if update_step > 0:  # fast-forward the (unshuffled) data order to the saved position
            for _ in itertools.islice(data_iter, update_step * accum_steps):
                pass

        start_time = time.time()
        ckpt_time_at_start = self.checkpoint_time_sec

        while update_step < cfg.max_updates:
            current_lr = get_cosine_schedule_with_warmup_lr(update_step, cfg.warmup_updates, cfg.max_updates, cfg.learning_rate)
            set_optimizer_lr(optimizer, current_lr)
            optimizer.zero_grad(set_to_none=True)

            batches = []
            for _ in range(accum_steps):
                try:
                    batches.append(next(data_iter))
                except StopIteration:
                    break
            if not batches:
                break

            denom = max(1, sum(b["num_valid_tokens"] for b in batches))  # valid tokens of the effective batch
            sft_loss = 0.0

            for mb in batches:
                input_ids = mb["input_ids"].to(self.device)
                attention_mask = mb["attention_mask"].to(self.device)
                labels = mb["labels"].to(self.device)
                with self._autocast():
                    loss = self._microbatch_loss(input_ids, attention_mask, labels, denom)
                sft_loss += float(loss.detach().item())
                scaler.scale(loss).backward()
                examples_seen += int(input_ids.shape[0])

            scaler.unscale_(optimizer)
            omega = None
            if self.regularizer is not None:
                omega = self.regularizer.add_penalty_gradient(self.model)  # + lambda * (theta - theta_0)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm))

            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            update_step += 1
            if skipped:
                self.skipped_updates.append(update_step)
                logger.warning(f"Update {update_step} skipped: non-finite gradients (GradScaler scale {old_scale} -> {scaler.get_scale()}).")

            record: Dict[str, Any] = {
                "update": update_step,
                "examples_seen": examples_seen,
                "lr": current_lr,
                "sft_loss": sft_loss,
                "grad_norm": grad_norm,
                "skipped": bool(skipped),
                "loss_scale": float(scaler.get_scale()) if scaler.is_enabled() else 1.0,
                "peak_mem_mb": get_peak_memory_mb(),
            }
            if omega is not None:
                record["omega"] = omega
                record["lambda_omega"] = cfg.lambda_l2sp * omega
                record["total_loss"] = sft_loss + cfg.lambda_l2sp * omega

            if update_step in cfg.displacement_updates:
                if self.regularizer is not None:
                    record["displacement"] = self.regularizer.compute_displacement(self.model)
                elif self.initial_params is not None:
                    record["displacement"] = compute_parameter_displacement(self.model, self.initial_params)
                if "displacement" in record:
                    logger.info(f"Update {update_step}: ||theta - theta_0||_2 = {record['displacement']:.6f}")

            self.history.append(record)

            if update_step % 25 == 0 or update_step == 1:
                msg = (f"Update {update_step}/{cfg.max_updates} | ex {examples_seen} | sft {sft_loss:.4f} | "
                       f"lr {current_lr:.2e} | gnorm {grad_norm:.2f} | mem {record['peak_mem_mb']:.0f}MB")
                if omega is not None:
                    msg += f" | Omega {omega:.3f} | total {record['total_loss']:.4f}"
                logger.info(msg)

            if update_step in cfg.checkpoint_updates:
                self.save_checkpoint(update_step, examples_seen, optimizer, scaler)

        total_time = time.time() - start_time
        train_only_time = total_time - (self.checkpoint_time_sec - ckpt_time_at_start)
        summary = {
            "method": cfg.method,
            "total_updates": update_step,
            "examples_seen": examples_seen,
            "total_time_sec": total_time,
            "train_time_excl_checkpoints_sec": train_only_time,
            "checkpoint_time_sec": self.checkpoint_time_sec - ckpt_time_at_start,
            "setup_time_sec": self.setup_time_sec,
            "peak_memory_mb": get_peak_memory_mb(),
            "skipped_updates": len(self.skipped_updates),
            "skipped_update_indices": self.skipped_updates,
            "trainable_parameters": self.param_counts.get("trainable", 0),
            "total_parameters": self.param_counts.get("total", 0),
            "checkpoints": self.checkpoint_records,
            "oplora": self.oplora_stats or None,
            "l2sp": self.regularizer.get_stats() if self.regularizer is not None else None,
            "resumed_from": resume_from,
        }
        self.save_training_logs(summary)
        return summary

    def save_training_logs(self, summary: Dict[str, Any]) -> None:
        """Write learning curves / resource summary to results_dir (small, committed files)."""
        json_path = os.path.join(self.config.results_dir, "training_history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "config": asdict(self.config), "history": self.history}, f, indent=2)

        csv_path = os.path.join(self.config.results_dir, "training_history.csv")
        if self.history:
            keys: List[str] = []
            for rec in self.history:
                for k in rec:
                    if k not in keys:
                        keys.append(k)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)
        logger.info(f"Saved training history to {json_path}")


def _dtype_for(method: str, fp16: bool) -> Any:
    """FP32 master weights for full fine-tuning variants; frozen FP16 backbone for adapters."""
    if method in FULL_FT_METHODS:
        return torch.float32
    return torch.float16 if fp16 else torch.float32


def run_task1_training(variant: str, cfg: Dict[str, Any], resume_from: Optional[str] = None) -> Dict[str, Any]:
    """Top-level training runner for scripts/run_task1.py and configs/task1_config.yaml."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to execute training")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    from task1_llm_ft.dataset import (
        filter_metamath_dataset,
        build_gsm8k_decontamination_set,
        MetaMathQADataset,
        create_dataloader,
    )

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    opt_cfg = cfg.get("optimizer", {})
    exp_cfg = cfg.get("experiments", {}).get(variant)
    if exp_cfg is None:
        raise KeyError(f"Unknown variant '{variant}' (configs/task1_config.yaml -> experiments)")
    ckpt_cfg = cfg.get("checkpoints", {})

    model_name = model_cfg.get("name_or_path", "Qwen/Qwen3-0.6B")
    seed = model_cfg.get("seed", 6304)
    set_seed(seed)

    variant_out_dir = os.path.join(ckpt_cfg.get("output_dir", "./outputs/task1"), variant)
    variant_results_dir = os.path.join(cfg.get("results_dir", "./results/task1"), variant)
    os.makedirs(variant_out_dir, exist_ok=True)
    os.makedirs(variant_results_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading MetaMathQA...")
    raw_ds = load_dataset(data_cfg.get("dataset_name", "meta-math/MetaMathQA"), split="train")

    decontam_set = set()
    if data_cfg.get("filter_gsm8k_contamination", True):
        decontam_set = build_gsm8k_decontamination_set()
        if not decontam_set:
            raise RuntimeError("GSM8K test questions could not be loaded: refusing to train without decontamination.")

    examples = filter_metamath_dataset(
        raw_dataset=raw_ds,
        tokenizer=tokenizer,
        gsm8k_decontamination_set=decontam_set,
        target_count=data_cfg.get("max_examples", 40000),
        max_seq_length=data_cfg.get("max_seq_length", 2048),
        enable_thinking=model_cfg.get("enable_thinking", False),
    )
    if len(examples) != data_cfg.get("max_examples", 40000):
        raise RuntimeError(f"Only {len(examples)} eligible examples found")

    # Fingerprint: all five runs must see identical examples, order and tokenization
    fingerprint = hashlib.sha256(json.dumps([(e["id"], e["length"], sum(e["input_ids"]) % 1000003) for e in examples]).encode()).hexdigest()
    ids_path = os.path.join(cfg.get("results_dir", "./results/task1"), "train_example_ids.json")
    if not os.path.exists(ids_path):
        os.makedirs(os.path.dirname(ids_path), exist_ok=True)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fingerprint, "n": len(examples), "source_indices": [e["id"] for e in examples]}, f)
    else:
        with open(ids_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        if prev["fingerprint"] != fingerprint:
            raise RuntimeError("Training examples differ from an earlier run (train_example_ids.json fingerprint mismatch).")

    dataloader = create_dataloader(
        MetaMathQADataset(examples),
        batch_size=data_cfg.get("micro_batch_size", 2),
        pad_token_id=tokenizer.pad_token_id or 0,
        shuffle=False,  # source order strictly preserved
    )

    method_name = "l2_sp" if variant == "full_ft_l2_sp" else variant
    dtype = _dtype_for(method_name, model_cfg.get("fp16", True))
    logger.info(f"Loading {model_name} as {dtype} for variant {variant}")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    training_config = TrainingConfig(
        model_name_or_path=model_name,
        output_dir=variant_out_dir,
        results_dir=variant_results_dir,
        method=method_name,
        num_examples=data_cfg.get("max_examples", 40000),
        max_updates=opt_cfg.get("total_updates", 1250),
        micro_batch_size=data_cfg.get("micro_batch_size", 2),
        gradient_accumulation_steps=data_cfg.get("gradient_accumulation_steps", 16),
        max_seq_length=data_cfg.get("max_seq_length", 2048),
        learning_rate=exp_cfg.get("lr", 2e-5),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
        adam_beta1=opt_cfg.get("betas", [0.9, 0.999])[0],
        adam_beta2=opt_cfg.get("betas", [0.9, 0.999])[1],
        adam_eps=opt_cfg.get("eps", 1e-8),
        warmup_updates=opt_cfg.get("warmup_updates", 38),
        max_grad_norm=opt_cfg.get("grad_clip_norm", 1.0),
        lambda_l2sp=exp_cfg.get("lambda_l2_sp", 0.0),
        lora_rank=exp_cfg.get("rank", 16),
        lora_alpha=exp_cfg.get("alpha", 32.0),
        lora_dropout=exp_cfg.get("dropout", 0.05),
        oplora_protection_rank=exp_cfg.get("protection_rank_k", 128),
        fp16=model_cfg.get("fp16", True),
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", True),
        enable_thinking=model_cfg.get("enable_thinking", False),
        seed=seed,
        checkpoint_updates=tuple(ckpt_cfg.get("save_updates", [312, 625, 937, 1250])),
        save_optimizer_state=ckpt_cfg.get("save_optimizer_state", True),
        l2sp_reference_on_cpu=exp_cfg.get("reference_on_cpu", False),
    )

    trainer = Task1Trainer(config=training_config, model=model, train_dataloader=dataloader, tokenizer=tokenizer)
    summary = trainer.train(resume_from=resume_from)
    summary["data_fingerprint"] = fingerprint
    with open(os.path.join(variant_results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
