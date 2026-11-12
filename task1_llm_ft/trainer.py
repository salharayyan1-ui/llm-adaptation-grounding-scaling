"""Training engine for Task 1: LLM Fine-Tuning.

Implements:
1. Training loop for:
   - Full FT (lr=2e-5, lambda=0)
   - Full FT + L2-SP (lr=2e-5, lambda=1e-3)
   - LoRA-4 (lr=2e-4, lambda=0, r=4, alpha=8)
   - LoRA-16 (lr=2e-4, lambda=0, r=16, alpha=32)
   - OPLoRA-16 (lr=2e-4, lambda=0, r=16, alpha=32, k=128)
2. Exact assignment constraints:
   - 40,000 examples, 1,250 optimizer updates
   - Microbatch 2, gradient accumulation 16, effective batch 32
   - AdamW: betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01 (excluding biases and 1D params)
   - Cosine decay schedule with 38 warmup updates
   - Gradient clipping at global norm 1.0 with unscaling
   - Dynamic gradient scaler (AMP FP16)
   - Gradient checkpointing enabled
   - Normalization of loss over ALL valid response tokens in the effective batch
   - Recovery checkpoints at 312, 625, 937, 1250 updates
   - Midpoint and final displacement tracking ||\\theta - \\theta_0||_2
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple, Any, Union
from dataclasses import dataclass, asdict
import os
import json
import csv
import time
import math
import random
import logging

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    autocast = None
    GradScaler = None
    HAS_TORCH = False

from common.seed import set_seed
from common.device import get_torch_device, get_peak_memory_mb, reset_peak_memory_stats
from task1_llm_ft.l2_sp import L2SPRegularizer, compute_parameter_displacement
from task1_llm_ft.models import (
    apply_lora_to_model,
    apply_oplora_to_model,
    count_parameters,
    export_all_projected_factors,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for Task 1 fine-tuning."""
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "./checkpoints/task1"
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
    
    # Precision & Performance
    fp16: bool = True
    gradient_checkpointing: bool = True
    enable_thinking: bool = False
    seed: int = 6304
    
    # Checkpoints
    checkpoint_updates: Tuple[int, ...] = (312, 625, 937, 1250)


def create_optimizer(model: Any, config: TrainingConfig) -> Any:
    """Create AdamW optimizer excluding biases and 1D parameters from weight decay."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for optimizer creation")
        
    decay_params: List[Any] = []
    no_decay_params: List[Any] = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude biases and 1D parameters (e.g. norm weights) from weight decay
        if param.ndim < 2 or "bias" in name or "norm" in name or "layernorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )
    return optimizer


def get_cosine_schedule_with_warmup_lr(
    update_step: int,
    warmup_updates: int,
    total_updates: int,
    peak_lr: float,
    min_lr: float = 0.0,
) -> float:
    """Compute learning rate using linear warmup followed by cosine decay."""
    if update_step < warmup_updates:
        return peak_lr * float(update_step) / float(max(1, warmup_updates))
    progress = float(update_step - warmup_updates) / float(max(1, total_updates - warmup_updates))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine_decay


def set_optimizer_lr(optimizer: Any, lr: float) -> None:
    """Set learning rate for all parameter groups in optimizer."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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
        
        set_seed(self.config.seed)
        self.device, self.has_cuda = get_torch_device()
        
        # Configure model precision and gradient checkpointing
        if HAS_TORCH:
            self._prepare_model()
            
        # Initialize L2-SP regularizer if requested
        self.regularizer: Optional[L2SPRegularizer] = None
        if self.config.method == "l2_sp" or self.config.lambda_l2sp > 0.0:
            self.regularizer = L2SPRegularizer(
                self.model,
                lambda_reg=self.config.lambda_l2sp,
                store_on_cpu=True,
            )
            
        # Displacement tracking reference (for Full FT and L2-SP)
        self.initial_state_dict: Optional[Dict[str, Any]] = None
        if self.config.method in ["full_ft", "l2_sp"] and HAS_TORCH:
            self.initial_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            }
            
        # Parameter counts
        self.param_counts = count_parameters(self.model) if HAS_TORCH else {}
        logger.info(f"Model parameters: {self.param_counts}")
        
        # Output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # History log
        self.history: List[Dict[str, Any]] = []
        self.skipped_updates = 0
        
    def _prepare_model(self) -> None:
        """Setup model precision, adapters, and gradient checkpointing."""
        if not HAS_TORCH:
            return
            
        # Target module injection based on method
        if self.config.method == "lora_4":
            self.config.lora_rank = 4
            self.config.lora_alpha = 8.0
            self.model, self.adapters = apply_lora_to_model(
                self.model,
                rank=4,
                alpha=8.0,
                dropout=self.config.lora_dropout,
            )
        elif self.config.method == "lora_16":
            self.config.lora_rank = 16
            self.config.lora_alpha = 32.0
            self.model, self.adapters = apply_lora_to_model(
                self.model,
                rank=16,
                alpha=32.0,
                dropout=self.config.lora_dropout,
            )
        elif self.config.method == "oplora_16":
            self.config.lora_rank = 16
            self.config.lora_alpha = 32.0
            self.model, self.adapters, self.oplora_stats = apply_oplora_to_model(
                self.model,
                rank=16,
                alpha=32.0,
                protection_rank=self.config.oplora_protection_rank,
                dropout=self.config.lora_dropout,
            )
        elif self.config.method in ["full_ft", "l2_sp"]:
            # Update all parameters
            for p in self.model.parameters():
                p.requires_grad = True
                
        # Gradient checkpointing
        if self.config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
                
        self.model.to(self.device)
        
    def train(self) -> Dict[str, Any]:
        """Execute fine-tuning over 1,250 updates (40,000 examples)."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for training")
            
        self.model.train()
        reset_peak_memory_stats()
        
        optimizer = create_optimizer(self.model, self.config)
        scaler = GradScaler(enabled=self.config.fp16 and self.has_cuda)
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        total_examples = self.config.num_examples
        max_updates = self.config.max_updates
        accum_steps = self.config.gradient_accumulation_steps
        
        logger.info(
            f"Starting training run: method={self.config.method}, "
            f"budget={total_examples} examples, updates={max_updates}, "
            f"microbatch={self.config.micro_batch_size}, accum={accum_steps}, "
            f"lr={self.config.learning_rate}, lambda={self.config.lambda_l2sp}"
        )
        
        start_time = time.time()
        update_step = 0
        examples_seen = 0
        
        data_iter = iter(self.train_dataloader)
        
        while update_step < max_updates:
            current_lr = get_cosine_schedule_with_warmup_lr(
                update_step=update_step,
                warmup_updates=self.config.warmup_updates,
                total_updates=max_updates,
                peak_lr=self.config.learning_rate,
            )
            set_optimizer_lr(optimizer, current_lr)
            
            optimizer.zero_grad()
            
            # 1. Collect microbatches for the current effective batch
            effective_batch_items: List[Dict[str, Any]] = []
            for _ in range(accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                effective_batch_items.append(batch)
                
            if not effective_batch_items:
                break
                
            # Compute total valid response tokens across all microbatches in this effective batch
            # for exact loss normalization over all valid response tokens in the effective batch
            total_response_tokens = sum(b["num_valid_tokens"] for b in effective_batch_items)
            norm_token_denom = max(1, total_response_tokens)
            
            accum_sft_loss = 0.0
            accum_omega = 0.0
            accum_total_loss = 0.0
            
            for microbatch in effective_batch_items:
                input_ids = microbatch["input_ids"].to(self.device)
                attention_mask = microbatch["attention_mask"].to(self.device)
                labels = microbatch["labels"].to(self.device)
                
                with autocast(enabled=self.config.fp16 and self.has_cuda, dtype=torch.float16):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # Next-token prediction cross entropy loss on valid response tokens
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    # Reduction='sum' over valid tokens in microbatch
                    vocab_size = shift_logits.size(-1)
                    token_loss_sum = F.cross_entropy(
                        shift_logits.view(-1, vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    
                    # Normalized by total response tokens in effective batch
                    loss_sft_micro = token_loss_sum / norm_token_denom
                    
                    # Add L2-SP penalty if applicable
                    if self.regularizer is not None and self.config.lambda_l2sp > 0.0:
                        penalty_tensor, omega_val = self.regularizer.compute_penalty(self.model)
                        loss_l2_micro = (self.config.lambda_l2sp * penalty_tensor) / accum_steps
                        micro_total_loss = loss_sft_micro + loss_l2_micro
                        accum_omega += omega_val / accum_steps
                    else:
                        micro_total_loss = loss_sft_micro
                        
                accum_sft_loss += float(loss_sft_micro.item())
                accum_total_loss += float(micro_total_loss.item())
                
                scaler.scale(micro_total_loss).backward()
                examples_seen += len(input_ids)
                
            # Optimizer update after accumulating 16 microbatches
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm))
            
            # Check if gradient is finite before stepping
            old_scaler_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            
            # Detect skipped updates
            if scaler.get_scale() < old_scaler_scale:
                self.skipped_updates += 1
                logger.warning(f"Update {update_step + 1} skipped due to non-finite gradients (GradScaler).")
                
            update_step += 1
            
            # Displacement at midpoint (625) and final (1250)
            displacement = 0.0
            if update_step in [625, 1250] and self.initial_state_dict is not None:
                displacement = compute_parameter_displacement(self.model, self.initial_state_dict)
                logger.info(f"Update {update_step}: Full-model displacement ||theta - theta_0||_2 = {displacement:.6f}")
                
            # Log progress
            step_record = {
                "update": update_step,
                "examples_seen": examples_seen,
                "lr": current_lr,
                "sft_loss": accum_sft_loss,
                "grad_norm": grad_norm,
                "peak_mem_mb": get_peak_memory_mb(),
            }
            if self.regularizer is not None:
                step_record["omega"] = accum_omega
                step_record["lambda_omega"] = self.config.lambda_l2sp * accum_omega
                step_record["total_loss"] = accum_total_loss
                
            if displacement > 0.0:
                step_record["displacement"] = displacement
                
            self.history.append(step_record)
            
            if update_step % 25 == 0 or update_step == 1:
                log_str = (
                    f"Update {update_step}/{max_updates} | "
                    f"Examples: {examples_seen} | "
                    f"SFT Loss: {accum_sft_loss:.4f} | "
                    f"LR: {current_lr:.3e} | "
                    f"GradNorm: {grad_norm:.3f}"
                )
                if self.regularizer is not None:
                    log_str += f" | Omega: {accum_omega:.2f} | TotLoss: {accum_total_loss:.4f}"
                logger.info(log_str)
                
            # Recovery Checkpoints: 312, 625, 937, 1250 updates
            if update_step in self.config.checkpoint_updates:
                self.save_checkpoint(
                    update_step=update_step,
                    examples_seen=examples_seen,
                    optimizer=optimizer,
                    scaler=scaler,
                )
                
        total_time = time.time() - start_time
        peak_gpu_mem = get_peak_memory_mb()
        
        summary = {
            "method": self.config.method,
            "total_updates": update_step,
            "examples_seen": examples_seen,
            "total_time_sec": total_time,
            "peak_memory_mb": peak_gpu_mem,
            "skipped_updates": self.skipped_updates,
            "trainable_parameters": self.param_counts.get("trainable", 0),
            "total_parameters": self.param_counts.get("total", 0),
        }
        
        # Save training curves and summary
        self.save_training_logs(summary)
        return summary
        
    def save_checkpoint(
        self,
        update_step: int,
        examples_seen: int,
        optimizer: Any,
        scaler: Any,
    ) -> str:
        """Save recovery checkpoint containing weights, optimizer, scaler, RNG, and history."""
        ckpt_dir = os.path.join(self.config.output_dir, f"checkpoint_step_{update_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # State dict
        state_dict_path = os.path.join(ckpt_dir, "model_state.pt")
        torch.save(self.model.state_dict(), state_dict_path)
        
        # Optimizer and scaler state
        opt_path = os.path.join(ckpt_dir, "optimizer_state.pt")
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "update_step": update_step,
            "examples_seen": examples_seen,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "config": asdict(self.config),
        }, opt_path)
        
        # If OPLoRA, export projected factors
        if self.config.method == "oplora_16":
            projected = export_all_projected_factors(self.model)
            torch.save(projected, os.path.join(ckpt_dir, "oplora_projected_factors.pt"))
            
        ckpt_size_mb = (
            os.path.getsize(state_dict_path) + os.path.getsize(opt_path)
        ) / (1024 * 1024)
        
        logger.info(
            f"Saved recovery checkpoint at update {update_step} ({examples_seen} examples) "
            f"to {ckpt_dir} ({ckpt_size_mb:.2f} MB)"
        )
        return ckpt_dir
        
    def save_training_logs(self, summary: Dict[str, Any]) -> None:
        """Save history to JSON and CSV."""
        json_path = os.path.join(self.config.output_dir, "training_history.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "config": asdict(self.config),
                "history": self.history,
            }, f, indent=2)
            
        csv_path = os.path.join(self.config.output_dir, "training_history.csv")
        if self.history:
            keys = list(self.history[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)
                
        logger.info(f"Saved training history to {json_path} and {csv_path}")


def run_task1_training(variant: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level training runner for scripts/run_task1.py and configs/task1_config.yaml."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to execute training")
        
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(f"HuggingFace transformers/datasets required: {e}")
        
    from task1_llm_ft.dataset import (
        filter_metamath_dataset,
        build_gsm8k_decontamination_set,
        MetaMathQADataset,
        create_dataloader,
    )
    
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    opt_cfg = cfg.get("optimizer", {})
    exp_cfg = cfg.get("experiments", {}).get(variant, {})
    ckpt_cfg = cfg.get("checkpoints", {})
    
    model_name = model_cfg.get("name_or_path", "Qwen/Qwen3-0.6B")
    seed = model_cfg.get("seed", 6304)
    set_seed(seed)
    
    # Output directory for this variant
    base_out_dir = ckpt_cfg.get("output_dir", "./outputs/task1")
    variant_out_dir = os.path.join(base_out_dir, variant)
    os.makedirs(variant_out_dir, exist_ok=True)
    
    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Dataset loading and decontamination filtering
    logger.info("Loading MetaMathQA dataset...")
    raw_ds = load_dataset(data_cfg.get("dataset_name", "meta-math/MetaMathQA"), split="train")
    
    decontam_set = set()
    if data_cfg.get("filter_gsm8k_contamination", True):
        logger.info("Building GSM8K contamination exclusion set...")
        decontam_set = build_gsm8k_decontamination_set()
        
    filtered_examples = filter_metamath_dataset(
        raw_dataset=raw_ds,
        tokenizer=tokenizer,
        gsm8k_decontamination_set=decontam_set,
        target_count=data_cfg.get("max_examples", 40000),
        max_seq_length=data_cfg.get("max_seq_length", 2048),
        enable_thinking=model_cfg.get("enable_thinking", False),
    )
    
    dataset = MetaMathQADataset(filtered_examples)
    dataloader = create_dataloader(
        dataset,
        batch_size=data_cfg.get("micro_batch_size", 2),
        pad_token_id=tokenizer.pad_token_id or 0,
        shuffle=False,  # source order strictly preserved
    )
    
    # 3. Model loading
    torch_dtype = torch.float16 if model_cfg.get("fp16", True) else torch.float32
    logger.info(f"Loading base model {model_name} in {torch_dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    
    # 4. Map configuration
    method_name = "l2_sp" if variant == "full_ft_l2_sp" else variant
    training_config = TrainingConfig(
        model_name_or_path=model_name,
        output_dir=variant_out_dir,
        method=method_name,
        num_examples=data_cfg.get("max_examples", 40000),
        max_updates=opt_cfg.get("total_updates", 1250),
        micro_batch_size=data_cfg.get("micro_batch_size", 2),
        gradient_accumulation_steps=data_cfg.get("gradient_accumulation_steps", 16),
        max_seq_length=data_cfg.get("max_seq_length", 2048),
        learning_rate=exp_cfg.get("lr", 2e-5),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
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
    )
    
    # 5. Execute training
    trainer = Task1Trainer(
        config=training_config,
        model=model,
        train_dataloader=dataloader,
        tokenizer=tokenizer,
    )
    return trainer.train()

