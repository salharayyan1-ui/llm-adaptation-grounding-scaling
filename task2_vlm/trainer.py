"""Curriculum Training Engine for Task 2 (Stages 1, 2, and 3).

Executes:
- 4,500 total updates across 3 stages (Stage 1: 2,250; Stage 2: 1,500; Stage 3: 750).
- Microbatch 8, grad accum 2 -> effective batch 16.
- AdamW (weight decay 0.01, grad clip 1.0).
- Per-stage 5% warmup, then cosine decay to 10% of peak LR (not zero).
- Example-weighted causal cross-entropy loss:
    L = - (1 / |B|) sum_{i in B} (1 / |R_i|) sum_{t in R_i} log p(y_{i,t} | I_i, q_i, y_{i,<t}).
- Resumable checkpoints every 250 updates and stage endpoint weight saves.
- Dev evaluation tracking caption loss and VQA answer NLL under Correct, Swap, and No-image,
  computing Image-Swap Penalty Delta NLL = NLL_partner - NLL_correct.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import time
import math
import json
import logging
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.cuda.amp import autocast, GradScaler
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    F = None
    AdamW = None
    autocast = None
    GradScaler = None
    HAS_TORCH = False

from common.device import get_peak_memory_mb, reset_peak_memory_stats
from task2_vlm.models import SiglipQwenVLM, build_qwen_chat_prompt, IMAGE_PAD_TOKEN_ID, NUM_VISUAL_TOKENS
from task2_vlm.dataset import (
    VLMExample, QuestionPair, CurriculumBatchSampler, VLMCollator,
    QUESTION_TOKEN_CAP, CAPTION_TOKEN_CAP, ANSWER_TOKEN_CAP,
    create_synthetic_task2_datasets,
)

logger = logging.getLogger(__name__)


class CosineWarmupTo10PercentScheduler:
    """Per-stage LR scheduler: 5% linear warmup, then cosine decay to 10% of peak LR."""
    
    def __init__(
        self,
        optimizer: Any,
        peak_lrs: List[float],
        total_steps: int,
        warmup_ratio: float = 0.05,
        min_lr_ratio: float = 0.10,
    ):
        self.optimizer = optimizer
        self.peak_lrs = peak_lrs
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(1, int(self.total_steps * warmup_ratio))
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0

    def step(self, step: Optional[int] = None) -> List[float]:
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        current_lrs = []
        for i, (param_group, peak_lr) in enumerate(zip(self.optimizer.param_groups, self.peak_lrs)):
            min_lr = peak_lr * self.min_lr_ratio
            if self.current_step < self.warmup_steps:
                # Linear warmup from 0 to peak_lr
                lr = peak_lr * (self.current_step / self.warmup_steps)
            else:
                # Cosine decay from peak_lr to min_lr (10% of peak)
                decay_steps = self.total_steps - self.warmup_steps
                progress = min(1.0, (self.current_step - self.warmup_steps) / max(1, decay_steps))
                lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
                
            param_group["lr"] = lr
            current_lrs.append(lr)
            
        return current_lrs

    def get_last_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "peak_lrs": self.peak_lrs,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.current_step = state["current_step"]
        self.peak_lrs = state["peak_lrs"]
        self.total_steps = state["total_steps"]
        self.warmup_steps = state["warmup_steps"]


class VLMCurriculumTrainer:
    """Trainer orchestrating 3-stage curriculum training and evaluation."""
    
    def __init__(
        self,
        model: SiglipQwenVLM,
        sampler: CurriculumBatchSampler,
        collator: VLMCollator,
        output_dir: Union[str, Path] = "checkpoints/task2_vlm",
        dev_captions: Optional[List[VLMExample]] = None,
        dev_grounding: Optional[List[QuestionPair]] = None,
        dev_vqav2: Optional[List[QuestionPair]] = None,
        device: Optional[str] = None,
        use_amp: bool = True,
    ):
        self.model = model
        self.sampler = sampler
        self.collator = collator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dev_captions = dev_captions or []
        self.dev_grounding = dev_grounding or []
        self.dev_vqav2 = dev_vqav2 or []
        
        self.device = device or ("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
        self.use_amp = use_amp and (self.device == "cuda") and HAS_TORCH
        
        # Training budget: Stage 1 = 2250, Stage 2 = 1500, Stage 3 = 750 (Total 4,500)
        self.stage_budgets = {1: 2250, 2: 1500, 3: 750}
        self.total_updates = sum(self.stage_budgets.values())  # 4500
        
        self.global_update = 0
        self.stage = 1
        self.stage_step = 0
        
        self.scaler = GradScaler() if self.use_amp else None
        self.optimizer = None
        self.scheduler = None
        
        # Metrics logs
        self.train_logs: List[Dict[str, Any]] = []
        self.dev_logs: List[Dict[str, Any]] = []
        self.mixture_counts: Dict[str, int] = {
            "caption": 0, "presence": 0, "counting": 0, "vqav2": 0
        }

    def _setup_stage_optimizer(self, stage: int) -> None:
        """Configure AdamW and Cosine Scheduler for the given stage."""
        if not HAS_TORCH:
            return
            
        stage_updates = self.stage_budgets[stage]
        
        if stage == 1:
            # Stage 1: Connector only, peak LR 10^-3
            peak_lrs = [1e-3]
            param_groups = [
                {"params": self.model.connector.parameters(), "lr": 1e-3, "weight_decay": 0.01}
            ]
        elif stage == 2:
            # Stage 2: Connector (10^-4) + LoRA (2 * 10^-5)
            self.model.enable_lora()
            lora_params = [p for p in self.model.llm.parameters() if p.requires_grad]
            peak_lrs = [1e-4, 2e-5]
            param_groups = [
                {"params": self.model.connector.parameters(), "lr": 1e-4, "weight_decay": 0.01},
                {"params": lora_params, "lr": 2e-5, "weight_decay": 0.01},
            ]
        elif stage == 3:
            # Stage 3: Connector (5 * 10^-5) + LoRA (10^-5)
            lora_params = [p for p in self.model.llm.parameters() if p.requires_grad]
            peak_lrs = [5e-5, 1e-5]
            param_groups = [
                {"params": self.model.connector.parameters(), "lr": 5e-5, "weight_decay": 0.01},
                {"params": lora_params, "lr": 1e-5, "weight_decay": 0.01},
            ]
        else:
            raise ValueError(f"Unknown stage: {stage}")
            
        self.optimizer = AdamW(param_groups, weight_decay=0.01)
        self.scheduler = CosineWarmupTo10PercentScheduler(
            optimizer=self.optimizer,
            peak_lrs=peak_lrs,
            total_steps=stage_updates,
            warmup_ratio=0.05,
            min_lr_ratio=0.10,
        )
        logger.info(
            f"Configured optimizer for Stage {stage}: {len(param_groups)} groups, peak LRs: {peak_lrs}"
        )

    def train_step(self, effective_batch_examples: List[VLMExample]) -> float:
        """Execute one training update with microbatch 8 and accumulation 2."""
        if not HAS_TORCH or self.optimizer is None:
            return 0.0
            
        self.model.train()
        self.optimizer.zero_grad()
        
        # Microbatch 8 x 2 = 16 effective batch
        microbatch_1 = effective_batch_examples[:8]
        microbatch_2 = effective_batch_examples[8:16]
        
        accum_loss = 0.0
        
        for mb_idx, mb_examples in enumerate([microbatch_1, microbatch_2]):
            batch = self.collator(mb_examples, include_images=True)
            
            # Record mixture counts
            for ex in mb_examples:
                if ex.example_type == "caption":
                    self.mixture_counts["caption"] += 1
                elif ex.example_type == "presence":
                    self.mixture_counts["presence"] += 1
                elif ex.example_type == "counting":
                    self.mixture_counts["counting"] += 1
                else:
                    self.mixture_counts["vqav2"] += 1
                    
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        visual_features=batch["visual_features"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs["loss"] / 2.0  # Normalized across 2 accumulation steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    visual_features=batch["visual_features"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs["loss"] / 2.0
                loss.backward()
                
            accum_loss += float(loss.item() * 2.0)
            
        # Unscale before clipping
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
            
        # Gradient clipping: global norm 1.0
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        
        # Optimizer step
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
        self.scheduler.step()
        return accum_loss / 2.0

    def evaluate_dev(self) -> Dict[str, Any]:
        """Run development evaluation tracking caption loss and VQA answer NLL."""
        logger.info(f"Running Dev Evaluation at global update {self.global_update}...")
        self.model.eval()
        
        dev_results: Dict[str, Any] = {
            "global_update": self.global_update,
            "stage": self.stage,
            "caption_loss": None,
            "grounding": {},
            "vqav2": {},
        }
        
        if not HAS_TORCH:
            return dev_results
            
        with torch.no_grad():
            # 1. Dev Caption Loss (96 images)
            if self.dev_captions:
                caption_losses = []
                for i in range(0, len(self.dev_captions), 8):
                    mb = self.dev_captions[i : i + 8]
                    batch = self.collator(mb, include_images=True)
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        visual_features=batch["visual_features"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    if outputs["loss"] is not None:
                        caption_losses.append(outputs["loss"].item())
                if caption_losses:
                    dev_results["caption_loss"] = float(np.mean(caption_losses))
                    
            # 2. VQA Suites: Grounding and Natural VQAv2 under Correct, Swap, and No-image
            for suite_name, pair_list in [("grounding", self.dev_grounding), ("vqav2", self.dev_vqav2)]:
                if not pair_list:
                    continue
                    
                nll_correct = []
                nll_partner = []
                nll_noimage = []
                
                for pair in pair_list:
                    # Condition 1: Correct (Original image with original question)
                    nll_a_corr = self._compute_single_answer_nll(pair.example_a, image_cond="correct")
                    nll_b_corr = self._compute_single_answer_nll(pair.example_b, image_cond="correct")
                    nll_correct.extend([nll_a_corr, nll_b_corr])
                    
                    # Condition 2: Complementary Swap (Image B with Question A, Image A with Question B)
                    # Note: Original reference used for likelihood!
                    nll_a_swap = self._compute_single_answer_nll(pair.example_a, image_cond="swap", partner_image_id=pair.example_b.image_id)
                    nll_b_swap = self._compute_single_answer_nll(pair.example_b, image_cond="swap", partner_image_id=pair.example_a.image_id)
                    nll_partner.extend([nll_a_swap, nll_b_swap])
                    
                    # Condition 3: No-Image
                    nll_a_noimg = self._compute_single_answer_nll(pair.example_a, image_cond="no_image")
                    nll_b_noimg = self._compute_single_answer_nll(pair.example_b, image_cond="no_image")
                    nll_noimage.extend([nll_a_noimg, nll_b_noimg])
                    
                mean_corr = float(np.mean(nll_correct))
                mean_swap = float(np.mean(nll_partner))
                mean_noimg = float(np.mean(nll_noimage))
                delta_nll = mean_swap - mean_corr  # Image-Swap Penalty
                
                dev_results[suite_name] = {
                    "nll_correct": mean_corr,
                    "nll_swap": mean_swap,
                    "nll_no_image": mean_noimg,
                    "delta_nll": delta_nll,
                }
                logger.info(
                    f"[{suite_name.upper()}] Correct NLL: {mean_corr:.4f}, Swap NLL: {mean_swap:.4f}, "
                    f"Delta NLL: {delta_nll:.4f}, No-Image NLL: {mean_noimg:.4f}"
                )
                
        self.dev_logs.append(dev_results)
        return dev_results

    def _compute_single_answer_nll(
        self,
        example: VLMExample,
        image_cond: str = "correct",
        partner_image_id: Optional[Union[str, int]] = None,
    ) -> float:
        """Compute teacher-forced answer NLL excluding termination token."""
        include_image = (image_cond != "no_image")
        
        # Determine image features
        vis_features = None
        if include_image and self.collator.feature_cache is not None:
            img_id = partner_image_id if (image_cond == "swap" and partner_image_id is not None) else example.image_id
            vis_features = self.collator.feature_cache.load(img_id, as_torch=True).unsqueeze(0).to(self.device)
            
        prompt_str, full_str = build_qwen_chat_prompt(
            question=example.question,
            answer=example.target,
            num_image_tokens=self.collator.num_visual_tokens if include_image else 0,
            include_image=include_image,
        )
        
        prompt_ids = self.model.tokenizer.encode(prompt_str, add_special_tokens=False)
        full_ids = self.model.tokenizer.encode(full_str, add_special_tokens=False)
        
        # Exclude termination token (<|im_end|>) from answer NLL
        eos_id = self.model.tokenizer.eos_token_id
        answer_ids = full_ids[len(prompt_ids):]
        if answer_ids and answer_ids[-1] == eos_id:
            answer_ids = answer_ids[:-1]
            
        if not answer_ids:
            return 0.0
            
        target_seq = full_ids[:len(prompt_ids) + len(answer_ids)]
        input_tensor = torch.tensor([target_seq], dtype=torch.long, device=self.device)
        attn_mask = torch.ones_like(input_tensor)
        
        outputs = self.model(
            input_ids=input_tensor,
            visual_features=vis_features,
            attention_mask=attn_mask,
        )
        logits = outputs["logits"]  # (1, L, V)
        
        # Shift logits to predict target tokens
        shift_logits = logits[0, :-1, :]
        shift_labels = input_tensor[0, 1:]
        
        # Slice answer token positions: len(prompt_ids) - 1 to end
        ans_start_idx = len(prompt_ids) - 1
        ans_logits = shift_logits[ans_start_idx:]
        ans_labels = shift_labels[ans_start_idx:]
        
        token_nlls = F.cross_entropy(ans_logits.float(), ans_labels, reduction="none")
        return float(token_nlls.mean().item())

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save full resumable training checkpoint."""
        if not HAS_TORCH:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "global_update": self.global_update,
            "stage": self.stage,
            "stage_step": self.stage_step,
            "connector_state": self.model.connector.state_dict(),
            "connector_scale": float(self.model.connector.scale.item() if hasattr(self.model.connector.scale, "item") else self.model.connector.scale),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "sampler_state": self.sampler.state_dict(),
            "mixture_counts": self.mixture_counts,
            "rng_state": torch.get_rng_state(),
        }
        if self.model.lora_enabled and self.model.llm is not None:
            checkpoint["lora_state"] = {k: v for k, v in self.model.llm.state_dict().items() if "lora" in k.lower()}
            
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Resume from training checkpoint."""
        if not HAS_TORCH:
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.global_update = checkpoint["global_update"]
        self.stage = checkpoint["stage"]
        self.stage_step = checkpoint["stage_step"]
        
        self.model.connector.load_state_dict(checkpoint["connector_state"])
        if "connector_scale" in checkpoint:
            self.model.connector.scale.copy_(torch.tensor(checkpoint["connector_scale"], dtype=torch.float32))
            
        self._setup_stage_optimizer(self.stage)
        if self.optimizer and checkpoint.get("optimizer_state"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler and checkpoint.get("scheduler_state"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if self.scaler and checkpoint.get("scaler_state"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        if checkpoint.get("sampler_state"):
            self.sampler.load_state_dict(checkpoint["sampler_state"])
        if checkpoint.get("mixture_counts"):
            self.mixture_counts = checkpoint["mixture_counts"]
        if checkpoint.get("rng_state") is not None:
            torch.set_rng_state(checkpoint["rng_state"])
            
        if self.stage >= 2 and "lora_state" in checkpoint:
            self.model.enable_lora()
            self.model.llm.load_state_dict(checkpoint["lora_state"], strict=False)
            
        logger.info(f"Resumed from update {self.global_update} in Stage {self.stage}")

    def run_training(self) -> None:
        """Execute complete curriculum training through Stages 1, 2, and 3."""
        start_time = time.time()
        logger.info(f"Starting curriculum training ({self.total_updates} total updates)...")
        
        if not HAS_TORCH:
            logger.info("Torch is not installed on this machine; completing training in stub mode.")
            summary = {
                "total_updates": self.total_updates,
                "training_time_sec": 0.0,
                "peak_memory_mb": 0.0,
                "mixture_counts": self.mixture_counts,
                "train_logs": [],
                "dev_logs": [],
                "stub_mode": True,
            }
            with open(self.output_dir / "training_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            return

        # Initial dev evaluation (Update 0)
        if self.global_update == 0:
            self.evaluate_dev()
            
        for stage in [1, 2, 3]:
            if self.stage > stage:
                continue
                
            self.stage = stage
            self._setup_stage_optimizer(stage)
            stage_budget = self.stage_budgets[stage]
            logger.info(f"=== Beginning Stage {stage}: {stage_budget} updates ===")
            
            while self.stage_step < stage_budget:
                self.stage_step += 1
                self.global_update += 1
                
                # Draw effective batch (16 examples)
                effective_batch = self.sampler.get_effective_batch(stage)
                loss = self.train_step(effective_batch)
                
                if self.global_update % 50 == 0:
                    lr_str = f"{self.scheduler.get_last_lr()[0]:.2e}" if self.scheduler else "N/A"
                    logger.info(
                        f"Stage {stage} [{self.stage_step}/{stage_budget}] | Update {self.global_update} | "
                        f"Loss: {loss:.4f} | LR: {lr_str} | Mem: {get_peak_memory_mb():.1f}MB"
                    )
                    self.train_logs.append({
                        "global_update": self.global_update,
                        "stage": stage,
                        "stage_step": self.stage_step,
                        "loss": loss,
                        "lrs": self.scheduler.get_last_lr() if self.scheduler else [],
                    })
                    
                # Save resumable checkpoint every 250 main-run updates
                if self.global_update % 250 == 0:
                    ckpt_path = self.output_dir / f"checkpoint_step_{self.global_update}.pt"
                    self.save_checkpoint(ckpt_path)
                    
                # Dev evaluation every 500 updates
                if self.global_update % 500 == 0:
                    self.evaluate_dev()
                    
            # Stage endpoint: save weights and evaluate
            logger.info(f"--- Completed Stage {stage} at global update {self.global_update} ---")
            self.model.save_connector(self.output_dir / f"connector_stage_{stage}.pt")
            if stage >= 2:
                self.model.save_adapters(self.output_dir / f"lora_adapters_stage_{stage}")
                
            self.evaluate_dev()
            self.stage_step = 0
            
        total_time = time.time() - start_time
        logger.info(f"Training completed successfully in {total_time:.2f} seconds.")
        
        # Save training summary
        summary = {
            "total_updates": self.global_update,
            "training_time_sec": total_time,
            "peak_memory_mb": get_peak_memory_mb(),
            "mixture_counts": self.mixture_counts,
            "train_logs": self.train_logs,
            "dev_logs": self.dev_logs,
        }
        with open(self.output_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def run_curriculum_training(cfg: Optional[Dict[str, Any]] = None) -> None:
    """CLI runner function for 3-stage curriculum training compatible with scripts/run_task2.py."""
    cfg = cfg or {}
    model_cfg = cfg.get("models", {})
    output_dir = cfg.get("evaluation", {}).get("output_dir", "./outputs/task2")
    device = cfg.get("device", "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
    
    logger.info("Initializing synthetic/real datasets for curriculum training...")
    datasets = create_synthetic_task2_datasets(seed=model_cfg.get("seed", 6304))
    
    sampler = CurriculumBatchSampler(
        caption_pool=datasets["train_captions"],
        presence_pool=datasets["train_presence"],
        counting_pool=datasets["train_counting"],
        vqav2_pool=datasets["train_vqav2"],
        seed=model_cfg.get("seed", 6304),
    )
    
    model = SiglipQwenVLM(
        llm_model_name=model_cfg.get("language_model", "Qwen/Qwen3-0.6B"),
        device=device,
        load_pretrained=HAS_TORCH and HAS_TRANSFORMERS,
    )
    
    collator = VLMCollator(tokenizer=model.tokenizer, device=device)
    
    trainer = VLMCurriculumTrainer(
        model=model,
        sampler=sampler,
        collator=collator,
        output_dir=output_dir,
        dev_captions=datasets["dev_captions"],
        dev_grounding=datasets["dev_grounding"],
        dev_vqav2=datasets["dev_vqav2"],
        device=device,
    )
    
    trainer.run_training()

