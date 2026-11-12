"""Vision-Language Model architecture for SigLIP + Qwen3-0.6B.

Components:
- Vision Encoder: google/siglip-base-patch16-224 (frozen, features pre-cached).
- Connector: 2-layer MLP (Linear 768 -> 1024, GELU, Linear 1024 -> 1024). Trainable throughout.
- Connector calibration: Scale connector output magnitude against text embeddings.
- Language Model: Qwen/Qwen3-0.6B with enable_thinking=False. Original weights frozen,
  autograd flows from Qwen into the connector. LoRA enabled in Stages 2 and 3.
- Visual tokens inserted into user message before question, preserving Qwen's chat structure.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
import logging
import math
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

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoConfig = None
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    HAS_PEFT = True
except ImportError:
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    PeftModel = None
    HAS_PEFT = False

logger = logging.getLogger(__name__)

# Constants for SigLIP and Qwen
VISUAL_DIM = 768        # SigLIP hidden dimension
LLM_DIM = 1024          # Qwen3-0.6B hidden dimension
NUM_VISUAL_TOKENS = 196 # 14x14 patches
COMPRESSED_TOKENS = 49  # 7x7 pooled patches
IMAGE_PAD_TOKEN_ID = 151646  # Placeholder token id for image patches


class MultimodalConnector(nn.Module if HAS_TORCH else object):
    """2-layer MLP connector projecting SigLIP patch features into Qwen embedding space.
    
    Architecture:
    Linear(768, 1024) -> GELU() -> Linear(1024, 1024)
    
    Calibration:
    Applies a constant scalar multiplier `scale` determined by calibrating connector
    output L2 norms against text embedding L2 norms using training data only.
    """
    
    def __init__(
        self,
        in_dim: int = VISUAL_DIM,
        out_dim: int = LLM_DIM,
        initial_scale: float = 1.0,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        if HAS_TORCH:
            super().__init__()
            self.linear1 = nn.Linear(in_dim, out_dim, bias=True)
            self.act = nn.GELU()
            self.linear2 = nn.Linear(out_dim, out_dim, bias=True)
            self.register_buffer("scale", torch.tensor(initial_scale, dtype=torch.float32))
        else:
            self.scale = float(initial_scale)
            self.training = False
            # Deterministic initialization for numpy MLP weights
            rng = np.random.default_rng(6304)
            limit1 = np.sqrt(6.0 / (in_dim + out_dim))
            self.w1 = rng.uniform(-limit1, limit1, (in_dim, out_dim)).astype(np.float32)
            self.b1 = np.zeros(out_dim, dtype=np.float32)
            limit2 = np.sqrt(6.0 / (out_dim + out_dim))
            self.w2 = rng.uniform(-limit2, limit2, (out_dim, out_dim)).astype(np.float32)
            self.b2 = np.zeros(out_dim, dtype=np.float32)
            
        self.calibration_info: Dict[str, Any] = {
            "rule": "ratio of mean L2 norm of Qwen text embeddings to mean L2 norm of unscaled connector visual embeddings over training samples",
            "scale": initial_scale,
            "calibrated": False,
        }

    def eval(self) -> "MultimodalConnector":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "MultimodalConnector":
        self.training = mode
        return self

    def parameters(self) -> List[Any]:
        if HAS_TORCH and isinstance(self, nn.Module):
            return super().parameters()
        return []

    def state_dict(self) -> Dict[str, Any]:
        if HAS_TORCH and isinstance(self, nn.Module):
            return super().state_dict()
        return {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "scale": self.scale,
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        if HAS_TORCH and isinstance(self, nn.Module):
            super().load_state_dict(state, strict=strict)
        else:
            if "w1" in state:
                self.w1 = np.asarray(state["w1"], dtype=np.float32)
            if "b1" in state:
                self.b1 = np.asarray(state["b1"], dtype=np.float32)
            if "w2" in state:
                self.w2 = np.asarray(state["w2"], dtype=np.float32)
            if "b2" in state:
                self.b2 = np.asarray(state["b2"], dtype=np.float32)
            if "scale" in state:
                self.scale = float(state["scale"])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def _forward_unscaled(self, x: Any) -> Any:
        """Compute unscaled 2-layer MLP projection Linear(in_dim, out_dim) -> GELU -> Linear(out_dim, out_dim)."""
        if HAS_TORCH and isinstance(x, torch.Tensor):
            return self.linear2(self.act(self.linear1(x)))
            
        arr = np.asarray(x, dtype=np.float32)
        if HAS_TORCH and hasattr(self, "linear1"):
            w1 = self.linear1.weight.detach().cpu().numpy().T
            b1 = self.linear1.bias.detach().cpu().numpy()
            w2 = self.linear2.weight.detach().cpu().numpy().T
            b2 = self.linear2.bias.detach().cpu().numpy()
        else:
            w1 = self.w1
            b1 = self.b1
            w2 = self.w2
            b2 = self.b2
            
        h1 = arr @ w1 + b1
        # GELU approximation
        h1 = 0.5 * h1 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h1 + 0.044715 * (h1 ** 3))))
        h2 = h1 @ w2 + b2
        return h2

    def forward(self, x: Any) -> Any:
        """Forward pass projecting visual tokens into language space.
        
        Args:
            x: Visual features of shape (..., in_dim).
            
        Returns:
            Projected tokens of shape (..., out_dim) scaled by calibration factor.
        """
        if HAS_TORCH and isinstance(x, torch.Tensor):
            h = self.linear1(x)
            h = self.act(h)
            h = self.linear2(h)
            return h * self.scale
            
        h_unscaled = self._forward_unscaled(x)
        scale_val = float(self.scale.item() if hasattr(self.scale, "item") else self.scale)
        return (h_unscaled * scale_val).astype(np.asarray(x).dtype)

    def calibrate(
        self,
        sample_visual_features: Union[np.ndarray, Any],
        text_embedding_matrix: Union[np.ndarray, Any],
    ) -> float:
        """Calibrate connector output magnitude against text embeddings using training data only.
        
        Rule:
        s = E_{w}[||e_text(w)||_2] / E_{x in D_train}[||MLP(x)||_2]
        
        Args:
            sample_visual_features: Sample batch of visual features from training data (N, 196, 768).
            text_embedding_matrix: Frozen LLM text embedding table (V, 1024).
            
        Returns:
            Computed calibration scale factor `s`.
        """
        was_training = self.training
        self.eval()
        
        if HAS_TORCH and isinstance(sample_visual_features, torch.Tensor) and isinstance(text_embedding_matrix, torch.Tensor):
            with torch.no_grad():
                vis_sample = sample_visual_features.float()
                if vis_sample.ndim == 2:
                    vis_sample = vis_sample.unsqueeze(0)
                    
                param_device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
                vis_sample = vis_sample.to(param_device)
                
                h_unscaled = self._forward_unscaled(vis_sample)
                vis_norm = torch.linalg.norm(h_unscaled, dim=-1).mean().item()
                
                text_embeds = text_embedding_matrix.float().to(param_device)
                text_norm = torch.linalg.norm(text_embeds, dim=-1).mean().item()
                
                scale_val = float(text_norm / vis_norm) if vis_norm > 1e-8 else 1.0
                self.scale.copy_(torch.tensor(scale_val, dtype=torch.float32, device=param_device))
        else:
            vis_sample = np.asarray(sample_visual_features, dtype=np.float32)
            if vis_sample.ndim == 2:
                vis_sample = np.expand_dims(vis_sample, 0)
                
            h_unscaled = self._forward_unscaled(vis_sample)
            vis_norm = float(np.mean(np.linalg.norm(h_unscaled, axis=-1)))
            
            text_embeds = np.asarray(text_embedding_matrix, dtype=np.float32)
            text_norm = float(np.mean(np.linalg.norm(text_embeds, axis=-1)))
            
            scale_val = float(text_norm / vis_norm) if vis_norm > 1e-8 else 1.0
            if HAS_TORCH and hasattr(self, "scale") and isinstance(self.scale, torch.Tensor):
                param_device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
                self.scale.copy_(torch.tensor(scale_val, dtype=torch.float32, device=param_device))
            else:
                self.scale = scale_val
                
        self.calibration_info = {
            "rule": "ratio of mean L2 norm of Qwen text embeddings to mean L2 norm of unscaled connector visual embeddings over training samples",
            "scale": scale_val,
            "mean_text_norm": text_norm,
            "mean_unscaled_vis_norm": vis_norm,
            "calibrated": True,
        }
        logger.info(
            f"Connector calibrated: text_norm={text_norm:.4f}, vis_norm={vis_norm:.4f}, scale={scale_val:.4f}"
        )
            
        if was_training:
            self.train()
            
        return scale_val


class SiglipQwenVLM(nn.Module if HAS_TORCH else object):
    """SigLIP-Qwen Vision-Language Model.
    
    Maintains frozen Qwen3-0.6B backbone, trainable MultimodalConnector,
    and optional PEFT LoRA adapters for Stages 2 and 3.
    Autograd flows smoothly from Qwen to the connector during training.
    """
    
    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen3-0.6B",
        visual_dim: int = VISUAL_DIM,
        llm_dim: int = LLM_DIM,
        torch_dtype: Optional[Any] = None,
        device: Optional[str] = None,
        load_pretrained: bool = True,
    ):
        if HAS_TORCH:
            super().__init__()
            
        self.llm_model_name = llm_model_name
        self.visual_dim = visual_dim
        self.llm_dim = llm_dim
        self.device = device or ("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if (HAS_TORCH and self.device == "cuda") else (torch.float32 if HAS_TORCH else None))
        
        # Trainable connector
        self.connector = MultimodalConnector(in_dim=visual_dim, out_dim=llm_dim)
        
        # Qwen language model and tokenizer
        self.llm = None
        self.tokenizer = None
        self.lora_enabled = False
        self.lora_config = None
        
        if load_pretrained and HAS_TORCH and HAS_TRANSFORMERS:
            self._load_llm()
            
        self.training = False

    def eval(self) -> "SiglipQwenVLM":
        self.training = False
        if hasattr(self, "connector"):
            self.connector.eval()
        return self

    def train(self, mode: bool = True) -> "SiglipQwenVLM":
        self.training = mode
        if hasattr(self, "connector"):
            self.connector.train(mode)
        return self

    def parameters(self) -> List[Any]:
        if HAS_TORCH and isinstance(self, nn.Module):
            return super().parameters()
        if hasattr(self, "connector"):
            return self.connector.parameters()
        return []

    def to(self, device: str) -> "SiglipQwenVLM":
        self.device = device
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


    def _load_llm(self) -> None:
        """Load Qwen model, freeze original weights, and configure gradient checkpointing."""
        logger.info(f"Loading LLM {self.llm_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        config = AutoConfig.from_pretrained(self.llm_model_name, trust_remote_code=True)
        # Ensure thinking mode is disabled as mandated by assignment
        if hasattr(config, "enable_thinking"):
            config.enable_thinking = False
            
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            config=config,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )
        self.llm.to(self.device)
        
        # Explicitly freeze all original LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # Enable gradient checkpointing if supported, ensuring inputs require grads
        try:
            self.llm.gradient_checkpointing_enable()
            if hasattr(self.llm, "enable_input_require_grads"):
                self.llm.enable_input_require_grads()
        except Exception as e:
            logger.debug(f"Gradient checkpointing configuration notice: {e}")

    def enable_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """Enable LoRA on Qwen attention projections for Stages 2 and 3.
        
        Config:
        - Rank: 16
        - Alpha: 32
        - Dropout: 0.05
        - Bias: 'none' (no trainable biases)
        - Target modules: q_proj, k_proj, v_proj, o_proj
        """
        if self.lora_enabled:
            logger.info("LoRA already enabled.")
            return
            
        target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        if not HAS_PEFT or self.llm is None:
            logger.warning("PEFT or LLM not available; marking LoRA enabled in stub mode.")
            self.lora_enabled = True
            return
            
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM if TaskType is not None else None,
        )
        
        self.llm = get_peft_model(self.llm, self.lora_config)
        self.lora_enabled = True
        
        # Re-ensure inputs require grads for gradient checkpointing with LoRA
        if hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()
            
        # Verify trainable parameter state
        trainable_params, all_params = self.get_trainable_parameters()
        logger.info(
            f"LoRA enabled: {trainable_params:,} trainable / {all_params:,} total parameters."
        )

    def get_trainable_parameters(self) -> Tuple[int, int]:
        """Count trainable vs total parameters across connector and language model."""
        if not HAS_TORCH:
            return (0, 0)
            
        trainable = 0
        total = 0
        for p in self.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        return trainable, total

    def embed_multimodal_inputs(
        self,
        input_ids: Any,
        visual_features: Optional[Any] = None,
        image_pad_token_id: int = IMAGE_PAD_TOKEN_ID,
    ) -> Any:
        """Construct multimodal inputs_embeds by replacing image pad tokens with connector outputs.
        
        Preserves autograd connection between inputs_embeds and connector parameters.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len).
            visual_features: Optional visual features of shape (batch_size, num_visual_tokens, 768).
            image_pad_token_id: Token ID marking visual token positions.
            
        Returns:
            inputs_embeds: Spliced embeddings of shape (batch_size, seq_len, 1024).
        """
        if not HAS_TORCH or self.llm is None:
            return None
            
        # Get base text embeddings (frozen)
        embed_tokens = self.llm.get_input_embeddings()
        text_embeds = embed_tokens(input_ids)  # Shape (B, L, 1024)
        
        if visual_features is None:
            # Condition: No image (or text-only pass)
            return text_embeds
            
        # Project visual tokens through trainable connector (autograd graph root)
        vis_embeds = self.connector(visual_features)  # Shape (B, N_vis, 1024)
        vis_embeds = vis_embeds.to(dtype=text_embeds.dtype)
        
        # Clone text embeds so in-place slice replacement preserves autograd
        inputs_embeds = text_embeds.clone()
        
        batch_size, seq_len, _ = inputs_embeds.shape
        num_vis = vis_embeds.shape[1]
        
        for b in range(batch_size):
            # Locate image placeholder positions
            is_image = (input_ids[b] == image_pad_token_id)
            indices = torch.where(is_image)[0]
            if len(indices) > 0:
                count = min(len(indices), num_vis)
                inputs_embeds[b, indices[:count], :] = vis_embeds[b, :count, :]
                
        return inputs_embeds

    def forward(
        self,
        input_ids: Any,
        visual_features: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        position_ids: Optional[Any] = None,
        labels: Optional[Any] = None,
    ) -> Any:
        """Forward pass through connector and Qwen language model.
        
        Calculates logits and example-weighted causal cross-entropy loss if labels provided.
        """
        if not HAS_TORCH or self.llm is None:
            # Stub return for syntax/type testing
            return {"loss": None, "logits": None}
            
        inputs_embeds = self.embed_multimodal_inputs(input_ids, visual_features)
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = outputs.logits  # Shape (B, L, V)
        
        loss = None
        if labels is not None:
            loss = self.compute_example_weighted_loss(logits, labels)
            
        return {"loss": loss, "logits": logits}

    @staticmethod
    def compute_example_weighted_loss(
        logits: Any,
        labels: Any,
        ignore_index: int = -100,
    ) -> Any:
        """Compute the assignment's example-weighted causal cross-entropy loss:
        
        L = - (1 / |B|) sum_{i in B} (1 / |R_i|) sum_{t in R_i} log p(y_{i,t} | I_i, q_i, y_{i,<t}).
        
        Each example's loss is averaged over its own supervised tokens |R_i| first,
        then averaged equally across the batch |B|. Visual tokens, prompts, and padding
        are ignored (-100).
        """
        if not HAS_TORCH:
            return 0.0
            
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        batch_size = shift_labels.shape[0]
        example_losses = []
        
        # Token-level cross entropy without reduction: (B, L-1)
        # Using float32 for stable log-softmax
        loss_flat = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]).float(),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=ignore_index,
        ).view(batch_size, -1)
        
        for b in range(batch_size):
            valid_mask = (shift_labels[b] != ignore_index)
            r_i = valid_mask.sum().float()
            if r_i > 0:
                example_loss = (loss_flat[b] * valid_mask.float()).sum() / r_i
                example_losses.append(example_loss)
            else:
                example_losses.append(torch.tensor(0.0, device=logits.device, requires_grad=True))
                
        if example_losses:
            return torch.stack(example_losses).mean()
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    def generate(
        self,
        input_ids: Any,
        visual_features: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        temperature: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Any:
        """Autoregressive greedy generation with 16-token cap and thinking disabled.
        
        Args:
            input_ids: Prompt token IDs (B, L).
            visual_features: Pre-cached visual tokens (B, N, 768) or None.
            max_new_tokens: Hard cap of 16 tokens.
            do_sample: Greedy decoding (False).
        """
        if not HAS_TORCH or self.llm is None:
            return input_ids
            
        inputs_embeds = self.embed_multimodal_inputs(input_ids, visual_features)
        
        pad_token_id = pad_token_id or (self.tokenizer.pad_token_id if self.tokenizer else 0)
        eos_token_id = eos_token_id or (self.tokenizer.eos_token_id if self.tokenizer else 2)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        return outputs

    def save_connector(self, path: Union[str, os.PathLike]) -> None:
        """Save connector state dict and calibration info."""
        if not HAS_TORCH:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "state_dict": self.connector.state_dict(),
            "calibration_info": self.connector.calibration_info,
            "scale": float(self.connector.scale.item() if hasattr(self.connector.scale, "item") else self.connector.scale),
        }, path)
        logger.info(f"Saved connector weights to {path}")

    def load_connector(self, path: Union[str, os.PathLike]) -> None:
        """Load connector state dict and calibration factor."""
        if not HAS_TORCH:
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.connector.load_state_dict(checkpoint["state_dict"])
        if "scale" in checkpoint:
            self.connector.scale.copy_(torch.tensor(checkpoint["scale"], dtype=torch.float32))
        if "calibration_info" in checkpoint:
            self.connector.calibration_info = checkpoint["calibration_info"]
        logger.info(f"Loaded connector weights from {path}")

    def save_adapters(self, save_directory: Union[str, os.PathLike]) -> None:
        """Save LoRA adapter weights after Stage 2 or Stage 3."""
        if not self.lora_enabled or self.llm is None:
            logger.warning("LoRA not enabled; skipping adapter save.")
            return
        os.makedirs(save_directory, exist_ok=True)
        if hasattr(self.llm, "save_pretrained"):
            self.llm.save_pretrained(save_directory)
            logger.info(f"Saved LoRA adapters to {save_directory}")

    def load_adapters(self, adapter_path: Union[str, os.PathLike]) -> None:
        """Load trained LoRA adapters."""
        if not HAS_PEFT or self.llm is None:
            return
        logger.info(f"Loading LoRA adapters from {adapter_path}")
        self.llm = PeftModel.from_pretrained(self.llm, adapter_path)
        self.lora_enabled = True


def build_qwen_chat_prompt(
    question: str,
    answer: Optional[str] = None,
    num_image_tokens: int = NUM_VISUAL_TOKENS,
    include_image: bool = True,
    image_token: str = "<|image_pad|>",
) -> Tuple[str, Optional[str]]:
    """Format prompt conforming to Qwen's chat template with visual token insertion.
    
    Structure:
    <|im_start|>user
    <image_tokens>
    {question}<|im_end|>
    <|im_start|>assistant
    {answer}<|im_end|>
    
    Args:
        question: The user query.
        answer: Target assistant response (optional).
        num_image_tokens: Number of visual tokens (196 or 49).
        include_image: True for Correct/Swap/Mismatched conditions, False for No-Image.
        image_token: String representation of image placeholder token.
        
    Returns:
        (prompt_text, full_text_with_answer_if_provided)
    """
    if include_image and num_image_tokens > 0:
        image_prefix = (image_token * num_image_tokens) + "\n"
    else:
        image_prefix = ""
        
    user_content = f"{image_prefix}{question}"
    prompt_str = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    
    if answer is not None:
        full_str = f"{prompt_str}{answer}<|im_end|>"
        return prompt_str, full_str
    return prompt_str, None
