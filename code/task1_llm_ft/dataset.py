"""Dataset loading, filtering, formatting, and batching for Task 1: LLM Fine-Tuning.

Implements:
1. MetaMathQA filtering:
   - Exclude empty query or response
   - Exclude formatted token length > 2,048 tokens (do not truncate)
   - Exclude contamination: match against normalized GSM8K test questions
   - First 40,000 eligible examples in source order (no shuffle)
2. Prompt and response formatting for Qwen3 (enable_thinking=False)
   - Mask prompt tokens and padding with -100
   - Compute loss only on assistant response tokens and end-of-turn token
3. Effective batch token loss normalization support:
   - Microbatch 2, gradient accumulation 16, effective batch 32
   - Normalization over all valid response tokens in the effective batch
"""

from typing import Dict, List, Optional, Sequence, Set, Tuple, Any, Union
import os
import json
import logging

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, Sampler
    HAS_TORCH = True
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None
    Sampler = object
    HAS_TORCH = False

from common.normalization import normalize_question_for_contamination

logger = logging.getLogger(__name__)


def build_gsm8k_decontamination_set(
    gsm8k_questions: Optional[Sequence[str]] = None,
    gsm8k_dataset_path: Optional[str] = None,
) -> Set[str]:
    """Build a set of normalized GSM8K test questions for contamination filtering.
    
    Args:
        gsm8k_questions: Optional explicit list of GSM8K test question strings.
        gsm8k_dataset_path: Optional path to a local json/jsonl file or dataset directory.
        
    Returns:
        Set of normalized question strings.
    """
    normalized_set: Set[str] = set()
    
    if gsm8k_questions is not None:
        for q in gsm8k_questions:
            norm_q = normalize_question_for_contamination(q)
            if norm_q:
                normalized_set.add(norm_q)
        return normalized_set
        
    if gsm8k_dataset_path and os.path.exists(gsm8k_dataset_path):
        try:
            with open(gsm8k_dataset_path, "r", encoding="utf-8") as f:
                if gsm8k_dataset_path.endswith(".jsonl"):
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            q = item.get("question", "")
                            norm_q = normalize_question_for_contamination(q)
                            if norm_q:
                                normalized_set.add(norm_q)
                else:
                    data = json.load(f)
                    for item in data:
                        q = item.get("question", "") if isinstance(item, dict) else str(item)
                        norm_q = normalize_question_for_contamination(q)
                        if norm_q:
                            normalized_set.add(norm_q)
            logger.info(f"Loaded {len(normalized_set)} GSM8K test questions from {gsm8k_dataset_path}")
            return normalized_set
        except Exception as e:
            logger.warning(f"Failed to load GSM8K dataset from {gsm8k_dataset_path}: {e}")
            
    # Try loading from datasets library if available
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for item in ds:
            q = item.get("question", "")
            norm_q = normalize_question_for_contamination(q)
            if norm_q:
                normalized_set.add(norm_q)
        logger.info(f"Loaded {len(normalized_set)} GSM8K test questions from HuggingFace datasets")
    except Exception as e:
        logger.warning(f"Could not load GSM8K test split via datasets: {e}. Contamination set may be empty.")
        
    return normalized_set


def _as_id_list(x: Any) -> List[int]:
    """Normalize apply_chat_template output: list of ids, BatchEncoding/dict, or batched list."""
    if hasattr(x, "keys") and "input_ids" in x:
        x = x["input_ids"]
    if hasattr(x, "tolist"):
        x = x.tolist()
    if len(x) > 0 and isinstance(x[0], (list, tuple)):
        x = x[0]
    return [int(t) for t in x]


def _end_of_turn_id(tokenizer: Any) -> Optional[int]:
    fn = getattr(tokenizer, "convert_tokens_to_ids", None)
    if fn is None:
        return None
    try:
        tid = fn("<|im_end|>")
    except Exception:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    return int(tid) if isinstance(tid, int) and tid >= 0 and tid != unk else None


def format_qwen_chat(
    tokenizer: Any,
    query: str,
    response: str,
    enable_thinking: bool = False,
) -> Tuple[List[int], List[int]]:
    """Format user query and assistant response into input_ids and labels.
    
    Enforces enable_thinking=False strictly. Prompt tokens are masked with -100.
    Response tokens and the end-of-turn token are retained for next-token prediction loss.
    
    Returns:
        (input_ids, labels)
    """
    messages = [
        {"role": "user", "content": query.strip()},
        {"role": "assistant", "content": response.strip()},
    ]
    
    prompt_messages = [
        {"role": "user", "content": query.strip()},
    ]
    
    # Check if tokenizer has apply_chat_template
    if hasattr(tokenizer, "apply_chat_template"):
        # Qwen chat template kwargs
        template_kwargs: Dict[str, Any] = {"tokenize": True}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
            
        try:
            # Tokenize prompt only (with generation prompt added)
            prompt_kwargs = dict(template_kwargs)
            prompt_kwargs["add_generation_prompt"] = True
            prompt_ids = tokenizer.apply_chat_template(prompt_messages, **prompt_kwargs)
            
            # Tokenize full conversation
            full_kwargs = dict(template_kwargs)
            full_kwargs["add_generation_prompt"] = False
            full_ids = tokenizer.apply_chat_template(messages, **full_kwargs)
        except TypeError:
            # Fallback if tokenizer.apply_chat_template doesn't accept enable_thinking
            template_kwargs.pop("enable_thinking", None)
            prompt_ids = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
            full_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    else:
        # Standard Qwen chat template string format fallback
        # <|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n
        prompt_text = f"<|im_start|>user\n{query.strip()}<|im_end|>\n<|im_start|>assistant\n"
        full_text = f"<|im_start|>user\n{query.strip()}<|im_end|>\n<|im_start|>assistant\n{response.strip()}<|im_end|>\n"
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False) if tokenizer else [1] * len(prompt_text.split())
        full_ids = tokenizer.encode(full_text, add_special_tokens=False) if tokenizer else [1] * len(full_text.split())

    prompt_ids = _as_id_list(prompt_ids)
    full_ids = _as_id_list(full_ids)
    prompt_len = len(prompt_ids)
    input_ids = list(full_ids)

    # Mask prompt tokens with -100; keep response tokens and the end-of-turn token
    labels = [-100] * prompt_len + list(full_ids[prompt_len:])

    # The chat template appends a newline after <|im_end|>: it is not part of the supervised target
    end_id = _end_of_turn_id(tokenizer)
    if end_id is not None:
        ends = [i for i in range(prompt_len, len(full_ids)) if full_ids[i] == end_id]
        if ends:
            for j in range(ends[-1] + 1, len(labels)):
                labels[j] = -100
    
    # Safety check: ensure length matches
    if len(labels) != len(input_ids):
        # Truncate or pad labels to match input_ids
        if len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]
        else:
            labels = labels + [-100] * (len(input_ids) - len(labels))
            
    return input_ids, labels


def filter_metamath_dataset(
    raw_dataset: Sequence[Dict[str, Any]],
    tokenizer: Any,
    gsm8k_decontamination_set: Optional[Set[str]] = None,
    target_count: int = 40000,
    max_seq_length: int = 2048,
    enable_thinking: bool = False,
) -> List[Dict[str, Any]]:
    """Filter MetaMathQA dataset to the first 40,000 eligible examples in source order.
    
    Filters:
    1. Empty query or empty response.
    2. GSM8K contamination match on normalized query or original_question.
    3. Formatted token length > 2,048 tokens.
    
    Does NOT truncate sequences; does NOT shuffle examples.
    """
    eligible_examples: List[Dict[str, Any]] = []
    decontam_set = gsm8k_decontamination_set or set()
    
    filtered_empty = 0
    filtered_contam = 0
    filtered_length = 0
    
    for idx, item in enumerate(raw_dataset):
        if not isinstance(item, dict):
            filtered_empty += 1
            continue
        query = str(item.get("query") or item.get("question") or "").strip()
        response = str(item.get("response") or item.get("answer") or "").strip()
        original_question = str(item.get("original_question") or "").strip()
        
        # 1. Exclude empty question or response
        if not query or not response:
            filtered_empty += 1
            continue
            
        # 2. Exclude GSM8K contamination
        norm_query = normalize_question_for_contamination(query)
        norm_orig = normalize_question_for_contamination(original_question) if original_question else ""
        if norm_query in decontam_set or (norm_orig and norm_orig in decontam_set):
            filtered_contam += 1
            continue
            
        # 3. Format and check length
        input_ids, labels = format_qwen_chat(
            tokenizer=tokenizer,
            query=query,
            response=response,
            enable_thinking=enable_thinking,
        )
        
        if len(input_ids) > max_seq_length:
            filtered_length += 1
            continue
            
        # Valid eligible example
        eligible_examples.append({
            "id": idx,
            "query": query,
            "response": response,
            "input_ids": input_ids,
            "labels": labels,
            "length": len(input_ids),
        })
        
        if len(eligible_examples) >= target_count:
            break
            
    logger.info(
        f"Filtered MetaMathQA: collected {len(eligible_examples)} examples. "
        f"Filtered empty: {filtered_empty}, contamination: {filtered_contam}, "
        f"exceeding {max_seq_length} tokens: {filtered_length}."
    )
    return eligible_examples


class MetaMathQADataset(Dataset):
    """PyTorch Dataset for fine-tuning on filtered MetaMathQA."""
    
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.examples[idx]
        return {
            "id": item["id"],
            "input_ids": item["input_ids"],
            "labels": item["labels"],
            "length": item["length"],
        }


class EffectiveBatchCollator:
    """Collates microbatches and computes response token counts for effective batch normalization.
    
    Pads input_ids with pad_token_id, attention_mask with 0, and labels with -100.
    Returns:
        dict with:
          - input_ids: Tensor of shape (B, S)
          - attention_mask: Tensor of shape (B, S)
          - labels: Tensor of shape (B, S)
          - num_valid_tokens: int count of tokens where label != -100
    """
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for EffectiveBatchCollator")
            
        if not batch:
            return {
                "input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
                "labels": torch.empty((0, 0), dtype=torch.long),
                "num_valid_tokens": 0,
            }
            
        max_len = max(len(x["input_ids"]) for x in batch)
        batch_size = len(batch)
        
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        for i, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            input_ids[i, :seq_len] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = torch.tensor(item["labels"], dtype=torch.long)
            
        num_valid_tokens = int((labels != -100).sum().item())
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "num_valid_tokens": num_valid_tokens,
        }


def create_dataloader(
    dataset: MetaMathQADataset,
    batch_size: int = 2,
    pad_token_id: int = 0,
    shuffle: bool = False,
    num_workers: int = 0,
) -> Any:
    """Create a PyTorch DataLoader with EffectiveBatchCollator."""
    if not HAS_TORCH or DataLoader is None:
        raise RuntimeError("PyTorch is required to create a DataLoader")
        
    collator = EffectiveBatchCollator(pad_token_id=pad_token_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
