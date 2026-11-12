"""Live Generation Subtask module for Task 3.

Implements Step 4:
- 50 MATH evaluation questions with Qwen2.5-3B-Instruct in FP16, batch size 1.
- Generation hyperparameters: temp 0.7, top-p 0.8, top-k 20, rep penalty 1.0.
- Total budget: 3,072 output tokens per question.
- Strategy 1: Single longer response (capped at 3,072 new tokens).
- Strategy 2: Parallel sampling (3 independent responses capped at 1,024; majority vote).
- Strategy 3: Sequential refinement (reuse 1st parallel response, prompt critique-and-revision capped at 2,048).
- Tracks actual tokens (input/output), inference runtime, peak GPU memory, truncation,
  and wrong-to-correct / correct-to-wrong transitions.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import time
import json
import numpy as np

from common.device import get_peak_memory_mb, reset_peak_memory_stats, get_torch_device
from common.normalization import extract_boxed_answer
from task3_test_time.equivalence import grade_answer, are_answers_equivalent, NO_ANSWER
from task3_test_time.scoring import select_self_consistency
from task3_test_time.data_loader import RolloutRecord

MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-3B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."
THINK_INSTRUCTION = "Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."
REVISE_INSTRUCTION = (
    "Review the solution above. Identify and correct any errors; if it is correct, "
    "retain its answer. Give your final answer inside \\boxed{}."
)


@dataclass
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    runtime_sec: float
    peak_memory_mb: float
    truncated: bool
    extracted_answer: str
    seed: int


@dataclass
class QuestionLiveResult:
    sample_id: str
    question: str
    gold_answer: str
    strategy_single: GenerationResult
    strategy_parallel: List[GenerationResult]
    parallel_chosen_answer: str
    strategy_refine: GenerationResult
    revision_transition: str  # 'wrong_to_correct', 'correct_to_wrong', 'correct_to_correct', 'wrong_to_wrong'


class QwenLiveGenerator:
    """Safe wrapper for Qwen2.5-3B-Instruct FP16 generation with dry-run support."""

    def __init__(
        self,
        model_name: str = MODEL_NAME_OR_PATH,
        device: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.model_name = model_name
        self.dry_run = dry_run
        self.device = device or ("cuda" if get_torch_device()[1] else "cpu")
        self.model = None
        self.tokenizer = None

        if not dry_run:
            self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading {self.model_name} in FP16 on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device if self.device != "cpu" else None,
                trust_remote_code=True,
            )
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load real model ({e}). Switching to dry-run mode.")
            self.dry_run = True

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        seed: int = 6304,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
    ) -> GenerationResult:
        """Run single generation with specified parameters and memory tracking."""
        if self.dry_run:
            # Deterministic simulated generation for offline testing
            prompt_len = sum(len(m["content"].split()) for m in messages)
            gen_len = min(max_new_tokens, 150)
            mock_ans = "42"
            mock_text = (
                f"<think> Reasoning with seed {seed}... </think>\n"
                f"The final answer is \\boxed{{{mock_ans}}}."
            )
            return GenerationResult(
                text=mock_text,
                input_tokens=prompt_len,
                output_tokens=gen_len,
                runtime_sec=0.05,
                peak_memory_mb=0.0,
                truncated=False,
                extracted_answer=mock_ans,
                seed=seed,
            )

        import torch
        reset_peak_memory_stats()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt_str], return_tensors="pt").to(self.device)
        input_token_count = inputs.input_ids.shape[1]

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed_sec = time.perf_counter() - start_time
        peak_mem = get_peak_memory_mb()

        # Extract only newly generated tokens
        new_tokens = outputs[0][input_token_count:]
        output_token_count = len(new_tokens)
        truncated = output_token_count >= max_new_tokens

        text_out = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        boxed = extract_boxed_answer(text_out)
        extracted = boxed if boxed is not None else NO_ANSWER

        return GenerationResult(
            text=text_out,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            runtime_sec=elapsed_sec,
            peak_memory_mb=peak_mem,
            truncated=truncated,
            extracted_answer=extracted,
            seed=seed,
        )


def run_live_generation_experiment(
    questions: Sequence[Dict[str, Any]],
    generator: QwenLiveGenerator,
    base_seed: int = 6304,
) -> List[QuestionLiveResult]:
    """Run live generation across the 50 MATH evaluation questions."""
    results: List[QuestionLiveResult] = []

    for i, q in enumerate(questions):
        sample_id = q.get("sample_id", f"q_{i:04d}")
        problem = q.get("problem", q.get("question", ""))
        gold_answer = q.get("gold_answer", q.get("gold", ""))

        # Base messages
        base_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{problem}\n\n{THINK_INSTRUCTION}"},
        ]

        # 1. Strategy 1: Single longer response (capped at 3,072)
        single_res = generator.generate(
            messages=base_messages,
            max_new_tokens=3072,
            seed=base_seed + i * 100 + 1,
        )

        # 2. Strategy 2: Parallel sampling (3 x 1,024)
        parallel_runs: List[GenerationResult] = []
        for p_idx in range(3):
            p_res = generator.generate(
                messages=base_messages,
                max_new_tokens=1024,
                seed=base_seed + i * 100 + 10 + p_idx,
            )
            parallel_runs.append(p_res)

        # Self-consistency vote for parallel
        dummy_rollouts = [
            RolloutRecord(
                dataset="math",
                sample_id=sample_id,
                rollout_id=r_i,
                extracted_answer=r.extracted_answer,
                length=r.output_tokens,
            )
            for r_i, r in enumerate(parallel_runs)
        ]
        _, sc_ans = select_self_consistency(dummy_rollouts, dataset="math")

        # 3. Strategy 3: Sequential refinement
        # Reuses the first parallel response
        initial_p0 = parallel_runs[0]
        refine_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{problem}\n\n{THINK_INSTRUCTION}"},
            {"role": "assistant", "content": initial_p0.text},
            {"role": "user", "content": REVISE_INSTRUCTION},
        ]

        refine_res = generator.generate(
            messages=refine_messages,
            max_new_tokens=2048,
            seed=base_seed + i * 100 + 50,
        )

        # Determine revision transition
        p0_correct = grade_answer(initial_p0.extracted_answer, gold_answer, dataset="math")
        ref_correct = grade_answer(refine_res.extracted_answer, gold_answer, dataset="math")

        if not p0_correct and ref_correct:
            transition = "wrong_to_correct"
        elif p0_correct and not ref_correct:
            transition = "correct_to_wrong"
        elif p0_correct and ref_correct:
            transition = "correct_to_correct"
        else:
            transition = "wrong_to_wrong"

        q_res = QuestionLiveResult(
            sample_id=sample_id,
            question=problem,
            gold_answer=gold_answer,
            strategy_single=single_res,
            strategy_parallel=parallel_runs,
            parallel_chosen_answer=sc_ans,
            strategy_refine=refine_res,
            revision_transition=transition,
        )
        results.append(q_res)

    return results


def summarize_live_results(results: Sequence[QuestionLiveResult]) -> Dict[str, Any]:
    """Compute aggregate accuracy, actual token costs, runtimes, truncation rates, and transitions."""
    N = len(results)
    if N == 0:
        return {}

    single_accs = []
    parallel_accs = []
    refine_accs = []

    single_out_tokens = []
    parallel_out_tokens = []
    refine_out_tokens = []

    single_runtimes = []
    parallel_runtimes = []
    refine_runtimes = []

    single_trunc = []
    parallel_trunc = []
    refine_trunc = []

    transitions = {
        "wrong_to_correct": 0,
        "correct_to_wrong": 0,
        "correct_to_correct": 0,
        "wrong_to_wrong": 0,
    }

    for r in results:
        # Single
        s_corr = grade_answer(r.strategy_single.extracted_answer, r.gold_answer, dataset="math")
        single_accs.append(1.0 if s_corr else 0.0)
        single_out_tokens.append(r.strategy_single.output_tokens)
        single_runtimes.append(r.strategy_single.runtime_sec)
        single_trunc.append(1.0 if r.strategy_single.truncated else 0.0)

        # Parallel
        p_corr = grade_answer(r.parallel_chosen_answer, r.gold_answer, dataset="math")
        parallel_accs.append(1.0 if p_corr else 0.0)
        total_p_tok = sum(p.output_tokens for p in r.strategy_parallel)
        total_p_time = sum(p.runtime_sec for p in r.strategy_parallel)
        parallel_out_tokens.append(total_p_tok)
        parallel_runtimes.append(total_p_time)
        parallel_trunc.append(1.0 if any(p.truncated for p in r.strategy_parallel) else 0.0)

        # Refine (initial response charged + refine response)
        ref_corr = grade_answer(r.strategy_refine.extracted_answer, r.gold_answer, dataset="math")
        refine_accs.append(1.0 if ref_corr else 0.0)
        ref_tok = r.strategy_parallel[0].output_tokens + r.strategy_refine.output_tokens
        ref_time = r.strategy_parallel[0].runtime_sec + r.strategy_refine.runtime_sec
        refine_out_tokens.append(ref_tok)
        refine_runtimes.append(ref_time)
        refine_trunc.append(1.0 if r.strategy_refine.truncated else 0.0)

        transitions[r.revision_transition] += 1

    return {
        "count": N,
        "accuracy": {
            "single": float(np.mean(single_accs)),
            "parallel_sc": float(np.mean(parallel_accs)),
            "refinement": float(np.mean(refine_accs)),
        },
        "mean_output_tokens": {
            "single": float(np.mean(single_out_tokens)),
            "parallel": float(np.mean(parallel_out_tokens)),
            "refinement": float(np.mean(refine_out_tokens)),
        },
        "mean_runtime_sec": {
            "single": float(np.mean(single_runtimes)),
            "parallel": float(np.mean(parallel_runtimes)),
            "refinement": float(np.mean(refine_runtimes)),
        },
        "truncation_rate": {
            "single": float(np.mean(single_trunc)),
            "parallel": float(np.mean(parallel_trunc)),
            "refinement": float(np.mean(refine_trunc)),
        },
        "transitions_count": transitions,
        "transitions_pct": {k: float(v / N) for k, v in transitions.items()},
    }


def save_live_results(results: Sequence[QuestionLiveResult], output_path: str) -> None:
    """Save live generation outputs and metrics to JSON."""
    data = [asdict(r) for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
