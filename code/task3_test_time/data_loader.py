"""Data loader and alignment verification module for Task 3.

Ingests rollout data from ATML-PA3-Data:
- Qwen2.5-3B-Instruct on MATH (1,000 questions x 32 rollouts)
- Qwen2.5-VL-3B-Instruct on MathVista (1,000 questions x 32 rollouts)
- Tables: generations.parquet, probability_metrics.parquet, attention_metrics.parquet
- Joined on (dataset, sample_id, rollout_id)
- Verifies token list alignment across tables.
- Includes synthetic dataset generation for standalone testing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union
import os
import json
import numpy as np


@dataclass
class RolloutRecord:
    dataset: str
    sample_id: str
    rollout_id: int
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    length: int = 0
    extracted_answer: str = "NO_ANSWER"
    gold_answer: str = ""
    logp: List[float] = field(default_factory=list)
    kl_u_p: List[float] = field(default_factory=list)
    att_image: Optional[List[float]] = None
    att_prompt: Optional[List[float]] = None
    att_generated: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.length if self.length > 0 else len(self.logp)


@dataclass
class QuestionRecord:
    sample_id: str
    dataset: str
    gold_answer: str
    rollouts: List[RolloutRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def verify_token_alignment(rollout: RolloutRecord) -> bool:
    """Verify that token counts match across lengths and metric sequences."""
    T = rollout.T
    if len(rollout.logp) != T:
        return False
    if len(rollout.kl_u_p) != T:
        return False
    if rollout.att_image is not None and len(rollout.att_image) != T:
        return False
    if rollout.att_prompt is not None and len(rollout.att_prompt) != T:
        return False
    if rollout.att_generated is not None and len(rollout.att_generated) != T:
        return False
    return True


def load_dataset(
    data_dir: str,
    dataset_name: str = "math",
    verify_alignment: bool = True,
) -> Dict[str, QuestionRecord]:
    """Load and join parquet tables for the specified dataset.
    
    Args:
        data_dir: Root directory of ATML-PA3-Data or specific model folder.
        dataset_name: 'math' or 'mathvista'.
        verify_alignment: If True, asserts token list alignment across tables.
        
    Returns:
        Dictionary mapping sample_id to QuestionRecord.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required to load parquet rollout tables.") from e

    ds_lower = dataset_name.lower()
    
    # Locate subfolder
    subfolder_candidates = [
        data_dir,
        os.path.join(data_dir, "math" if "vista" not in ds_lower else "mathvista"),
        os.path.join(data_dir, "Qwen2.5-3B-Instruct" if "vista" not in ds_lower else "Qwen2.5-VL-3B-Instruct"),
    ]
    target_dir = None
    for cand in subfolder_candidates:
        if os.path.exists(cand) and (
            os.path.exists(os.path.join(cand, "generations.parquet"))
            or os.path.exists(os.path.join(cand, "generations.json"))
        ):
            target_dir = cand
            break
            
    if target_dir is None:
        target_dir = data_dir

    gen_path = os.path.join(target_dir, "generations.parquet")
    prob_path = os.path.join(target_dir, "probability_metrics.parquet")
    att_path = os.path.join(target_dir, "attention_metrics.parquet")

    # Fallback to json if parquet does not exist
    if not os.path.exists(gen_path) and os.path.exists(os.path.join(target_dir, "generations.json")):
        with open(os.path.join(target_dir, "generations.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            qid: QuestionRecord(
                sample_id=qid,
                dataset=dataset_name,
                gold_answer=rec["gold_answer"],
                rollouts=[RolloutRecord(**r) for r in rec["rollouts"]],
                metadata=rec.get("metadata", {}),
            )
            for qid, rec in data.items()
        }

    # Load Parquet
    df_gen = pd.read_parquet(gen_path)
    df_prob = pd.read_parquet(prob_path)
    
    join_keys = ["dataset", "sample_id", "rollout_id"]
    # Check if dataset column exists
    for key in ["dataset", "sample_id", "rollout_id"]:
        if key not in df_gen.columns:
            join_keys = [k for k in join_keys if k != key]

    df_merged = pd.merge(df_gen, df_prob, on=join_keys, how="inner", suffixes=("", "_prob"))

    df_att = None
    if os.path.exists(att_path):
        df_att = pd.read_parquet(att_path)
        df_merged = pd.merge(df_merged, df_att, on=join_keys, how="inner", suffixes=("", "_att"))

    questions: Dict[str, QuestionRecord] = {}

    for _, row in df_merged.iterrows():
        sample_id = str(row["sample_id"])
        rollout_id = int(row["rollout_id"])
        dataset = str(row.get("dataset", dataset_name))
        gold = str(row.get("gold_answer", row.get("gold", "")))
        extracted = str(row.get("extracted_answer", row.get("pred", "NO_ANSWER")))
        
        token_ids = list(row["token_ids"]) if "token_ids" in row and hasattr(row["token_ids"], "__iter__") else []
        length = int(row.get("length", len(token_ids)))
        logp = [float(x) for x in row["logp"]] if "logp" in row else []
        kl_u_p = [float(x) for x in row["kl_u_p"]] if "kl_u_p" in row else []

        att_image = [float(x) for x in row["image_attention"]] if "image_attention" in row and row["image_attention"] is not None else None
        att_prompt = [float(x) for x in row["prompt_attention"]] if "prompt_attention" in row and row["prompt_attention"] is not None else None
        att_gen = [float(x) for x in row["generated_attention"]] if "generated_attention" in row and row["generated_attention"] is not None else None

        rec = RolloutRecord(
            dataset=dataset,
            sample_id=sample_id,
            rollout_id=rollout_id,
            text=str(row.get("text", "")),
            token_ids=token_ids,
            length=length if length > 0 else len(logp),
            extracted_answer=extracted,
            gold_answer=gold,
            logp=logp,
            kl_u_p=kl_u_p,
            att_image=att_image,
            att_prompt=att_prompt,
            att_generated=att_gen,
            metadata=row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {},
        )

        if verify_alignment and not verify_token_alignment(rec):
            raise ValueError(
                f"Token alignment mismatch for sample {sample_id}, rollout {rollout_id}: "
                f"length={rec.length}, logp_len={len(rec.logp)}, kl_len={len(rec.kl_u_p)}"
            )

        if sample_id not in questions:
            questions[sample_id] = QuestionRecord(
                sample_id=sample_id,
                dataset=dataset,
                gold_answer=gold,
                metadata=rec.metadata,
                rollouts=[],
            )
        questions[sample_id].rollouts.append(rec)

    # Sort rollouts by rollout_id for each question
    for q in questions.values():
        q.rollouts.sort(key=lambda r: r.rollout_id)

    return questions


def generate_synthetic_data(
    dataset_name: str = "math",
    n_questions: int = 10,
    n_rollouts: int = 32,
    seed: int = 6304,
) -> Dict[str, QuestionRecord]:
    """Generate realistic synthetic question & rollout records for verification and testing.
    
    Args:
        dataset_name: 'math' or 'mathvista'.
        n_questions: Number of questions to simulate.
        n_rollouts: Number of rollouts per question (default: 32).
        seed: Random seed.
        
    Returns:
        Dict mapping sample_id to QuestionRecord.
    """
    rng = np.random.default_rng(seed)
    questions: Dict[str, QuestionRecord] = {}
    is_vista = "vista" in dataset_name.lower()
    
    math_answers = ["4", "\\frac{1}{2}", "12", "0", "\\pi", "25", "\\sqrt{3}", "NO_ANSWER"]
    vista_answers = ["A", "B", "C", "D", "12.5", "42", "NO_ANSWER"]
    
    for q_idx in range(n_questions):
        sample_id = f"q_{q_idx:04d}"
        gold_ans = math_answers[q_idx % (len(math_answers) - 1)] if not is_vista else vista_answers[q_idx % (len(vista_answers) - 1)]
        rollouts: List[RolloutRecord] = []
        
        q_meta = {
            "question_type": "multi_choice" if is_vista and (q_idx % 2 == 0) else "number",
            "choices": ["A", "B", "C", "D"] if is_vista and (q_idx % 2 == 0) else [],
        }

        for r_id in range(n_rollouts):
            T = int(rng.integers(20, 100))
            # 60% chance of matching gold if correct rollout
            if rng.random() < 0.35:
                ans = gold_ans
            else:
                pool = math_answers if not is_vista else vista_answers
                ans = pool[int(rng.integers(0, len(pool)))]

            # Probabilities and KL
            logp = (-rng.exponential(scale=0.5, size=T)).tolist()
            kl_u_p = (rng.normal(loc=3.5, scale=0.8, size=T) + 4.0).clip(0.1, 15.0).tolist()

            att_img, att_pmt, att_gen = None, None, None
            if is_vista:
                att_img = rng.uniform(10.0, 50.0, size=T).tolist()
                att_pmt = rng.uniform(20.0, 60.0, size=T).tolist()
                att_gen = rng.uniform(5.0, 40.0, size=T).tolist()

            r = RolloutRecord(
                dataset=dataset_name,
                sample_id=sample_id,
                rollout_id=r_id,
                text=f"Sample solution for {sample_id} rollout {r_id} with answer \\boxed{{{ans}}}.",
                token_ids=list(range(T)),
                length=T,
                extracted_answer=ans,
                gold_answer=gold_ans,
                logp=logp,
                kl_u_p=kl_u_p,
                att_image=att_img,
                att_prompt=att_pmt,
                att_generated=att_gen,
                metadata=q_meta,
            )
            rollouts.append(r)

        questions[sample_id] = QuestionRecord(
            sample_id=sample_id,
            dataset=dataset_name,
            gold_answer=gold_ans,
            rollouts=rollouts,
            metadata=q_meta,
        )

    return questions
