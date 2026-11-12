"""Load the real MATH problem statements needed by the live-generation subtask.

The supplied rollout tables contain generations, not necessarily the question text, so the problem
text is resolved from (in order):
  1. an explicit file (``--problems_file``): json / jsonl / parquet with ``sample_id`` and ``problem``;
  2. a problem/question column of the supplied generations table (stored by the loader in
     ``QuestionRecord.metadata['problem']``);
  3. the public ``HuggingFaceH4/MATH-500`` dataset, matched on ``unique_id`` == ``sample_id``.
Nothing is substituted: if any requested question cannot be resolved, a ``ValueError`` is raised.
"""

from typing import Dict, List, Optional, Sequence
import json
import os


def _from_file(path: str) -> Dict[str, str]:
    import pandas as pd

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            return {str(k): (v if isinstance(v, str) else v["problem"]) for k, v in raw.items()}
        df = pd.DataFrame(raw)
    id_col = "sample_id" if "sample_id" in df.columns else "unique_id"
    text_col = next((c for c in ("problem", "question", "prompt") if c in df.columns), None)
    if id_col not in df.columns or text_col is None:
        raise KeyError(f"{path} needs sample_id and problem columns; found {list(df.columns)}")
    return {str(a): str(b) for a, b in zip(df[id_col], df[text_col])}


def _from_math500() -> Dict[str, str]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return {str(r["unique_id"]): str(r["problem"]) for r in ds}


def load_math_problems(
    questions: Dict[str, "object"],
    qids: Sequence[str],
    problems_file: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return [{'sample_id', 'problem', 'gold_answer'}] for ``qids`` using real problem statements."""
    resolved: Dict[str, str] = {}
    if problems_file:
        resolved.update(_from_file(problems_file))
    for qid in qids:
        if qid not in resolved:
            prob = questions[qid].metadata.get("problem")
            if prob:
                resolved[qid] = prob
    missing = [q for q in qids if q not in resolved]
    if missing:
        try:
            hub = _from_math500()
            for qid in missing:
                for key in (qid, f"{qid}.json"):
                    if key in hub:
                        resolved[qid] = hub[key]
                        break
        except Exception as e:  # noqa: BLE001
            print(f"Could not use HuggingFaceH4/MATH-500 as a problem source: {e}")
    missing = [q for q in qids if q not in resolved]
    if missing:
        raise ValueError(
            f"No problem text for {len(missing)} live-generation questions (e.g. {missing[:3]}). "
            "Pass --problems_file with sample_id/problem columns (see task3_test_time/problems.py)."
        )
    return [
        {"sample_id": q, "problem": resolved[q], "gold_answer": questions[q].gold_answer}
        for q in qids
    ]
