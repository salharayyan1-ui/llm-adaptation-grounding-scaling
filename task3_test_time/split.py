"""Dataset splitting module for Task 3.

Implements the specification:
- Sort 1,000 question IDs lexicographically.
- Permute with NumPy's default_rng(6304).
- First 200 questions: development.
- Remaining 800 questions: held-out evaluation.
- First 50 MATH evaluation questions: live generation subtask.
"""

from typing import Dict, List, Sequence, Union
import json
import numpy as np


def generate_splits(
    question_ids: Sequence[Union[str, int]],
    seed: int = 6304,
    dev_count: int = 200,
    live_count: int = 50,
) -> Dict[str, List[str]]:
    """Generate deterministic train/dev/eval splits.
    
    Args:
        question_ids: Collection of sample/question IDs.
        seed: Random seed for default_rng (default: 6304).
        dev_count: Number of questions for development split (default: 200).
        live_count: Number of questions for live generation (first N of eval).
        
    Returns:
        Dict with keys:
            'sorted': Lexicographically sorted unique question IDs.
            'permuted': Permuted question IDs.
            'dev': First dev_count question IDs.
            'eval': Remaining question IDs.
            'live_math': First live_count question IDs of eval split.
    """
    unique_ids = sorted(list({str(qid) for qid in question_ids}))
    
    rng = np.random.default_rng(seed)
    permuted_indices = rng.permutation(len(unique_ids))
    permuted_ids = [unique_ids[i] for i in permuted_indices]
    
    dev_split = permuted_ids[:dev_count]
    eval_split = permuted_ids[dev_count:]
    live_math_split = eval_split[:live_count]
    
    return {
        "sorted": unique_ids,
        "permuted": permuted_ids,
        "dev": dev_split,
        "eval": eval_split,
        "live_math": live_math_split,
    }


def save_splits(splits: Dict[str, List[str]], filepath: str) -> None:
    """Save split dictionary to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def load_splits(filepath: str) -> Dict[str, List[str]]:
    """Load split dictionary from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
