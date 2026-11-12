"""Equivalence and normalization module for Task 3.

Implements:
- Math equivalence: math-verify integration for MATH with sympy fallback.
- MathVista normalization: answer-type, choice options (A-H), precision rules.
- Missing extractions mapped to single "NO_ANSWER" group (always graded incorrect).
- Answer grouping strictly WITHOUT consulting gold answers.
- Scorer and equivalence verification for grading against gold answers.
- High-performance memoization cache and fast numeric paths.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import functools
import re
import math

try:
    import sympy
    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False

try:
    import math_verify
    _HAS_MATH_VERIFY = True
except ImportError:
    _HAS_MATH_VERIFY = False

NO_ANSWER = "NO_ANSWER"
SCORER_VERSION = "math-verify-0.3.0+sympy-fallback"


def normalize_math_candidate(ans: Optional[str]) -> str:
    """Normalize extracted math candidate string."""
    if ans is None:
        return NO_ANSWER
    text = str(ans).strip()
    if not text or text == NO_ANSWER:
        return NO_ANSWER
    
    # Strip wrapping $ ... $ or $$ ... $$
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    # Strip \text{...} and \mathrm{...}
    text = re.sub(r"^\\(?:text|mathrm|mathbf)\{([^\}]+)\}$", r"\1", text).strip()
    # Remove leading/trailing spaces and trailing periods
    text = re.sub(r"\.+$", "", text).strip()
    return text if text else NO_ANSWER


def normalize_mathvista_candidate(
    ans: Optional[str],
    question_type: Optional[str] = None,
    choices: Optional[Sequence[str]] = None,
) -> str:
    """Normalize MathVista answer string according to type and choice rules."""
    if ans is None:
        return NO_ANSWER
    text = str(ans).strip()
    if not text or text == NO_ANSWER:
        return NO_ANSWER
        
    # Check for multiple-choice letter (e.g. "A", "(A)", "A.", "Option A")
    mc_match = re.search(r"\b([A-H])\b", text, re.IGNORECASE)
    is_mc = (question_type == "multi_choice") or (choices is not None and len(choices) > 0)
    if is_mc or (len(text) <= 5 and mc_match):
        if mc_match:
            return mc_match.group(1).upper()
            
    # Numeric normalization
    num_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if num_match and (question_type in ["integer", "float", "number"] or not question_type):
        try:
            val = float(num_match.group(0))
            if math.isfinite(val):
                if val.is_integer():
                    return str(int(val))
                return str(round(val, 4))
        except (ValueError, OverflowError):
            pass
            
    return text.lower().strip()


def check_math_verify_equiv(pred: str, gold: str) -> bool:
    """Use math_verify package to test equivalence."""
    if not _HAS_MATH_VERIFY:
        return False
    try:
        from math_verify import parse, verify
        parsed_pred = parse(pred)
        parsed_gold = parse(gold)
        return bool(verify(parsed_pred, parsed_gold))
    except Exception:
        return False


def _latex_to_sympy_str(s: str) -> str:
    """Convert common LaTeX math expressions to SymPy-readable syntax."""
    prev = ""
    curr = s
    while prev != curr:
        prev = curr
        curr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", curr)
    curr = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", curr)
    curr = curr.replace("^", "**")
    curr = curr.replace("\\cdot", "*").replace("\\times", "*")
    curr = curr.replace("{", "(").replace("}", ")")
    return curr


def check_sympy_equiv(pred: str, gold: str) -> bool:
    """Fallback mathematical equivalence check using sympy."""
    if not _HAS_SYMPY:
        return False
    p = _latex_to_sympy_str(pred)
    g = _latex_to_sympy_str(gold)
    try:
        sp_pred = sympy.sympify(p, evaluate=True)
        sp_gold = sympy.sympify(g, evaluate=True)
        diff = sympy.simplify(sp_pred - sp_gold)
        return bool(diff == 0)
    except Exception:
        pass
    return False


@functools.lru_cache(maxsize=32768)
def _cached_check_math_equiv(n1: str, n2: str) -> bool:
    """Cached math equivalence comparison."""
    if n1.lower() == n2.lower():
        return True

    # Fast float parse
    try:
        f1 = float(n1)
        f2 = float(n2)
        if math.isfinite(f1) and math.isfinite(f2):
            return abs(f1 - f2) < 1e-5
    except ValueError:
        pass

    # Simple fraction parse: a/b or \frac{a}{b}
    if "/" in n1 or "/" in n2 or "\\frac" in n1 or "\\frac" in n2:
        try:
            def _parse_frac(s: str) -> float:
                s = s.strip()
                m = re.match(r"^\\frac\{([^}]+)\}\{([^}]+)\}$", s)
                if m:
                    return float(m.group(1)) / float(m.group(2))
                if "/" in s:
                    num, den = s.split("/", 1)
                    return float(num) / float(den)
                return float(s)
            f1 = _parse_frac(n1)
            f2 = _parse_frac(n2)
            if math.isfinite(f1) and math.isfinite(f2):
                return abs(f1 - f2) < 1e-5
        except (ValueError, ZeroDivisionError):
            pass

    if _HAS_MATH_VERIFY and check_math_verify_equiv(n1, n2):
        return True

    if check_sympy_equiv(n1, n2):
        return True

    return False


def are_answers_equivalent(
    ans1: str,
    ans2: str,
    dataset: str = "math",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Check whether two answer strings are mathematically or semantically equivalent."""
    if ans1 == NO_ANSWER or ans2 == NO_ANSWER:
        return ans1 == ans2  # NO_ANSWER is only equal to NO_ANSWER
        
    ds = dataset.lower()
    if "vista" in ds:
        q_type = metadata.get("question_type") if metadata else None
        choices = metadata.get("choices") if metadata else None
        n1 = normalize_mathvista_candidate(ans1, q_type, choices)
        n2 = normalize_mathvista_candidate(ans2, q_type, choices)
        if n1 == n2:
            return True
        # Fast numeric tolerance for floats
        try:
            f1, f2 = float(n1), float(n2)
            if math.isfinite(f1) and math.isfinite(f2):
                return abs(f1 - f2) < 1e-4
        except ValueError:
            pass
        return False
    else:
        # MATH dataset
        n1 = normalize_math_candidate(ans1)
        n2 = normalize_math_candidate(ans2)
        if n1 == NO_ANSWER or n2 == NO_ANSWER:
            return n1 == n2
        return _cached_check_math_equiv(n1, n2)


def group_answers(
    answers: Sequence[Optional[str]],
    dataset: str = "math",
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Dict[str, List[int]]]:
    """Group candidate answers strictly WITHOUT consulting gold answers.
    
    Args:
        answers: Sequence of candidate answers (length N).
        dataset: Dataset name ('math' or 'mathvista').
        metadata: Optional question metadata.
        
    Returns:
        canonical_labels: List of canonical group strings for each candidate.
        groups: Dict mapping canonical group name to list of candidate indices.
    """
    canonical_labels: List[str] = []
    groups: Dict[str, List[int]] = {}
    representatives: List[str] = []
    
    for idx, raw_ans in enumerate(answers):
        norm = (
            normalize_mathvista_candidate(raw_ans, metadata.get("question_type") if metadata else None)
            if "vista" in dataset.lower()
            else normalize_math_candidate(raw_ans)
        )
        
        if norm == NO_ANSWER:
            canonical_labels.append(NO_ANSWER)
            if NO_ANSWER not in groups:
                groups[NO_ANSWER] = []
            groups[NO_ANSWER].append(idx)
            continue
            
        matched_rep = None
        for rep in representatives:
            if are_answers_equivalent(norm, rep, dataset=dataset, metadata=metadata):
                matched_rep = rep
                break
                
        if matched_rep is not None:
            canonical_labels.append(matched_rep)
            groups[matched_rep].append(idx)
        else:
            representatives.append(norm)
            canonical_labels.append(norm)
            groups[norm] = [idx]
            
    return canonical_labels, groups


def grade_answer(
    predicted_answer: Optional[str],
    gold_answer: Optional[str],
    dataset: str = "math",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Grade a predicted answer against the gold answer.
    
    Missing extractions (NO_ANSWER) are always graded incorrect.
    """
    if predicted_answer is None or gold_answer is None:
        return False
    norm_pred = (
        normalize_mathvista_candidate(predicted_answer, metadata.get("question_type") if metadata else None)
        if "vista" in dataset.lower()
        else normalize_math_candidate(predicted_answer)
    )
    if norm_pred == NO_ANSWER:
        return False
        
    return are_answers_equivalent(norm_pred, gold_answer, dataset=dataset, metadata=metadata)
