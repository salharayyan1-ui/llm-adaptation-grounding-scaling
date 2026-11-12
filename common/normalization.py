import re
from typing import Optional, List

def normalize_question_for_contamination(text: str) -> str:
    """Normalize question text for contamination comparison against GSM8K."""
    if not text:
        return ""
    text = str(text).lower()
    # Strip LaTeX commands (e.g. \textbf, \mathrm, \text, \frac, \mathbf)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # Strip non-alphanumeric characters (keep letters and digits; remove punctuation and symbols)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Normalize whitespaces (collapsing multiple spaces, including those resulting from stripped symbols)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_boxed_answer(text: Optional[str]) -> Optional[str]:
    """Extract the last \boxed{...} answer using balanced-brace parsing.
    
    Returns the inner text of the last valid \boxed{...}, or None if not found.
    Handles deeply nested braces, environments, and whitespace between boxed and brace.
    Safe against malformed text, non-string inputs, and unbalanced braces.
    """
    if text is None or not isinstance(text, str) or "\\boxed" not in text:
        return None
    
    last_boxed_content = None
    pattern = re.compile(r"\\boxed\s*\{")
    
    for match in pattern.finditer(text):
        brace_start = match.end()
        depth = 1
        curr = brace_start
        while curr < len(text):
            ch = text[curr]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_boxed_content = text[brace_start:curr].strip()
                    break
            curr += 1
            
    return last_boxed_content

def normalize_math_answer(answer: Optional[str]) -> str:
    """Normalize extracted math answer string."""
    if answer is None:
        return "NO_ANSWER"
    ans = str(answer).strip()
    if not ans:
        return "NO_ANSWER"
    # Remove leading/trailing dollar signs or markdown
    ans = re.sub(r"^\$+|\$+$", "", ans).strip()
    # Remove text{} wraps
    ans = re.sub(r"^\\text\{([^\}]+)\}$", r"\1", ans).strip()
    ans = re.sub(r"^\\mathrm\{([^\}]+)\}$", r"\1", ans).strip()
    return ans if ans else "NO_ANSWER"

def normalize_mathvista_answer(answer: Optional[str], question_type: Optional[str] = None) -> str:
    """Normalize MathVista answer following its answer-type and choice rules."""
    if answer is None:
        return "NO_ANSWER"
    ans = str(answer).strip()
    if not ans:
        return "NO_ANSWER"
    
    # Check for multiple choice single letter (e.g., '(A)', 'A.', 'A')
    mc_match = re.search(r"\b([A-H])\b", ans, re.IGNORECASE)
    if question_type == "multi_choice" or (len(ans) <= 4 and mc_match):
        if mc_match:
            return mc_match.group(1).upper()
            
    # For numeric answers, strip extraneous text
    num_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ans)
    if num_match and question_type in ["integer", "float", "number"]:
        try:
            val = float(num_match.group(0))
            if val.is_integer():
                return str(int(val))
            return str(round(val, 4))
        except ValueError:
            pass
            
    return ans.strip().lower()
