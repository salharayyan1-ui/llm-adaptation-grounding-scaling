"""Automatic first-pass flags for the Task 1 qualitative study (pure Python; no torch, no model).

The assignment asks whether mathematical fine-tuning "intrudes into unrelated conversations". The qualitative prompts
contain no mathematics, so any math-training *style* in a response is unnecessary mathematical framing. Two kinds of
evidence are flagged, kept apart so the annotator can tell them apart:

* ``artefact`` signals are fixed strings that MetaMathQA / GSM8K training targets end in, or that math-tuned chat
  models emit; they are rare in ordinary prose and high precision:
    - ``answer_is``      the MetaMathQA suffix ``The answer is: <x>``  (the colon is required, so the ordinary
                          sentence "the answer is yes" is not flagged)
    - ``gsm8k_marker``   a line starting with the GSM8K final-answer marker ``#### <number>`` (a markdown
                          heading such as ``#### Summary`` is not flagged)
    - ``boxed``          ``\\boxed{...}``
    - ``latex``          ``$$``, ``\\( \\)``, ``\\[ \\]`` or a math macro such as ``\\frac`` / ``\\times``
    - ``calc_annotation`` GSM8K calculator annotations ``<<48/2=24>>``
* ``vocabulary`` signals are the generic math words used before (``equation``, ``theorem``, ...). They are
  lower precision (an essay about AI may legitimately say "variable") and are reported separately.

``unnecessary_math_framing`` is true when any signal fires; the kinds are stored in ``math_framing_signals`` so a
reader can see why. These are heuristics for orienting the manual annotation, not a replacement for it.
"""

from typing import Dict, List
import re

ARTEFACT_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    "answer_is": re.compile(r"\bthe answer is\s*:", re.IGNORECASE),
    "gsm8k_marker": re.compile(r"^[ \t]*####[ \t]*[-$\d]", re.MULTILINE),
    "boxed": re.compile(r"\\boxed\s*\{"),
    "latex": re.compile(r"\$\$|\\\(|\\\)|\\\[|\\\]|\\(?:frac|times|cdot|sqrt|div|text)\b"),
    "calc_annotation": re.compile(r"<<[^<>\n]*=[^<>\n]*>>"),
}

MATH_KEYWORDS = ["equation", "variable", "theorem", "lemma", "derivative", "integral", "matrix",
                 "vector", "optimize", "objective function", "let x =", "let y =", "proof:", "q.e.d."]
VOCABULARY_PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in MATH_KEYWORDS) + r")(?!\w)", re.IGNORECASE)

SIGNAL_NAMES = tuple(ARTEFACT_PATTERNS) + ("vocabulary",)


def math_framing_signals(text: str) -> List[str]:
    """Names of the math-framing signals present in ``text`` (empty list = none), in a fixed order."""
    found = [name for name, pat in ARTEFACT_PATTERNS.items() if pat.search(text or "")]
    if VOCABULARY_PATTERN.search(text or ""):
        found.append("vocabulary")
    return found


def has_math_artefact(text: str) -> bool:
    """True when a high-precision training-format artefact (not just math vocabulary) is present."""
    return any(name != "vocabulary" for name in math_framing_signals(text))
