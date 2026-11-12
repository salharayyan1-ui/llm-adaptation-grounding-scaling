from typing import List, Sequence, Tuple, Union, Optional
import numpy as np
import re

_VQA_PUNCT = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
_VQA_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_VQA_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_VQA_NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_VQA_ARTICLES = {"a", "an", "the"}
_VQA_CONTRACTIONS = {
    "dont": "don't", "doesnt": "doesn't", "didnt": "didn't", "isnt": "isn't", "arent": "aren't",
    "wasnt": "wasn't", "werent": "weren't", "cant": "can't", "couldnt": "couldn't", "wont": "won't",
    "wouldnt": "wouldn't", "shouldnt": "shouldn't", "hasnt": "hasn't", "havent": "haven't",
    "hadnt": "hadn't", "im": "i'm", "ive": "i've", "youre": "you're", "thats": "that's",
    "whats": "what's", "theres": "there's", "its": "it's", "lets": "let's",
}


def normalize_text_vqa(text: str) -> str:
    """VQA answer normalization (follows the official VQA evaluation: punctuation handling, number
    words -> digits, article removal, common contractions). Decimals such as ``3.5`` are preserved."""
    out = str(text).replace("\n", " ").replace("\t", " ").strip().lower()
    in_text = out
    for p in _VQA_PUNCT:
        if (p + " " in in_text or " " + p in in_text) or (re.search(_VQA_COMMA_STRIP, in_text) is not None):
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    out = _VQA_PERIOD_STRIP.sub("", out)
    out = re.sub(_VQA_COMMA_STRIP, r"\1\3", out)
    words = []
    for word in out.split():
        word = _VQA_NUMBER_WORDS.get(word, word)
        if word not in _VQA_ARTICLES:
            words.append(_VQA_CONTRACTIONS.get(word, word))
    return " ".join(words)


def compute_vqa_accuracy(prediction: str, references: Sequence[str]) -> float:
    """Compute local VQA soft consensus accuracy against 10 reference answers:
    s(a_hat) = 1/10 sum_{i=1}^10 min(1, 1/3 sum_{j != i} 1[a_hat == a_j]).
    """
    pred_norm = normalize_text_vqa(prediction)
    refs_norm = [normalize_text_vqa(r) for r in references]
    
    n_refs = len(refs_norm)
    if n_refs == 0:
        return 0.0
    
    scores = []
    for i in range(n_refs):
        # Leave out the i-th annotator
        other_matches = sum(1 for j in range(n_refs) if j != i and pred_norm == refs_norm[j])
        score_i = min(1.0, other_matches / 3.0)
        scores.append(score_i)
        
    return float(np.mean(scores))

def compute_grounding_exact_match(prediction: str, reference: str) -> float:
    """Compute normalized exact match for presence and counting."""
    pred_norm = normalize_text_vqa(prediction)
    ref_norm = normalize_text_vqa(reference)
    return 1.0 if pred_norm == ref_norm else 0.0

def bootstrap_pair_metric_ci(
    pair_values: Sequence[float],
    n_resamples: int = 4000,
    seed: int = 6304,
    ci_level: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval resampling complete pairs.
    
    Args:
        pair_values: Array or list of metric values, one per pair (or paired differences).
        n_resamples: Number of bootstrap draws (default: 4000).
        seed: Random seed for bootstrap.
        ci_level: Confidence level (default: 0.95).
        
    Returns:
        (mean, lower_bound, upper_bound)
    """
    arr = np.asarray(pair_values, dtype=np.float64)
    n_pairs = len(arr)
    if n_pairs == 0:
        return 0.0, 0.0, 0.0
    
    rng = np.random.default_rng(seed)
    boot_means = np.zeros(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        sampled_indices = rng.integers(0, n_pairs, size=n_pairs)
        boot_means[b] = np.mean(arr[sampled_indices])
        
    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    mean = float(np.mean(arr))
    return mean, lower, upper

def compute_subspace_interference(
    delta_w: np.ndarray,
    u_k: np.ndarray,
) -> float:
    """Compute subspace-interference ratio:
    rho_k = ||U_k U_k^T Delta_W||_F^2 / ||Delta_W||_F^2.
    
    Args:
        delta_w: Weight update matrix, shape (d_out, d_in).
        u_k: Leading singular vectors of starting weight, shape (d_out, k).
    """
    norm_delta_sq = np.sum(delta_w ** 2)
    if norm_delta_sq < 1e-12:
        return 0.0
    
    # U_k^T Delta_W has shape (k, d_in)
    projected = u_k.T @ delta_w
    # ||U_k U_k^T Delta_W||_F^2 = ||U_k^T Delta_W||_F^2 because U_k has orthonormal columns
    norm_proj_sq = np.sum(projected ** 2)
    return float(norm_proj_sq / norm_delta_sq)

def compute_relative_update_magnitude(
    delta_w: np.ndarray,
    w_0: np.ndarray,
) -> float:
    """Compute relative update magnitude:
    eta = ||Delta_W||_F / ||W_0||_F.
    """
    norm_w0 = np.linalg.norm(w_0, ord="fro")
    if norm_w0 < 1e-12:
        return 0.0
    norm_delta = np.linalg.norm(delta_w, ord="fro")
    return float(norm_delta / norm_w0)

def compute_activation_ratio(
    delta_w: np.ndarray,
    w_0: np.ndarray,
    z: np.ndarray,
) -> float:
    """Compute activation diagnostic ratio:
    R(z) = ||Delta_W z||_2 / (||W_0 z||_2 + 10^-8).
    
    Args:
        delta_w: shape (d_out, d_in).
        w_0: shape (d_out, d_in).
        z: activation vector, shape (d_in,).
    """
    delta_out = delta_w @ z
    w0_out = w_0 @ z
    norm_delta = np.linalg.norm(delta_out, ord=2)
    norm_w0 = np.linalg.norm(w0_out, ord=2)
    return float(norm_delta / (norm_w0 + 1e-8))
