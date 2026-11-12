"""Shared helpers for the analysis notebooks.

Only numpy / pandas / matplotlib are needed (no torch, no GPU): copy ``results/`` back from the GPU machine and run
the notebooks locally. Two environment variables redirect I/O (used by the tests, handy for experiments):
``PA3_RESULTS`` (where the result files are read from) and ``PA3_OUT`` (where figures/ and tables/ are written).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # rich tables inside Jupyter; plain print when the file is run as a script
    from IPython.display import display
except Exception:  # noqa: BLE001
    display = print

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("PA3_RESULTS", ROOT / "results"))
_OUT = Path(os.environ.get("PA3_OUT", ROOT))
FIGURES = _OUT / "figures"
TABLES = _OUT / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------------------------------------
# Colour. Validated categorical palette (light surface), FIXED slot order; a colour belongs to an entity (run, condition,
# method) and never moves with its rank. Text always wears the ink tokens, never a series colour. Identity is never
# colour-only: every multi-series chart also has a legend and a secondary encoding (line style / marker / position).
# --------------------------------------------------------------------------------------------------------------------
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#b9b8b2"
DIV_LOW, DIV_MID, DIV_HIGH = "#e34948", "#f0efec", "#2a78d6"     # diverging pair: red <-> neutral grey <-> blue


def set_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.5, "lines.markersize": 4.5, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


set_style()


# --------------------------------------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------------------------------------
_warned: set = set()


def _warn(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(msg)


def load_json(rel: str, default: Any = None) -> Any:
    """Read ``RESULTS/rel``; returns ``default`` (and says so once) if the file does not exist yet."""
    p = RESULTS / rel
    if not p.exists():
        _warn(f"[not available yet] {rel}")
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(rel: str) -> Optional[pd.DataFrame]:
    p = RESULTS / rel
    if not p.exists():
        _warn(f"[not available yet] {rel}")
        return None
    return pd.read_csv(p)


def exists(rel: str) -> bool:
    return (RESULTS / rel).exists()


def savefig(fig: "plt.Figure", name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    print(f"saved figures/{name}.pdf (+png)")


# --------------------------------------------------------------------------------------------------------------------
# Number / table formatting
# --------------------------------------------------------------------------------------------------------------------
def pct(x: Optional[float], d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def signed(x: Optional[float], d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{d}f}"


def fmt_ci(mean: float, lo: float, hi: float, scale: float = 100.0, d: int = 1) -> str:
    return f"{mean * scale:.{d}f} [{lo * scale:.{d}f}, {hi * scale:.{d}f}]"


def ci_of(d: Dict[str, Any]) -> tuple:
    """(mean, lo, hi) from either {'mean','ci_lower','ci_upper'} or {'mean','ci_95':[lo,hi]}."""
    if "ci_95" in d:
        return d["mean"], d["ci_95"][0], d["ci_95"][1]
    return d["mean"], d["ci_lower"], d["ci_upper"]


def _tex_escape(s: Any) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    d = df.reset_index() if index else df
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def tex_table(df: pd.DataFrame, caption: str = "", label: str = "", index: bool = True) -> str:
    d = df.reset_index() if index else df
    cols = "l" + "r" * (len(d.columns) - 1)
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(_tex_escape(c) for c in d.columns) + r" \\", r"\midrule"]
    for r in d.itertuples(index=False):
        lines.append(" & ".join(_tex_escape(v) for v in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(rf"\caption{{{_tex_escape(caption)}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, name: str, caption: str = "", label: str = "", index: bool = True) -> pd.DataFrame:
    """Write tables/<name>.md and tables/<name>.tex (booktabs) and return the frame for display."""
    (TABLES / f"{name}.md").write_text(md_table(df, index) + "\n", encoding="utf-8")
    (TABLES / f"{name}.tex").write_text(tex_table(df, caption, label or f"tab:{name}", index) + "\n", encoding="utf-8")
    print(f"saved tables/{name}.md / .tex")
    return df


# --------------------------------------------------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------------------------------------------------
def smooth(y: Sequence[float], window: int = 25) -> np.ndarray:
    return pd.Series(np.asarray(y, dtype=float)).rolling(window, min_periods=1, center=True).mean().to_numpy()


def end_labels(ax, xs: Sequence[float], ys: Sequence[float], labels: Sequence[str], min_gap: float = 0.05,
               dx_frac: float = 0.01) -> None:
    """Direct labels at the right end of lines (ink colour), spread vertically so they never collide."""
    if not len(labels):
        return
    y0, y1 = ax.get_ylim()
    span = (y1 - y0) or 1.0
    order = np.argsort(ys)
    pos = {}
    last = None
    for i in order:
        y = ys[i]
        if last is not None and y - last < min_gap * span:
            y = last + min_gap * span
        pos[i] = y
        last = y
    x0, x1 = ax.get_xlim()
    for i, lab in enumerate(labels):
        ax.annotate(lab, (xs[i], ys[i]), xytext=(x1 + dx_frac * (x1 - x0), pos[i]), textcoords="data",
                    fontsize=7.5, color=INK2, va="center", annotation_clip=False)


def diverging_cmap() -> "mpl.colors.Colormap":
    return mpl.colors.LinearSegmentedColormap.from_list("div", [DIV_LOW, DIV_MID, DIV_HIGH])


def panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, fontsize=8.5, pad=4)


def stage_bands(ax, bounds: Sequence[float], labels: Sequence[str] = ("S1", "S2", "S3")) -> None:
    """Light vertical rules at stage boundaries plus stage names along the top of the axes."""
    edges = [0.0] + list(bounds)
    for b in bounds[:-1]:
        ax.axvline(b, color=AXIS, lw=0.8, ls=(0, (3, 3)), zorder=0)
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        ax.text((lo + hi) / 2, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7, color=MUTED)


def show_excerpt(text: str, n: int = 220) -> str:
    t = re.sub(r"\s+", " ", str(text)).strip()
    return t if len(t) <= n else t[:n] + "…"
