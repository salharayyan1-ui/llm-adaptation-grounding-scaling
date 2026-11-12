"""Build the .ipynb notebooks from the percent-format sources (``# %%`` / ``# %% [markdown]`` cells).

    python notebooks/build_ipynb.py            # writes notebooks/task{1,2,3}_analysis.ipynb

The .py files are the source of truth (readable diffs; VS Code and Jupytext open them as notebooks too).
"""
import re
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def convert(py: Path) -> Path:
    cells, kind, buf = [], None, []

    def flush():
        nonlocal buf
        text = "\n".join(buf).strip("\n")
        if kind is not None and text.strip():
            if kind == "markdown":
                text = "\n".join(re.sub(r"^# ?", "", ln) for ln in text.splitlines())
                cells.append(nbf.v4.new_markdown_cell(text))
            else:
                cells.append(nbf.v4.new_code_cell(text))
        buf = []

    for line in py.read_text(encoding="utf-8").splitlines():
        m = re.match(r"# %%(\s*\[markdown\])?\s*$", line)
        if m:
            flush()
            kind = "markdown" if m.group(1) else "code"
        else:
            buf.append(line)
    flush()

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = py.with_suffix(".ipynb")
    nbf.write(nb, out)
    return out


if __name__ == "__main__":
    for p in sorted(HERE.glob("task*_analysis.py")):
        print("wrote", convert(p).relative_to(HERE.parent))
