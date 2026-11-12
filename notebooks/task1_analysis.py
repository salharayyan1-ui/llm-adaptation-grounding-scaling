# %% [markdown]
# # Task 1 — Fine-tuning and retention: analysis
#
# Reads the small result files produced by `scripts/run_task1.py` / `scripts/queue_overnight.sh`
# (`results/task1/…`) and produces the evidence required for Task 1:
#
# 1. benchmark table — 7 conditions × 4 benchmarks, primary scores and changes from the baseline in **percentage points**
# 2. the controlled comparisons (rank, explicit regularisation, orthogonal protection, Full FT vs LoRA-16)
# 3. learning curves for the five runs + L2-SP penalty / displacement
# 4. resources (trainable parameters, training time, peak memory, checkpoint sizes)
# 5. weight-space and activation diagnostics (ρ_k, η, R(z)), with OPLoRA's numerical verification kept separate
# 6. the fixed 42-response qualitative subset and aligned excerpts
#
# Every section degrades gracefully: a missing file only prints `[not available yet]`.
# Figures go to `figures/`, tables to `tables/` (`.md` and `.tex`).

# %%
from nb_utils import *

T1 = "task1"
CONDS = ["baseline", "full_ft_midpoint", "full_ft_final", "full_ft_l2_sp_final",
         "lora_4_final", "lora_16_final", "oplora_16_final"]
LABEL = {"baseline": "Baseline", "full_ft_midpoint": "Full FT (mid)", "full_ft_final": "Full FT",
         "full_ft_l2_sp_final": "Full FT + L2-SP", "lora_4_final": "LoRA-4", "lora_16_final": "LoRA-16",
         "oplora_16_final": "OPLoRA-16"}
RUNS = ["full_ft", "full_ft_l2_sp", "lora_4", "lora_16", "oplora_16"]
RUN_LABEL = {"full_ft": "Full FT", "full_ft_l2_sp": "Full FT + L2-SP", "lora_4": "LoRA-4", "lora_16": "LoRA-16",
             "oplora_16": "OPLoRA-16"}
# one colour per run, in fixed palette order; the full-parameter runs are solid, the adapter runs dashed
RUN_COLOR = dict(zip(RUNS, CAT[:5]))
RUN_STYLE = {"full_ft": "-", "full_ft_l2_sp": "-", "lora_4": (0, (5, 2)), "lora_16": (0, (5, 2)), "oplora_16": (0, (5, 2))}
COND_RUN = {"full_ft_midpoint": "full_ft", "full_ft_final": "full_ft", "full_ft_l2_sp_final": "full_ft_l2_sp",
            "lora_4_final": "lora_4", "lora_16_final": "lora_16", "oplora_16_final": "oplora_16"}

# (key in results, display name, lm-eval task, metric key, filter, primary-score key)
BENCH = [("gsm8k", "GSM8K", "gsm8k_cot", "exact_match", "flexible-extract", "exact_match_flexible"),
         ("hellaswag", "HellaSwag", "hellaswag", "acc_norm", "none", "acc_norm"),
         ("boolq", "BoolQ", "boolq", "acc", "none", "acc"),
         ("arc_easy", "ARC-Easy", "arc_easy", "acc_norm", "none", "acc_norm")]

evals = {c: load_json(f"{T1}/eval/{c}.json") for c in CONDS}
have = [c for c in CONDS if evals[c]]
print("conditions evaluated:", [LABEL[c] for c in have])

# Offline re-analysis of the saved per-example outputs (extraction sensitivity, paired bootstrap, BoolQ bias).
# Produced on any machine that has outputs/task1/eval_samples/ with:  python scripts/run_task1.py --action offline_analysis
OA = load_json(f"{T1}/offline_analysis.json")


def score(c, bench_key, key):
    e = evals.get(c)
    v = None if not e else e["scores"].get(bench_key, {}).get(key)
    return None if v is None else 100.0 * float(v)


def stderr(c, b):
    """lm-eval standard error (in percentage points) of the primary score."""
    e = evals.get(c)
    if not e:
        return None
    _, _, task, metric, filt, _ = b
    v = e["lm_eval_results"].get(task, {}).get(f"{metric}_stderr,{filt}")
    return None if v in (None, "N/A") else 100.0 * float(v)


# %% [markdown]
# ## 1. Benchmark results
# Primary scores: GSM8K = flexible-extract exact match (zero-shot CoT, non-thinking chat template, greedy, 1,024 new
# tokens); HellaSwag / ARC-Easy = length-normalised accuracy; BoolQ = accuracy. Changes are **percentage points** from the shared baseline.

# %%
prim = pd.DataFrame({LABEL[c]: {b[1]: score(c, b[0], b[5]) for b in BENCH} for c in have}).T
if "Baseline" in prim.index:
    delta = prim - prim.loc["Baseline"]
    table = prim.copy().astype(object)
    for r in prim.index:
        for col in prim.columns:
            table.loc[r, col] = pct(prim.loc[r, col], 2) if r == "Baseline" else f"{pct(prim.loc[r, col], 2)} ({signed(delta.loc[r, col], 2)})"
    table.index.name = "Condition"
    save_table(table, "task1_benchmarks", "Task 1 primary benchmark scores (%) and change from baseline (percentage points).")
    display(table)

# %%
# standard errors, raw vs normalised accuracy and GSM8K answer-extraction / truncation diagnostics
se = pd.DataFrame({LABEL[c]: {b[1]: stderr(c, b) for b in BENCH} for c in have}).T.round(2)
se.index.name = "Condition (lm-eval stderr, pp)"
display(se)

raw = pd.DataFrame({LABEL[c]: {
    "HellaSwag acc": score(c, "hellaswag", "acc"), "HellaSwag acc_norm": score(c, "hellaswag", "acc_norm"),
    "BoolQ acc": score(c, "boolq", "acc"),
    "ARC-Easy acc": score(c, "arc_easy", "acc"), "ARC-Easy acc_norm": score(c, "arc_easy", "acc_norm")} for c in have}).T.round(2)
display(raw)

def _oa_gsm(c, key):
    """Value from the offline analysis (None when it has not been produced yet)."""
    return OA["gsm8k_extraction"][c][key] if OA and c in OA.get("gsm8k_extraction", {}) else None


# '[invalid] fallback' is what lm-eval itself reports as "no match" (always 0 here). The stock flexible regex also matches
# number-free strings such as "$$", so 'no-digit extraction' is the rate at which the answer was NOT actually read.
gsm = pd.DataFrame({LABEL[c]: {
    "flexible EM (%)": score(c, "gsm8k", "exact_match_flexible"), "strict EM (%)": score(c, "gsm8k", "exact_match_strict"),
    "truncated (%)": score(c, "gsm8k", "truncated_fraction"),
    "[invalid] fallback (%)": score(c, "gsm8k", "unextractable_fraction_flexible"),
    "no-digit extraction (%)": _oa_gsm(c, "no_digit_extraction_pct"),
    "n": (evals[c]["scores"]["gsm8k"].get("n") if evals[c] else None)} for c in have}).T
gsm = gsm.round(2)
gsm.index.name = "Condition"
save_table(gsm, "task1_gsm8k_format",
           "GSM8K accuracy vs answer-extraction and truncation rates. '[invalid] fallback': the harness found no match. "
           "'no-digit extraction': the stock regex returned a number-free string such as '$$', i.e. the answer was not read.")
display(gsm)

# %% [markdown]
# ### GSM8K: how much of the change is answer extraction?
# The stock `flexible-extract` filter (kept as the primary score) takes the last match of `(-?[$0-9.,]{2,})|(-?[0-9]+)`,
# which also matches a bare `$$`. Re-scoring with the **same rule but skipping digit-free matches** (`digit-required`), and
# with the last numeric `\boxed{}` first (`boxed-first`), shows how much of the stock-score change is extraction rather than
# arithmetic. The stock filter is replicated exactly (0 mismatches against the stored lm-eval output, see
# `results/task1/offline_analysis.json`). These are sensitivity checks next to the stock score, not replacements.

# %%
if OA:
    ge, gain = OA["gsm8k_extraction"], OA["gsm8k_gain_vs_baseline_pp"]
    sens = pd.DataFrame({LABEL[c]: {
        "stock EM (%)": ge[c]["stock_flexible_pct"], "digit-required EM (%)": ge[c]["digit_flexible_pct"],
        "boxed-first EM (%)": ge[c]["boxed_first_pct"],
        "Δ vs baseline, stock (pp)": gain.get(c, {}).get("stock_flexible_pct"),
        "Δ vs baseline, digit-required (pp)": gain.get(c, {}).get("digit_flexible_pct"),
        "Δ vs baseline, boxed-first (pp)": gain.get(c, {}).get("boxed_first_pct"),
        "boxed answer present (%)": ge[c]["boxed_present_pct"], "'answer is' phrase (%)": ge[c]["answer_is_phrase_pct"],
        "mean response tokens": ge[c]["mean_tokens"]} for c in CONDS if c in ge}).T
    sens = sens.astype(float).round(2).astype(object).where(sens.notna(), "–")
    sens.index.name = "Condition"
    save_table(sens, "task1_gsm8k_sensitivity",
               "GSM8K exact match under three extraction rules and the resulting change from the baseline (percentage points).")
    display(sens)

# %% [markdown]
# ### Heat table: change from baseline (pp)

# %%
if "Baseline" in prim.index and len(prim) > 1:
    d = delta.drop(index="Baseline")
    fig, ax = plt.subplots(figsize=(4.6, 0.38 * len(d) + 0.9))
    lim = max(1.0, float(np.nanmax(np.abs(d.values))))
    im = ax.imshow(d.values.astype(float), cmap=diverging_cmap(), vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(d.shape[1]), d.columns)
    ax.set_yticks(range(d.shape[0]), d.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.values[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:+.1f}", ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_title("Change from baseline (percentage points)", loc="left", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7)
    savefig(fig, "task1_delta_heat")
    plt.show()

# %% [markdown]
# ## 2. Controlled comparisons (final checkpoints)
# Differences in percentage points. When `results/task1/offline_analysis.json` exists, each cell is the difference with a
# **paired** bootstrap 95% interval (the same examples are scored under both conditions, so examples are resampled
# jointly; 2,000 resamples, seed 6304); `*` = the interval excludes 0. The unpaired fallback below (± = one standard error
# √(se₁²+se₂²) from lm-eval, `*` = |z| > 1.96) ignores that pairing and is too conservative; it is used only if the offline
# analysis has not been produced.

# %%
PAIR_COLS = {"gsm8k": "GSM8K", "gsm8k_digit_flexible": "GSM8K (digit-required)", "hellaswag": "HellaSwag",
             "boolq": "BoolQ", "arc_easy": "ARC-Easy"}


def _paired_cell(b):
    return f"{b['diff_pp']:+.2f} [{b['ci_lo_pp']:+.2f}, {b['ci_hi_pp']:+.2f}]" + (" *" if b["significant"] else "")


if OA:
    comp = pd.DataFrame({p["label"]: {PAIR_COLS[k]: _paired_cell(b) for k, b in p["benchmarks"].items()}
                         for p in OA["paired"] if p["kind"] == "controlled"}).T
    comp.index.name = "Comparison (pp, paired 95% CI)"
    save_table(comp, "task1_comparisons",
               "Controlled comparisons at the final checkpoints: difference in percentage points with a paired bootstrap "
               "95% interval (* = interval excludes 0).")
    display(comp)
    vsb = pd.DataFrame({LABEL[p["a"]]: {PAIR_COLS[k]: _paired_cell(b) for k, b in p["benchmarks"].items()}
                        for p in OA["paired"] if p["kind"] == "vs_baseline"}).T
    vsb.index.name = "Change from baseline (pp, paired 95% CI)"
    save_table(vsb, "task1_vs_baseline",
               "Change from the shared baseline in percentage points with a paired bootstrap 95% interval "
               "(* = interval excludes 0).")
    display(vsb)

PAIRS = [("Rank: LoRA-16 − LoRA-4", "lora_16_final", "lora_4_final"),
         ("Explicit regularisation: Full FT + L2-SP − Full FT", "full_ft_l2_sp_final", "full_ft_final"),
         ("Orthogonal protection: OPLoRA-16 − LoRA-16", "oplora_16_final", "lora_16_final"),
         ("Method: Full FT − LoRA-16", "full_ft_final", "lora_16_final"),
         ("Progress: Full FT final − midpoint", "full_ft_final", "full_ft_midpoint")]
rows = {}
for name, a, b in PAIRS:
    if not (evals.get(a) and evals.get(b)):
        continue
    row = {}
    for bm in BENCH:
        va, vb = score(a, bm[0], bm[5]), score(b, bm[0], bm[5])
        sa, sb = stderr(a, bm), stderr(b, bm)
        if va is None or vb is None:
            row[bm[1]] = "n/a"
            continue
        s = f"{va - vb:+.2f}"
        if sa is not None and sb is not None:
            z = (va - vb) / max(np.hypot(sa, sb), 1e-9)
            s += f" (1 SE {np.hypot(sa, sb):.2f})" + (" *" if abs(z) > 1.96 else "")
        row[bm[1]] = s
    rows[name] = row
if rows and not OA:   # unpaired fallback only; the paired table above is the one to report
    comp = pd.DataFrame(rows).T
    comp.index.name = "Comparison (pp, unpaired 1 SE)"
    save_table(comp, "task1_comparisons",
               "Controlled comparisons at the final checkpoints (percentage points; unpaired standard error of the difference; "
               "* = |z| > 1.96). Run the offline analysis for paired intervals.")
    display(comp)

# %% [markdown]
# ### BoolQ: accuracy vs yes/no prior
# BoolQ is a two-way choice, so accuracy mixes discrimination with a global yes/no bias. Shown per condition: the share of
# "yes" predictions (the gold rate is the reference), accuracy within each gold class, balanced accuracy, and the ROC-AUC of
# the yes-minus-no log-likelihood gap, which does not depend on where the decision threshold sits. Descriptive point
# estimates (no intervals).

# %%
if OA and OA.get("boolq_bias"):
    bb = OA["boolq_bias"]
    # (no '|' in column names: it would split the markdown table)
    bt = pd.DataFrame({LABEL[c]: {
        "accuracy (%)": f"{bb[c]['accuracy_pct']:.2f}", "predicted yes (%)": f"{bb[c]['pred_yes_pct']:.1f}",
        "gold yes (%)": f"{bb[c]['gold_yes_pct']:.1f}",
        "acc on gold-yes (%)": f"{bb[c]['acc_gold_yes_pct']:.1f}", "acc on gold-no (%)": f"{bb[c]['acc_gold_no_pct']:.1f}",
        "balanced acc (%)": f"{bb[c]['balanced_accuracy_pct']:.1f}", "AUC (yes−no log-lik gap)": f"{bb[c]['auc_yes_vs_no_gap']:.3f}",
        "mean log-lik gap (yes−no)": f"{bb[c]['mean_loglik_gap_yes_minus_no']:+.3f}"} for c in CONDS if c in bb}).T
    bt.index.name = "Condition"
    save_table(bt, "task1_boolq_bias", "BoolQ validation (3,270): yes/no prediction bias versus threshold-free discrimination (AUC).")
    display(bt)

# %% [markdown]
# ## 3. Learning curves and the L2-SP penalty
# SFT loss is the token-averaged cross-entropy of the effective batch (32 sequences). For the L2-SP run the *SFT loss* is
# compared with the other runs; the penalty term and the total objective are shown separately.

# %%
hist = {v: load_csv(f"{T1}/{v}/training_history.csv") for v in RUNS}
summ = {v: load_json(f"{T1}/{v}/summary.json") for v in RUNS}
ok_runs = [v for v in RUNS if hist[v] is not None]

if ok_runs:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
    # draw order: the thick Full FT line underneath, the (nearly coincident) L2-SP line thinner on top, so both stay visible
    LW = {"full_ft": 2.6, "full_ft_l2_sp": 1.1}
    ax = axes[0]
    ax_zoom = axes[1]
    zoom_vals = []
    for v in ok_runs:
        h = hist[v]
        y = smooth(h["sft_loss"], 25)
        kw = dict(color=RUN_COLOR[v], ls=RUN_STYLE[v], lw=LW.get(v, 1.4), label=RUN_LABEL[v], zorder=3 if v == "full_ft_l2_sp" else 2)
        ax.plot(h["update"], y, **kw)
        ax_zoom.plot(h["update"], y, **kw)
        zoom_vals.append(y[h["update"].to_numpy() >= 300])
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("SFT loss (25-update moving mean)")
    ax.set_ylim(top=min(ax.get_ylim()[1], 1.0))
    panel_title(ax, "SFT loss")
    ax.legend(loc="upper right", ncol=1, fontsize=6.8)
    zv = np.concatenate(zoom_vals) if zoom_vals else np.array([0.0, 1.0])
    ax_zoom.set_xlim(300, 1260)
    ax_zoom.set_ylim(zv.min() - 0.005, zv.max() + 0.005)
    ax_zoom.set_xlabel("optimizer update")
    panel_title(ax_zoom, "Zoom: updates ≥ 300")
    ax = axes[2]
    for v in ok_runs:
        h = hist[v]
        ax.plot(h["update"], h["lr"], color=RUN_COLOR[v], ls=RUN_STYLE[v], lw=LW.get(v, 1.4), label=RUN_LABEL[v])
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 4e-4)   # the cosine schedule ends at ~0; clip so the peaks stay readable
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("learning rate")
    panel_title(ax, "Learning-rate schedule")
    fig.tight_layout()
    savefig(fig, "task1_learning_curves")
    plt.show()

    tail = {RUN_LABEL[v]: {"final SFT loss (mean of last 50 updates)": float(hist[v]["sft_loss"].tail(50).mean()),
                           "first 50 updates": float(hist[v]["sft_loss"].head(50).mean()),
                           "skipped updates (fp16 overflow)": (summ[v] or {}).get("skipped_updates")} for v in ok_runs}
    tail = pd.DataFrame(tail).T.round(4)
    tail.index.name = "Run"
    save_table(tail, "task1_loss_summary", "SFT loss summary per run.")
    display(tail)

# %%
if hist["full_ft_l2_sp"] is not None:
    h = hist["full_ft_l2_sp"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ax = axes[0]
    ax.plot(h["update"], smooth(h["sft_loss"], 25), color=CAT[1], label="SFT loss")
    if "total_loss" in h:
        ax.plot(h["update"], smooth(h["total_loss"], 25), color=CAT[1], ls=(0, (5, 2)), label="total = SFT + λΩ")
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("loss")
    panel_title(ax, "L2-SP run: SFT loss vs total objective")
    ax.legend()
    ax = axes[1]
    if "omega" in h:
        ax.plot(h["update"], h["omega"], color=CAT[1], label="Ω = ½‖θ−θ₀‖²")
        ax.set_yscale("log")
        ax.set_ylabel("Ω (log scale)")
        ax2 = None
    ax.set_xlabel("optimizer update")
    panel_title(ax, "Starting-point penalty Ω")
    fig.tight_layout()
    savefig(fig, "task1_l2sp_objective")
    plt.show()

# displacement ‖θ−θ0‖₂ at the evaluation points
disp = {}
for v in ("full_ft", "full_ft_l2_sp"):
    h = hist[v]
    if h is not None and "displacement" in h:
        s = h.dropna(subset=["displacement"])
        disp[RUN_LABEL[v]] = {int(u): float(d) for u, d in zip(s["update"], s["displacement"])}
if disp:
    dtab = pd.DataFrame(disp)
    dtab.index.name = "update"
    if {"Full FT", "Full FT + L2-SP"} <= set(dtab.columns):
        dtab["L2-SP / Full FT"] = (dtab["Full FT + L2-SP"] / dtab["Full FT"]).round(3)
    save_table(dtab.round(4), "task1_displacement", "Full-model displacement ‖θ−θ₀‖₂ at the evaluation checkpoints.")
    display(dtab.round(4))

# %% [markdown]
# ## 4. Resources
# Training time excludes checkpoint writing (reported separately). Peak memory is the maximum allocated by PyTorch during
# training. Checkpoint sizes: Full FT stores FP32 weights (+ optimizer state when kept); LoRA/OPLoRA store adapters only.

# %%
res = {}
for v in RUNS:
    s = summ[v]
    if not s:
        continue
    ck = s.get("checkpoints") or []
    last = ck[-1] if ck else {}
    res[RUN_LABEL[v]] = {
        "trainable params (M)": s["trainable_parameters"] / 1e6,
        "trainable (%)": 100 * s["trainable_parameters"] / max(s["total_parameters"], 1),
        "train time (min)": s["train_time_excl_checkpoints_sec"] / 60,
        "setup (s)": s.get("setup_time_sec"),
        "checkpoint writes (s, total)": s["checkpoint_time_sec"],
        "peak GPU mem (GB)": s["peak_memory_mb"] / 1024,
        "weights file (MB)": last.get("weights_mb"),
        "optimizer file (MB)": last.get("optimizer_mb"),
        "OPLoRA projected export (MB)": last.get("projected_factors_mb"),
        "checkpoints saved": len(ck),
    }
if res:
    rt = pd.DataFrame(res).T
    rt.index.name = "Run"
    save_table(rt.round(2), "task1_resources", "Resource comparison of the five training runs.")
    display(rt.round(2))
    extras = {RUN_LABEL[v]: {"oplora": summ[v].get("oplora"), "l2sp": summ[v].get("l2sp")} for v in RUNS
              if summ[v] and (summ[v].get("oplora") or summ[v].get("l2sp"))}
    for k, val in extras.items():
        print(k, "method-specific overhead / statistics:", {a: b for a, b in val.items() if b})

# %% [markdown]
# ## 5. Diagnostics: subspace interference, update magnitude, activation corrections
# ρ_k = share of ‖ΔW‖² inside span(U_k) (top-k singular directions of W₀, k = 128); η = ‖ΔW‖_F/‖W₀‖_F;
# R(z) = ‖ΔW z‖/‖W₀ z‖ on cached down_proj inputs (32 GSM8K "math" prompts vs 32 HellaSwag "non-math" prompts).
# These are *local* measurements on selected layers: they do not establish why behaviour changed.

# %%
# Qwen3-0.6B mlp.down_proj weight is (hidden 1024) x (intermediate 3072): a random update puts a fraction k/d_out of its squared
# norm inside a k-dimensional output subspace (k/d_in on the input side), the reference level for ρ_k.
D_OUT_DOWN, D_IN_DOWN = 1024, 3072
dpath = next(RESULTS.glob(f"{T1}/**/diagnostics_summary.json"), None)
diag = load_json(str(dpath.relative_to(RESULTS)).replace("\\", "/")) if dpath else None
if dpath is None:
    print("[not available yet] task1 diagnostics_summary.json")
if diag:
    rows = []
    for cond, per_layer in diag["conditions"].items():
        for lk, m in per_layer.items():
            rows.append({"condition": cond, "layer": int(lk.split("_")[1]), **m})
    D = pd.DataFrame(rows)
    D["run"] = D["condition"].map(COND_RUN)
    conds_d = [c for c in CONDS if c in set(D["condition"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, (col, ttl, logy) in zip(axes[:2], [("rho_k", "ρ_k  (output side)", True), ("eta", "η = ‖ΔW‖/‖W₀‖", True)]):
        for c in conds_d:
            g = D[D["condition"] == c].sort_values("layer")
            ls = "-" if c.startswith("full") else (0, (5, 2))
            ax.plot(g["layer"], g[col].clip(lower=1e-10), color=RUN_COLOR[COND_RUN[c]], ls=ls,
                    marker="o" if c == "full_ft_midpoint" else None, label=LABEL[c])
        ax.set_yscale("log" if logy else "linear")
        ax.set_xlabel("layer")
        panel_title(ax, ttl)
        ax.set_xticks(sorted(D["layer"].unique()))
    axes[0].legend(fontsize=6.5)
    chance_out = diag.get("k", 128) / D_OUT_DOWN            # reference level of ρ_k for an update unrelated to U_k
    _lo, _hi = axes[0].get_ylim()
    axes[0].set_ylim(_lo, max(_hi, chance_out * 6))          # headroom so the reference label stays inside the axes
    axes[0].axhline(chance_out, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=0)
    axes[0].annotate(f"chance k/d_out = {chance_out:.3f}", (D["layer"].max(), chance_out), xytext=(-2, 4),
                     textcoords="offset points", ha="right", fontsize=6.5, color=MUTED)
    ax = axes[2]
    agg = D.groupby("condition")[["math_mean_r", "non_math_mean_r"]].mean().reindex(conds_d)
    for i, c in enumerate(conds_d):
        col = RUN_COLOR[COND_RUN[c]]
        ax.plot([agg.loc[c, "non_math_mean_r"], agg.loc[c, "math_mean_r"]], [i, i], color=AXIS, lw=1.2, zorder=1)
        ax.scatter(agg.loc[c, "math_mean_r"], i, color=col, s=28, zorder=3, label="math prompts" if i == 0 else None)
        ax.scatter(agg.loc[c, "non_math_mean_r"], i, facecolor="white", edgecolor=col, s=28, zorder=3,
                   label="non-math prompts" if i == 0 else None)
    ax.set_yticks(range(len(conds_d)), [LABEL[c] for c in conds_d])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("mean R(z), layer average")
    panel_title(ax, "Activation correction R(z)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2, fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "task1_diagnostics")
    plt.show()

    summ_d = D.groupby("condition")[["rho_k", "rho_k_input_side", "eta", "math_mean_r", "non_math_mean_r", "math_to_non_math_ratio"]].mean()
    summ_d = summ_d.reindex(conds_d)
    summ_d.index = [LABEL[c] for c in conds_d]
    summ_d.columns = ["ρ_k (out)", "ρ_k (in)", "η", "R math", "R non-math", "R math / non-math"]
    summ_d.index.name = "Condition (layer mean)"
    save_table(summ_d.map(lambda v: f"{v:.3g}"), "task1_diagnostics", "Layer-averaged weight-space and activation diagnostics.")
    display(summ_d.map(lambda v: f"{v:.3g}"))

    # The layer mean hides heterogeneity (e.g. R(math)/R(non-math) < 1 in layer 27 for every run), so the per-layer values are
    # saved as well. Note the math probes all end on the same chat-template token, the non-math probes on varied content tokens.
    by_layer = D[D["condition"].isin(conds_d)].copy()
    by_layer["_o"] = by_layer["condition"].map({c: i for i, c in enumerate(CONDS)})
    by_layer = by_layer.sort_values(["_o", "layer"])
    by_layer["condition"] = by_layer["condition"].map(LABEL)
    by_layer = by_layer.set_index(["condition", "layer"])[
        ["rho_k", "rho_k_input_side", "eta", "math_mean_r", "non_math_mean_r", "math_to_non_math_ratio"]]
    by_layer.columns = ["ρ_k (out)", "ρ_k (in)", "η", "R math", "R non-math", "R math / non-math"]
    by_layer.index.names = ["Condition", "layer"]
    save_table(by_layer.map(lambda v: f"{v:.3g}"), "task1_diagnostics_by_layer",
               f"Per-layer diagnostics on down_proj. For a random update the expected ρ_k is k/d_out = {diag.get('k', 128) / D_OUT_DOWN:.3f} "
               f"(output side) and k/d_in = {diag.get('k', 128) / D_IN_DOWN:.3f} (input side).")
    display(by_layer.map(lambda v: f"{v:.3g}"))

    # numerical verification of OPLoRA's constraint (kept apart from benchmark evidence)
    chk = D[D["condition"].isin(["lora_16_final", "oplora_16_final"])].groupby("condition")[["rho_k", "rho_k_input_side"]].max()
    chk.index = [LABEL[c] for c in chk.index]
    chk.index.name = "max over layers"
    print("OPLoRA constraint check (should be ~0 for OPLoRA-16; a plain LoRA-16 is the reference):")
    display(chk.map(lambda v: f"{v:.2e}"))

# %% [markdown]
# ## 6. Qualitative study (fixed 42-response subset)
# 2 plain questions × 3 formats (plain / JSON / template) × 7 conditions. Automatic flags are a first pass only —
# `inappropriate_content` and `manual_notes` are filled in by hand in `annotation_subset_42.json`.

# %%
sub = load_json(f"{T1}/qualitative/annotation_subset_42.json")
if sub:
    Q = pd.DataFrame(sub)
    Q["condition"] = pd.Categorical(Q["condition"], [c for c in CONDS if c in set(Q["condition"])], ordered=True)
    fmt_df = Q[Q["format"].isin(["json", "template"])]
    summary = pd.DataFrame({
        "format compliance, JSON/template (%)": 100 * fmt_df.groupby("condition", observed=True)["format_compliance"].mean(),
        "repetition flagged (of 6)": Q.groupby("condition", observed=True)["repetition"].sum(),
        "math framing flagged (of 6)": Q.groupby("condition", observed=True)["unnecessary_math_framing"].sum(),
        "truncated (of 6)": Q.groupby("condition", observed=True)["truncated"].sum(),
        "manually annotated": Q.groupby("condition", observed=True)["inappropriate_content"].apply(lambda s: int(s.notna().sum()))})
    summary.index = [LABEL[c] for c in summary.index]
    summary.index.name = "Condition"
    save_table(summary.round(1), "task1_qualitative_summary", "Automatic first-pass flags over the 42-response subset.")
    display(summary.round(1))

    # The 42-response subset covers only 2 of the 8 questions (6 responses per condition). The same signals over all
    # 24 saved responses per condition (offline_analysis.json) show how widespread the math-style intrusion is.
    QI = (OA or {}).get("qualitative_math_intrusion", {})
    if QI:
        rows_q = {}
        for c in CONDS:
            if c in QI:
                v = QI[c]
                fm = v["by_format"]
                rows_q[LABEL[c]] = {
                    "math-style artefact (of 24)": v["n_with_artefact"],
                    "plain (of 8)": fm.get("plain", {}).get("flagged", 0),
                    "JSON (of 8)": fm.get("json", {}).get("flagged", 0),
                    "template (of 8)": fm.get("template", {}).get("flagged", 0),
                    "'The answer is:'": v["by_signal"]["answer_is"],
                    "'#### n'": v["by_signal"]["gsm8k_marker"],
                    "other (boxed / LaTeX / <<>>)": v["by_signal"]["boxed"] + v["by_signal"]["latex"] + v["by_signal"]["calc_annotation"],
                    "math vocabulary only": v["n_flagged"] - v["n_with_artefact"]}
        qi_df = pd.DataFrame(rows_q).T
        qi_df.index.name = "Condition"
        save_table(qi_df, "task1_qualitative_math_intrusion",
                   "Responses (of 24 per condition, all prompts and formats) containing MetaMathQA / GSM8K training artefacts. "
                   "Automatic first-pass flags (qualitative_flags.py), not a substitute for the manual annotation. "
                   "The artefact columns count signal occurrences (a response can carry two), so they can exceed the first column.")
        display(qi_df)

    lines = ["# Aligned excerpts (same question and format across conditions)\n"]
    for uq in Q["underlying_question"].unique():
        for fm in ("plain", "json", "template"):
            g = Q[(Q["underlying_question"] == uq) & (Q["format"] == fm)].sort_values("condition")
            lines.append(f"\n## {uq} — {fm}\n")
            for r in g.itertuples():
                flag = ("" if r.format_compliance else " [format violation]") + (" [truncated]" if r.truncated else "")
                lines.append(f"- **{LABEL[r.condition]}**{flag}: {show_excerpt(r.text, 260)}")
    (TABLES / "task1_qualitative_aligned.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved tables/task1_qualitative_aligned.md  (pick the excerpts you quote from here)")

    # inspect one aligned set
    uq = Q["underlying_question"].unique()[0]
    for fm in ("plain", "json", "template"):
        print(f"\n=== {uq} / {fm} ===")
        for r in Q[(Q["underlying_question"] == uq) & (Q["format"] == fm)].sort_values("condition").itertuples():
            print(f"[{LABEL[r.condition]:16s}] {show_excerpt(r.text, 200)}")
