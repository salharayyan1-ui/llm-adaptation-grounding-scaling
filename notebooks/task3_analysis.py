# %% [markdown]
# # Task 3 — Test-time compute: analysis
#
# Reads `results/task3/…` (written by `scripts/run_task3.py`) and produces the evidence required for Task 3:
#
# 1. sampling & selection — accuracy and oracle-coverage curves for both datasets, a compact method comparison, and the
#    decomposition of *sampling* failures (no candidate is correct) vs *selection* failures (a correct candidate exists but is not chosen)
# 2. signal analysis — length / correctness associations, confidence–vision relationships, token-wise vs trajectory-level fusion
# 3. live inference — accuracy, actual input/output tokens, runtime, truncation/extraction rates, revision transitions
# 4. adaptive inference — development-selected thresholds, held-out accuracy–cost results, stopping distributions, fixed-budget comparison
# 5. qualitative examples
#
# All intervals are the question-level bootstrap intervals stored in the result files. Missing files only print `[not available yet]`.

# %%
from nb_utils import *

T3 = "task3"
DATASETS = {"math": "MATH-500 · Qwen2.5-3B-Instruct", "mathvista": "MathVista · Qwen2.5-VL-3B-Instruct"}
NS = ["1", "2", "4", "8", "16", "32"]
SHORT = {"math": "MATH", "mathvista": "MathVista"}
KNOWN = ["first_ans", "self_consistency", "highest_logp", "highest_self_certainty", "self_certainty_rwc"]
NICE = {"first_ans": "First answer", "self_consistency": "Self-consistency", "highest_logp": "Highest log-prob",
        "highest_self_certainty": "Highest self-certainty", "self_certainty_rwc": "Self-certainty RWC",
        "highest_vision": "Highest vision", "vision_rwc": "Vision RWC",
        "highest_visconf": "Highest VisConf", "visconf_rwc": "VisConf RWC"}
# Colour follows the method (never its rank). The first-answer baseline is neutral grey so that the (up to) eight
# selection methods of the two datasets get the eight palette slots without any cycling; each method also has its own marker.
METHOD_ORDER = ["self_consistency", "highest_logp", "highest_self_certainty", "self_certainty_rwc",
                "highest_vision", "vision_rwc", "highest_visconf", "visconf_rwc"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def method_color(m):
    if m == "first_ans":
        return MUTED
    if m not in METHOD_ORDER:
        METHOD_ORDER.append(m)
    return CAT[METHOD_ORDER.index(m) % len(CAT)]


def method_marker(m):
    return "." if m == "first_ans" else MARKERS[METHOD_ORDER.index(m) % len(MARKERS)] if m in METHOD_ORDER else "o"


def nice(m):
    return NICE.get(m, m.replace("_", " "))


def methods_of(summary):
    ms = list(summary[NS[0]].keys()) if NS[0] in summary else []
    return [m for m in KNOWN if m in ms] + [m for m in ms if m not in KNOWN]


opp = {d: load_json(f"{T3}/{d}/step1_opportunity.json") for d in DATASETS}
scor = {d: load_json(f"{T3}/{d}/step2_scoring.json") for d in DATASETS}
sig = {d: load_json(f"{T3}/{d}/step3_signals.json") for d in DATASETS}
ada = {d: load_json(f"{T3}/{d}/step5_adaptive.json") for d in DATASETS}
splits = {d: load_json(f"{T3}/{d}/splits.json") for d in DATASETS}
live = load_json(f"{T3}/live_math/live_summary.json")
live_gen = load_json(f"{T3}/live_math/live_generations.json")
info = load_json(f"{T3}/run_info_all.json")
if info:
    print("scorer:", info.get("scorer_version"), "| synthetic data:", info.get("synthetic"))
    if info.get("synthetic"):
        print("!!! these results come from FAKE data (smoke test) — do not report them !!!")
for d, s in splits.items():
    if s:
        print(DATASETS[d], {k: len(v) for k, v in s.items()})

# %% [markdown]
# ## 1. Sampling and selection

# %%
avail = [d for d in DATASETS if opp[d]]
if avail:
    fig, axes = plt.subplots(1, len(avail), figsize=(3.6 * len(avail), 2.7), squeeze=False)
    for ax, d in zip(axes[0], avail):
        x = np.array([int(n) for n in NS])
        for key, lab, col, ls in (("first_ans_accuracy", "Accuracy of first answer", method_color("first_ans"), (0, (3, 2))),
                                  ("oracle_coverage", "Oracle coverage (≥1 correct candidate)", INK, (0, (1, 2)))):
            m = np.array([opp[d][n][key]["mean"] for n in NS]) * 100
            lo = np.array([opp[d][n][key]["ci_lower"] for n in NS]) * 100
            hi = np.array([opp[d][n][key]["ci_upper"] for n in NS]) * 100
            ax.plot(x, m, color=col, ls=ls, marker="o", ms=3.5, label=lab)
            ax.fill_between(x, lo, hi, color=col, alpha=0.14, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(v) for v in x])
        ax.set_xlabel("rollouts per question (N)")
        ax.set_ylabel("%")
        panel_title(ax, DATASETS[d])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=6.8)
    fig.tight_layout()
    savefig(fig, "task3_opportunity")
    plt.show()
    rows = {}
    for d in avail:
        for n in NS:
            o = opp[d][n]
            rows[(d, int(n))] = {"first-answer acc. % [CI]": fmt_ci(*ci_of(o["first_ans_accuracy"])),
                                 "oracle coverage % [CI]": fmt_ci(*ci_of(o["oracle_coverage"])),
                                 "distinct answer groups": f"{o['distinct_groups']['mean']:.2f}",
                                 "total candidate tokens": f"{o['total_candidate_tokens']['mean']:.0f}"}
    t = pd.DataFrame(rows).T
    t.index = pd.MultiIndex.from_tuples(t.index, names=["Dataset", "N"])
    save_table(t.reset_index(), "task3_opportunity", "Repeated-sampling opportunity: accuracy and oracle coverage vs N.", index=False)
    display(t)

# %%
avail = [d for d in DATASETS if scor[d]]
if avail:
    fig, axes = plt.subplots(2, len(avail), figsize=(3.8 * len(avail), 5.2), squeeze=False)
    seen = {}
    for j, d in enumerate(avail):
        S, TOK = scor[d]["summary"], scor[d]["mean_generated_tokens"]
        ms = methods_of(S)
        x = np.array([int(n) for n in NS])
        ax = axes[0, j]
        for m in ms:
            y = np.array([S[n][m]["mean"] for n in NS]) * 100
            lo = np.array([S[n][m]["ci_lower"] for n in NS]) * 100
            hi = np.array([S[n][m]["ci_upper"] for n in NS]) * 100
            ln, = ax.plot(x, y, color=method_color(m), marker=method_marker(m), ms=3.2, label=nice(m),
                          ls="-" if m != "first_ans" else (0, (3, 2)))
            seen.setdefault(m, ln)
            ax.fill_between(x, lo, hi, color=method_color(m), alpha=0.09, linewidth=0)
        oc = np.array([scor[d]["oracle_coverage"][n]["mean"] for n in NS]) * 100
        ln, = ax.plot(x, oc, color=INK, ls=(0, (1, 2)), lw=1.3, label="Oracle coverage")
        seen.setdefault("oracle", ln)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(v) for v in x])
        ax.set_xlabel("rollouts per question (N)")
        ax.set_ylabel("accuracy (%)")
        panel_title(ax, DATASETS[d])
        ax = axes[1, j]
        for m in ms:
            ax.plot([TOK[n][m] for n in NS], [S[n][m]["mean"] * 100 for n in NS], color=method_color(m), marker=method_marker(m),
                    ms=3.2, ls="-" if m != "first_ans" else (0, (3, 2)))
        ax.set_xscale("log")
        ax.set_xlabel("mean generated tokens / question")
        ax.set_ylabel("accuracy (%)")
        panel_title(ax, "Accuracy vs generation cost")
    fig.legend(list(seen.values()), [nice(m) if m != "oracle" else "Oracle coverage" for m in seen], loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, -0.01), fontsize=7)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    savefig(fig, "task3_methods")
    plt.show()

    # compact comparison at N = 8 and 32
    rows = {}
    for d in avail:
        S, TOK = scor[d]["summary"], scor[d]["mean_generated_tokens"]
        for n in ("8", "32"):
            for m in methods_of(S):
                rows[(d, int(n), nice(m))] = {"accuracy % [95% CI]": fmt_ci(*ci_of(S[n][m])), "mean tokens": f"{TOK[n][m]:.0f}",
                                              "oracle coverage %": pct(100 * scor[d]["oracle_coverage"][n]["mean"])}
    t = pd.DataFrame(rows).T
    t.index = pd.MultiIndex.from_tuples(t.index, names=["Dataset", "N", "Method"])
    save_table(t.reset_index(), "task3_methods", "Selection methods at N=8 and N=32.", index=False)
    display(t)

# %% [markdown]
# ### Sampling vs selection failures
# accuracy = 1 − sampling failure − selection failure. *Sampling failure*: none of the N candidates is correct (selection
# cannot help). *Selection failure*: a correct candidate exists but the method chose another answer.

# %%
avail = [d for d in DATASETS if scor[d]]
if avail:
    fig, axes = plt.subplots(1, len(avail), figsize=(3.7 * len(avail), 2.7), squeeze=False)
    for ax, d in zip(axes[0], avail):
        fd = scor[d]["failure_decomposition"]["32"]
        ms = methods_of(scor[d]["summary"])
        ms = [m for m in ms if m in fd]
        corr = np.array([1 - fd[m]["sampling_failure"] - fd[m]["selection_failure"] for m in ms]) * 100
        sel = np.array([fd[m]["selection_failure"] for m in ms]) * 100
        sam = np.array([fd[m]["sampling_failure"] for m in ms]) * 100
        y = np.arange(len(ms))
        ax.barh(y, corr, color=CAT[0], height=0.55, edgecolor="white", linewidth=1.2, label="correct")
        ax.barh(y, sel, left=corr, color=CAT[1], height=0.55, edgecolor="white", linewidth=1.2, label="selection failure")
        ax.barh(y, sam, left=corr + sel, color="#c9c8c2", height=0.55, edgecolor="white", linewidth=1.2, label="sampling failure")
        ax.set_yticks(y, [nice(m) for m in ms])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of questions (N = 32)")
        panel_title(ax, DATASETS[d])
        ax.grid(axis="y", visible=False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    savefig(fig, "task3_failure_decomposition")
    plt.show()

    rows = {}
    for d in avail:
        for n in ("8", "32"):
            for key, v in scor[d]["paired_differences"][n].items():
                rows[(d, int(n), key)] = {"Δ accuracy, pp [95% CI]": f"{100*v['mean_diff']:+.1f} [{100*v['ci_lower']:+.1f}, {100*v['ci_upper']:+.1f}]"}
    t = pd.DataFrame(rows).T
    t.index = pd.MultiIndex.from_tuples(t.index, names=["Dataset", "N", "Paired comparison"])
    save_table(t.reset_index(), "task3_paired", "Paired question-level differences between methods (same bootstrap resamples).", index=False)
    display(t)

# %% [markdown]
# ## 2. Signal analysis
# Within-question Spearman associations (averaged over questions) of correctness with length / log-probability /
# self-certainty; quartile contrasts; and, for MathVista, the confidence–vision relationship and the fusion ablation
# (trajectory-level VisConf vs token-wise fusion).

# %%
for d in DATASETS:
    s = sig[d]
    if not s:
        continue
    a = s.get("association_summary") or {}
    if a:
        names = list(a)
        fig, ax = plt.subplots(figsize=(4.2, 0.36 * len(names) + 0.9))
        for i, k in enumerate(names):
            ax.plot([a[k]["ci_lower"], a[k]["ci_upper"]], [i, i], color=CAT[0], lw=1.6)
            ax.scatter(a[k]["mean_spearman"], i, color=CAT[0], s=22, zorder=3)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.set_xlabel("mean within-question Spearman ρ  (95% CI)")
        panel_title(ax, f"{DATASETS[d]}: signal associations")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        savefig(fig, f"task3_signals_{d}")
        plt.show()
    q = s.get("quartile_summary") or {}
    if q:
        t = pd.DataFrame({k: {"Δ correct rate, top − bottom quartile [CI]": f"{v['mean_diff']:+.3f} [{v['ci_lower']:+.3f}, {v['ci_upper']:+.3f}]"} for k, v in q.items()}).T
        t.index.name = f"{d}: quartile contrast"
        display(t)
    c = s.get("control_summary") or {}
    if c:
        print(f"{d}: shortest-candidate control accuracy = {100*c['shortest_baseline_acc']:.1f}% [{100*c['ci_lower']:.1f}, {100*c['ci_upper']:.1f}]")

# %%
sv = sig.get("mathvista")
if sv:
    cs = sv.get("correlations_summary") or {}
    if cs:
        print(f"confidence–vision correlation ({cs.get('unit')}): mean {cs['mean_correlation']:+.4f} "
              f"[{cs['ci_lower']:+.4f}, {cs['ci_upper']:+.4f}], std {cs['std_correlation']:.3f}; "
              f"max covariance-identity error {cs['max_covariance_identity_error']:.2e}")
    ab = sv.get("ablation_summary") or {}
    if ab:
        names = list(ab)
        fig, ax = plt.subplots(figsize=(4.4, 0.4 * len(names) + 0.9))
        for i, k in enumerate(names):
            m, lo, hi = ci_of(ab[k])
            ax.plot([100 * lo, 100 * hi], [i, i], color=CAT[2], lw=1.6)
            ax.scatter(100 * m, i, color=CAT[2], s=22, zorder=3)
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.set_xlabel("accuracy % (95% CI)")
        panel_title(ax, "MathVista: VisConf fusion ablation")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        savefig(fig, "task3_visconf_ablation")
        plt.show()
        t = pd.DataFrame({k: {"accuracy % [95% CI]": fmt_ci(*ci_of(v))} for k, v in ab.items()}).T
        save_table(t, "task3_visconf_ablation", "Trajectory-level vs token-wise VisConf (score and RWC).")
        display(t)
    tr = sv.get("illustrative_trajectories") or []
    if tr:
        print("illustrative trajectories (selection rule is stored with each record):")
        display(pd.DataFrame(tr).head(8))

# %% [markdown]
# ## 3. Live inference (50-question subset, Qwen2.5-3B-Instruct)

# %%
if live:
    strat = ["single", "parallel_sc", "refinement"]
    t = pd.DataFrame({
        "accuracy %": {s: 100 * live["accuracy"][s] for s in strat},
        "mean input tokens": {s: live["mean_input_tokens"][s] for s in strat},
        "mean output tokens": {s: live["mean_output_tokens"][s] for s in strat},
        "mean runtime (s)": {s: live["mean_runtime_sec"][s] for s in strat},
        "truncated %": {s: 100 * live["truncation_rate"][s] for s in strat},
        "answer extracted %": {s: 100 * live["extraction_rate"][s] for s in strat},
        "peak GPU mem (GB)": {s: live["max_peak_memory_mb"][s] / 1024 for s in strat}}).round(2)
    t.index.name = f"Strategy (n = {live['count']})"
    save_table(t, "task3_live", "Live generation: three strategies on the same questions.")
    display(t)
    tc = live.get("transitions_count") or {}
    if tc:
        order = ["wrong_to_correct", "correct_to_wrong", "correct_to_correct", "wrong_to_wrong"]
        print("revision transitions (refinement vs its initial answer):", {k: tc.get(k) for k in order},
              "| initial-response truncation rate:", live.get("initial_response_truncation_rate"))
if live_gen:
    ex = [g for g in live_gen if g.get("revision_transition") in ("wrong_to_correct", "correct_to_wrong")][:4]
    for g in ex:
        print(f"\n[{g['revision_transition']}] {g['sample_id']} gold = {g['gold_answer']}")
        print("  question:", show_excerpt(g["question"], 160))
        print("  single   :", show_excerpt(g["strategy_single"]["text"], 140))
        print("  parallel : chosen", g["parallel_chosen_answer"])
        print("  refine   :", show_excerpt(g["strategy_refine"]["text"], 140))

# %% [markdown]
# ## 4. Adaptive inference
# Thresholds are selected on the development questions only (accuracy ≥ target × the fixed-32 self-consistency accuracy at the
# lowest token cost) and frozen before the held-out evaluation. If no setting meets the target the fixed-32 fallback is used.

# %%
avail = [d for d in DATASETS if ada[d]]
if avail:
    fig, axes = plt.subplots(1, len(avail) * 2, figsize=(3.4 * len(avail) * 2, 2.8), squeeze=False)
    for k, d in enumerate(avail):
        a = ada[d]
        ax = axes[0, 2 * k]
        ds = a["dev_search"]
        for key, col, lab in (("agreement", CAT[0], "agreement"), ("quality_aware", CAT[1], "quality-aware")):
            g = pd.DataFrame(ds[key]["grid"])
            ax.scatter(g["mean_tokens"], 100 * g["mean_accuracy"], s=14, color=col, alpha=0.8, label=lab)
            bp = ds[key]["best_params"]
            ax.scatter([bp["mean_tokens"]], [100 * bp["mean_accuracy"]], s=60, facecolor="none", edgecolor=INK, lw=1.2, zorder=4)
        b = ds["baseline_sc32"]
        ax.scatter([b["mean_tokens"]], [100 * b["mean_accuracy"]], marker="s", s=28, color=INK2, label="fixed SC-32")
        ax.axhline(100 * b["target_accuracy"], color=AXIS, ls=(0, (3, 3)), lw=0.8)
        ax.set_xlabel("mean tokens / question (dev)")
        ax.set_ylabel("dev accuracy (%)")
        panel_title(ax, f"{SHORT.get(d, d)}: dev threshold search (circled = selected)")
        ax.legend(fontsize=6.3)
        ax = axes[0, 2 * k + 1]
        fx = a["eval_fixed_budget_self_consistency"]
        xs = [fx[n]["mean_tokens"] for n in fx]
        ys = [100 * fx[n]["accuracy"]["mean"] for n in fx]
        ax.plot(xs, ys, color=INK2, marker="s", ms=3.5, label="fixed-N self-consistency", ls=(0, (4, 2)))
        for n, x_, y_ in zip(fx, xs, ys):
            ax.annotate(f"N={n}", (x_, y_), textcoords="offset points", xytext=(4, -9), fontsize=6.5, color=MUTED)
        for key, col, lab in (("eval_agreement", CAT[0], "adaptive: agreement"), ("eval_quality_aware", CAT[1], "adaptive: quality-aware")):
            e = a[key]
            m, lo, hi = ci_of(e["accuracy"])
            ax.errorbar(e["mean_tokens"], 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="o", color=col, capsize=2.5, label=lab)
        ax.set_xlabel("mean tokens / question (held-out)")
        ax.set_ylabel("held-out accuracy (%)")
        panel_title(ax, f"{SHORT.get(d, d)}: held-out accuracy–cost")
        ax.legend(fontsize=6.3, loc="lower right")
    fig.tight_layout()
    savefig(fig, "task3_adaptive")
    plt.show()

    rows = {}
    for d in avail:
        a = ada[d]
        for key, lab in (("eval_agreement", "adaptive: agreement"), ("eval_quality_aware", "adaptive: quality-aware")):
            e = a[key]
            rows[(d, lab)] = {"policy": e["policy"], "accuracy % [95% CI]": fmt_ci(*ci_of(e["accuracy"])),
                              "mean tokens": f"{e['mean_tokens']:.0f}", "mean rollouts": f"{e['mean_rollouts']:.1f}",
                              "stopped at 4/8/16/32 (%)": " / ".join(f"{100*e['stopping_distribution'].get(k, 0):.0f}" for k in ("4", "8", "16", "32")),
                              "early incorrect stops recovered by fixed-32": f"{e['early_stop_analysis']['recovered_by_fixed32_count']}/{e['early_stop_analysis']['total_early_incorrect_stops']}"}
        for n, v in a["eval_fixed_budget_self_consistency"].items():
            rows[(d, f"fixed SC N={n}")] = {"policy": f"fixed N={n}", "accuracy % [95% CI]": fmt_ci(*ci_of(v["accuracy"])),
                                            "mean tokens": f"{v['mean_tokens']:.0f}", "mean rollouts": n, "stopped at 4/8/16/32 (%)": "—",
                                            "early incorrect stops recovered by fixed-32": "—"}
    t = pd.DataFrame(rows).T
    t.index = pd.MultiIndex.from_tuples(t.index, names=["Dataset", "Strategy"])
    save_table(t.reset_index(), "task3_adaptive", "Held-out adaptive stopping vs fixed budgets (thresholds frozen on the development split).", index=False)
    display(t)

# %% [markdown]
# ## 5. Qualitative examples
# Questions are chosen by a rule fixed in advance (no gold-answer peeking beyond the per-question accuracies already in the
# results): the largest gain of self-consistency over the first answer, the largest loss, and one *selection failure*
# (oracle coverage 1 but low accuracy of the best selector) at N = 8.

# %%
for d in DATASETS:
    if not scor[d]:
        continue
    pq = scor[d]["per_question"]
    sa = load_json(f"{T3}/{d}/step2_selected_answers.json") or {}
    rows = []
    for qid, v in pq.items():
        r = {"qid": qid, "first": v["8"]["first_ans"], "sc": v["8"]["self_consistency"]}
        for m in v["8"]:
            r[m] = v["8"][m]
        rows.append(r)
    P = pd.DataFrame(rows).set_index("qid")
    P["gain_sc"] = P["sc"] - P["first"]
    picks = {"SC helps most": P["gain_sc"].idxmax(), "SC hurts most": P["gain_sc"].idxmin()}
    best_sel = P[[c for c in P.columns if c in KNOWN and c != "first_ans"]].max(axis=1)
    picks["weakest best-selector"] = best_sel.idxmin()
    print(f"\n=== {DATASETS[d]} (N = 8, accuracy averaged over the 10 seeded orders) ===")
    for lab, qid in picks.items():
        print(f"\n[{lab}] {qid}")
        print(P.loc[qid, [c for c in P.columns if c in KNOWN]].round(2).to_dict())
        s = sa.get(qid, {}).get("8", {}).get("selected_answers")
        if s:
            print("   selected answer under the first order:", {m: a[0] for m, a in s.items()})
