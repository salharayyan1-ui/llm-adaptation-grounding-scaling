# %% [markdown]
# # Task 2 — Visual interface and grounding: analysis
#
# Reads `results/task2/…` (written by `scripts/run_task2.py`) and produces the evidence required for Task 2:
#
# 1. learning curves — training loss, development caption loss, answer NLL, image-swap penalty, stage boundaries
# 2. final results for both suites × the four image conditions (accuracy, pair accuracy, identical/empty rates, NLL,
#    question-type results, generation time) with 95 % **pair-bootstrap** intervals
# 3. correct-image advantages over complementary swaps / mismatched / no-image (accuracy and NLL) and swap behaviour
# 4. aligned complementary-image examples (with the stated selection rule)
# 5. resources — training/generation time, peak memory, trainable parameters by stage, feature-caching cost
# 6. optional 49-visual-token run vs the 196-token run
#
# Missing files only print `[not available yet]`.

# %%
from nb_utils import *

T2 = "task2"
COND = ["correct", "swap", "mismatched", "no_image"]
COND_LABEL = {"correct": "Correct image", "swap": "Complementary swap", "mismatched": "Mismatched", "no_image": "No image"}
COND_COLOR = dict(zip(COND, CAT[:4]))          # colour follows the image condition everywhere
SUITE = {"grounding_suite": "Grounding suite (presence / counting)", "vqav2_suite": "VQAv2 suite (natural questions)"}
SUITE_SHORT = {"grounding_suite": "Grounding", "vqav2_suite": "VQAv2"}

report = load_json(f"{T2}/final_evaluation_report.json")
train = load_json(f"{T2}/training_summary.json")
dev = load_json(f"{T2}/dev_logs.json")
loss = load_csv(f"{T2}/loss_curve.csv")
caching = load_json(f"{T2}/caching_stats.json")
ids = load_json(f"{T2}/split_ids.json")
illu = load_json(f"{T2}/illustrative_pairs.json")

# %% [markdown]
# ## 0. Data, splits and the connector calibration

# %%
if ids and "stats" in ids:
    rows = {}
    for name, st in ids["stats"].items():
        if isinstance(st, dict):
            rows[name] = {"items": st.get("items"), "questions/captions": st.get("questions"),
                          "by type": ", ".join(f"{k}: {v}" for k, v in (st.get("by_type") or {}).items())}
    t = pd.DataFrame(rows).T
    t.index.name = "Split"
    save_table(t, "task2_data", "Sizes of the training / development / test sets (images are disjoint across splits).")
    display(t)
if train and train.get("calibration"):
    print("Connector calibration (rule and resulting scale):")
    display(pd.Series({k: (v if not isinstance(v, (list, dict)) else str(v)[:90]) for k, v in train["calibration"].items()}, name="value").to_frame())

# %% [markdown]
# ## 1. Learning curves
# Stage 1 = connector only (captions), stage 2 = connector + LoRA (mixture with grounding data), stage 3 = natural VQAv2
# mixture. Development metrics use the development split only (never the test pairs).

# %%
stage_edges = None
if train and train.get("stage_stats"):
    ups = [train["stage_stats"][str(s)]["updates"] for s in (1, 2, 3) if str(s) in train["stage_stats"]]
    stage_edges = list(np.cumsum(ups))


def dev_series(key):
    xs, ys = [], []
    for r in dev or []:
        v = r
        for k in key:
            v = (v or {}).get(k) if isinstance(v, dict) else None
        if v is not None:
            xs.append(r["global_update"])
            ys.append(v)
    return xs, ys


if loss is not None or dev:
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.4))
    ax = axes[0, 0]
    if loss is not None:
        ax.plot(loss["update"], smooth(loss["loss"], 50), color=CAT[0])
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("loss (50-update moving mean)")
    panel_title(ax, "Training loss")
    ax = axes[0, 1]
    x, y = dev_series(("caption_loss",))
    ax.plot(x, y, color=CAT[0], marker="o")
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("loss")
    panel_title(ax, "Development caption loss")
    for ax, suite, ttl in ((axes[0, 2], "grounding", "Dev answer NLL — grounding"), (axes[1, 0], "vqav2", "Dev answer NLL — VQAv2")):
        for cond, key in (("correct", "nll_correct"), ("swap", "nll_swap"), ("no_image", "nll_no_image")):
            x, y = dev_series((suite, key))
            ax.plot(x, y, color=COND_COLOR[cond], marker="o", ms=3.5, label=COND_LABEL[cond],
                    ls="-" if cond == "correct" else (0, (4, 2)))
        ax.set_xlabel("optimizer update")
        ax.set_ylabel("mean reference NLL")
        panel_title(ax, ttl)
    cond_handles = [plt.Line2D([0], [0], color=COND_COLOR[c], marker="o", ms=3.5, ls="-" if c == "correct" else (0, (4, 2))) for c in ("correct", "swap", "no_image")]
    ax = axes[1, 1]
    for suite, col, mk in (("grounding", CAT[2], "o"), ("vqav2", CAT[4], "s")):
        x, y = dev_series((suite, "delta_nll"))
        ax.plot(x, y, color=col, marker=mk, ms=3.5, label=SUITE_SHORT[suite + "_suite"])
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("optimizer update")
    ax.set_ylabel("NLL(swap) − NLL(correct)")
    panel_title(ax, "Image-swap penalty (dev)")
    ax.legend(fontsize=6.5)
    ax = axes[1, 2]
    mix = (train or {}).get("mixture_counts_by_stage")
    if mix:
        try:
            names = sorted({n for s in mix.values() for n in s})
            left = np.zeros(len(mix))
            for i, n in enumerate(names):
                vals = np.array([mix[s].get(n, 0) for s in sorted(mix)], dtype=float)
                tot = np.array([sum(mix[s].values()) for s in sorted(mix)], dtype=float)
                ax.barh([f"Stage {s}" for s in sorted(mix)], vals / np.maximum(tot, 1), left=left,
                        color=CAT[i % len(CAT)], label=n, height=0.55, edgecolor="white", linewidth=1.0)
                left += vals / np.maximum(tot, 1)
            ax.invert_yaxis()
            ax.set_xlabel("share of training examples")
            ax.legend(fontsize=5.8, loc="center left", bbox_to_anchor=(1.0, 0.5))
        except Exception as e:  # noqa: BLE001
            ax.text(0.5, 0.5, f"mixture not plotted\n({e})", ha="center", va="center", transform=ax.transAxes, fontsize=7)
    panel_title(ax, "Sampled data mixture")
    if stage_edges:
        for a in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]):
            for b in stage_edges[:-1]:
                a.axvline(b, color=AXIS, lw=0.8, ls=(0, (3, 3)), zorder=0)
        stage_bands(axes[0, 0], stage_edges)
        axes[0, 0].set_title("Training loss", loc="left", pad=15, color=INK, fontsize=8.5)
    fig.legend(cond_handles, [COND_LABEL[c] for c in ("correct", "swap", "no_image")], loc="lower center", ncol=3,
               bbox_to_anchor=(0.38, -0.02), fontsize=7)
    fig.text(0.01, -0.02, "dashed lines: stage boundaries (S1→S2→S3)", fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    savefig(fig, "task2_learning_curves")
    plt.show()

# %% [markdown]
# ## 2. Final results — both suites, four image conditions
# Intervals are 95 % bootstrap CIs over complete pairs (4,000 resamples, identical resamples across conditions).
# Accuracy of the grounding suite = exact match; of the VQAv2 suite = official VQA accuracy against the 10 references.

# %%
if report:
    main_tables = {}
    for suite in SUITE:
        if suite not in report:
            continue
        rs = {}
        for c in COND:
            m = report[suite].get(c)
            if not m:
                continue
            a, p, n = ci_of(m["accuracy"]), ci_of(m["pair_accuracy"]), ci_of(m["mean_reference_nll"])
            rs[COND_LABEL[c]] = {
                "Accuracy % [95% CI]": fmt_ci(*a), "Pair accuracy % [95% CI]": fmt_ci(*p),
                "Identical answers %": pct(100 * m["identical_answer_rate"]), "Empty answers %": pct(100 * m["empty_answer_rate"]),
                "Ref. NLL [95% CI]": fmt_ci(*n, scale=1.0, d=3), "Gen. time / answer (s)": f"{m['mean_gen_time_sec']:.3f}",
                "pairs": m["n_pairs"]}
        t = pd.DataFrame(rs).T
        t.index.name = SUITE[suite]
        main_tables[suite] = t
        save_table(t, f"task2_{suite}", f"{SUITE[suite]}: all four image conditions.")
        display(t)

    # question-type results
    for suite in SUITE:
        if suite not in report:
            continue
        sub = {COND_LABEL[c]: {k: 100 * v for k, v in report[suite][c]["subtypes"].items()} for c in COND if c in report[suite]}
        t = pd.DataFrame(sub).T.round(1)
        t.index.name = f"{SUITE_SHORT[suite]}: accuracy (%) by question type"
        save_table(t, f"task2_{suite}_types", f"{SUITE[suite]}: accuracy by question type.")
        display(t)

# %%
if report:
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4), sharey=False)
    k = 0
    for suite in SUITE:
        if suite not in report:
            continue
        for metric, mttl in (("accuracy", "accuracy"), ("pair_accuracy", "pair accuracy")):
            ax = axes[k]
            k += 1
            for i, c in enumerate(COND):
                m = report[suite].get(c)
                if not m:
                    continue
                mean, lo, hi = ci_of(m[metric])
                ax.bar(i, 100 * mean, width=0.62, color=COND_COLOR[c], edgecolor="white", linewidth=1.0,
                       yerr=[[100 * (mean - lo)], [100 * (hi - mean)]], ecolor=INK2, error_kw={"elinewidth": 0.9, "capsize": 2})
            ax.set_xticks(range(len(COND)), ["correct", "swap", "mism.", "none"], rotation=0)
            ax.set_ylim(0, 100)
            if k == 1:
                ax.set_ylabel("%")
            panel_title(ax, f"{SUITE_SHORT[suite]} {mttl}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COND_COLOR[c]) for c in COND]
    fig.legend(handles, [COND_LABEL[c] for c in COND], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    savefig(fig, "task2_final_results")
    plt.show()

# %% [markdown]
# ## 3. Does the answer depend on the image?
# *Correct-image advantage* = score(correct) − score(control) for accuracy, and NLL(control) − NLL(correct) for the
# reference likelihood (positive = the correct image helps). Swapped-image scores still use the **original** references;
# *follows partner* counts answers equal to the partner image's reference.

# %%
if report:
    rows = {}
    for suite in SUITE:
        if suite not in report:
            continue
        for ctrl in ("swap", "mismatched", "no_image"):
            adv = report[suite].get(f"correct_advantage_over_{ctrl}")
            if adv:
                rows[(SUITE_SHORT[suite], COND_LABEL[ctrl])] = {
                    "Δ accuracy, pp [95% CI]": fmt_ci(*ci_of(adv["delta_accuracy"])),
                    "Δ NLL penalty [95% CI]": fmt_ci(*ci_of(adv["delta_nll_penalty"]), scale=1.0, d=3)}
    if rows:
        t = pd.DataFrame(rows).T
        t.index = pd.MultiIndex.from_tuples(t.index, names=["Suite", "Control"])
        save_table(t.reset_index(), "task2_advantages", "Correct-image advantages over the controls (pair-bootstrap CIs).", index=False)
        display(t)
    sb = {SUITE_SHORT[s]: {k: 100 * v for k, v in report[s]["swap_behavior"].items()} for s in SUITE if s in report and "swap_behavior" in report[s]}
    if sb:
        t = pd.DataFrame(sb).T.round(1)
        t.columns = ["follows partner image %", "unchanged vs correct image %", "other %"] + (
            ["answer equals partner reference % (overlaps)"] if t.shape[1] == 4 else [])
        t.index.name = "Complementary swap behaviour"
        save_table(t, "task2_swap_behavior", "What happens to the answer when the image is swapped for its complementary partner.")
        display(t)

# %% [markdown]
# ## 4. Aligned complementary-image examples
# Selection rule (fixed before looking at outputs, pairs sorted by pair id): **A** correct on both images *and* both swapped
# answers follow the partner image; **B** correct on both images but the answers do not change under the swap
# (insensitive to the image); **C** wrong on at least one correct-image answer. The first three of each category are listed.

# %%
def find_image(key):
    """Locate the COCO jpg for an image key like 'val2014_000000123456' if the images are present locally."""
    m = re.match(r"(train|val)2014_0*(\d+)$", str(key))
    if not m:
        return None
    split, iid = m.group(1), int(m.group(2))
    for base in (ROOT / "data" / "coco2014", ROOT / "data" / "coco"):
        p = base / f"{split}2014" / f"COCO_{split}2014_{iid:012d}.jpg"
        if p.exists():
            return p
    return None


if illu:
    from PIL import Image
    for suite, cats in illu.items():
        for cat, items in cats.items():
            for it in items:
                print(f"[{SUITE_SHORT.get(suite, suite)} / {cat}] {it['pair_id']}: “{it['question']}”  references = {it['references']}")
                for c in COND:
                    if c in it["predictions"]:
                        print(f"    {COND_LABEL[c]:20s} -> {it['predictions'][c]}")
    # image panels for the first example of every category, if the jpgs are available locally
    shown = 0
    for suite, cats in illu.items():
        for cat, items in cats.items():
            if not items or shown >= 6:
                continue
            it = items[0]
            paths = [find_image(k) for k in it["images"]]
            if not all(paths):
                continue
            fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.5))
            for ax, p, ref, pred in zip(axes, paths, it["references"], it["predictions"]["correct"]):
                ax.imshow(Image.open(p).convert("RGB"))
                ax.axis("off")
                ax.set_title(f"ref: {ref}  |  pred: {pred}", fontsize=7.5)
            fig.suptitle(f"{SUITE_SHORT.get(suite, suite)} · {cat}: {show_excerpt(it['question'], 80)}", fontsize=8, x=0.02, ha="left")
            fig.tight_layout()
            savefig(fig, f"task2_example_{suite}_{cat}")
            plt.show()
            shown += 1
    if shown == 0:
        print("(images not found under data/coco2014/{train,val}2014 — copy the ~600 val2014 jpgs to show them)")

# %% [markdown]
# ## 5. Resources

# %%
if train and train.get("stage_stats"):
    st = {f"Stage {s}": {"updates": v.get("updates"),
                         "training time (min)": (v.get("train_seconds", v.get("seconds")) or 0) / 60,
                         "dev evaluation (min)": (v.get("dev_eval_seconds") or 0) / 60,
                         "peak GPU mem (GB)": (v.get("peak_memory_mb") or 0) / 1024,
                         "trainable params (M)": (v.get("trainable_parameters") or 0) / 1e6}
          for s, v in sorted(train["stage_stats"].items())}
    t = pd.DataFrame(st).T
    t.loc["Total / max"] = [t["updates"].sum(), t["training time (min)"].sum(), t["dev evaluation (min)"].sum(),
                            t["peak GPU mem (GB)"].max(), t["trainable params (M)"].max()]
    t = t.round(2)
    t.index.name = "Training (196 visual tokens)"
    save_table(t, "task2_resources_training", "Training cost by stage: pure training time, development-evaluation time, peak memory, trainable parameters.")
    display(t)
    print(f"wall-clock training run incl. dev evaluation and checkpoint writes: {train['training_time_sec'] / 60:.1f} min;",
          "skipped updates (fp16 overflow):", train.get("skipped_updates"))
if caching:
    print("Feature caching (SigLIP, FP16):")
    display(pd.Series({k: v for k, v in caching.items() if not isinstance(v, (list, dict))}, name="value").to_frame())
if report and "eval_settings" in report:
    es = report["eval_settings"]
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in es.items()})

# %% [markdown]
# ## 6. Optional: 49 visual tokens (2×2 average pooling) vs 196 tokens

# %%
rep49 = load_json("task2_49tok/final_evaluation_report.json")
tr49 = load_json("task2_49tok/training_summary.json")
if report and rep49:
    rows = {}
    for suite in SUITE:
        for name, rep in (("196 tokens", report), ("49 tokens", rep49)):
            m = rep.get(suite, {}).get("correct")
            adv = rep.get(suite, {}).get("correct_advantage_over_swap")
            if m:
                rows[(SUITE_SHORT[suite], name)] = {
                    "Accuracy % [CI]": fmt_ci(*ci_of(m["accuracy"])), "Pair acc. % [CI]": fmt_ci(*ci_of(m["pair_accuracy"])),
                    "Ref. NLL": f"{ci_of(m['mean_reference_nll'])[0]:.3f}",
                    "Swap advantage, Δacc pp": fmt_ci(*ci_of(adv["delta_accuracy"])) if adv else "n/a",
                    "Gen. time / answer (s)": f"{m['mean_gen_time_sec']:.3f}"}
    t = pd.DataFrame(rows).T
    t.index = pd.MultiIndex.from_tuples(t.index, names=["Suite", "Visual tokens"])
    save_table(t.reset_index(), "task2_compression", "196-token vs 49-token runs (correct-image condition).", index=False)
    display(t)
    if train and tr49:
        cmp = pd.DataFrame({"196 tokens": {"training time (min)": train["training_time_sec"] / 60, "peak GPU mem (GB)": train["peak_memory_mb"] / 1024,
                                           "eval time (s)": report["eval_settings"]["eval_seconds"]},
                            "49 tokens": {"training time (min)": tr49["training_time_sec"] / 60, "peak GPU mem (GB)": tr49["peak_memory_mb"] / 1024,
                                          "eval time (s)": rep49["eval_settings"]["eval_seconds"]}}).round(2)
        display(cmp)
else:
    print("49-token run not available yet (results/task2_49tok/…)")
