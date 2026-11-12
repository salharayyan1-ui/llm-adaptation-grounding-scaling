# ATML PA3: Adaptation, Grounding, and Inference Scaling in Language and Vision Models

Code for Programming Assignment 3 of EE-5102 / CS-6304 *Advanced Topics in Machine Learning* (Fall 2026).
Everything is configured for **one or two 24 GB GPUs (RTX 3090 / 4090)** and seed **6304**.

| Task | What | Entry point |
|---|---|---|
| 1 | Qwen3-0.6B on MetaMathQA: Full FT, Full FT + L2-SP, LoRA r4/r16, OPLoRA r16 (k=128); GSM8K / HellaSwag / BoolQ / ARC-Easy, weight-space + activation diagnostics, qualitative study | `scripts/run_task1.py` |
| 2 | SigLIP -> MLP -> Qwen3-0.6B VLM, 3-stage curriculum, complementary-pair grounding evaluation (optional 49-token run) | `scripts/run_task2.py` |
| 3 | Test-time compute on the supplied MATH / MathVista rollouts, live generation, adaptive stopping | `scripts/run_task3.py` |

Large files (checkpoints, features, raw predictions) go to `outputs/` and `data/` (git-ignored). Small machine-readable
results (training logs, benchmark tables, split / example ids, reports) go to `results/` and **are meant to be committed**.

> **Status of this folder — read first.** This is the original pipeline's documentation, but not every file below is the
> original code. See the table in the top-level [README](../README.md#code-status--please-read) and the list at the end of this file.
> **Task 1 is the original code**: the nine Python modules were restored and each compiles to bytecode identical to the saved
> original; `scripts/run_task1.py` and `configs/task1_config.yaml` were restored from saved copies (the config matches the one
> recorded in `results/task1/*/training_history.json`). **Task 2 and Task 3 modules listed as "draft" are early versions that
> did not produce the reported results**; their originals were not preserved.
> The commands below are the original ones and assume the layout in which `common/`, `scripts/`, `configs/`, `tests/` and
> the `task*/` packages sit next to `results/` (e.g. `cp -r code/* .` from the repository root). Per-example Task 1 outputs, needed
> by `offline_analysis`, are stored in `results/task1/eval_samples/` (the code reads them from `outputs/task1/eval_samples/`).

## Setup

```bash
pip install -r requirements.txt            # install the CUDA build of torch first (https://pytorch.org)
python -m unittest discover -s tests -t .  # unit tests (mock-torch based; run anywhere, CPU only)
python scripts/smoke_test_gpu.py           # REAL torch / transformers / PEFT smoke test with tiny random models (~3 min, GPU box)
```

The unit tests replace torch by a NumPy mock when torch is missing, so they validate formulas and bookkeeping, **not**
the GPU code paths. Run `scripts/smoke_test_gpu.py` on the training machine before the long runs.

## Task 1: LLM fine-tuning

```bash
python scripts/launch_multi_gpu.py --task 1          # 1 or 2 GPUs: trains the 5 variants, evaluates 7 conditions, diagnostics, tables

# or step by step (add --gpu N to pin a GPU):
python scripts/run_task1.py --action train --variant full_ft          # full_ft | full_ft_l2_sp | lora_4 | lora_16 | oplora_16
python scripts/run_task1.py --action train --variant lora_16 --resume_from outputs/task1/lora_16/checkpoint_step_625
python scripts/run_task1.py --action evaluate --conditions baseline,full_ft_final    # omit --conditions for all seven
python scripts/run_task1.py --action qualitative
python scripts/run_task1.py --action diagnostics
python scripts/run_task1.py --action aggregate        # benchmark_results.{json,md} with changes in percentage points
python scripts/run_task1.py --action offline_analysis # CPU only, no model: needs outputs/task1/eval_samples/ -> results/task1/offline_analysis.json
```

* Full FT / L2-SP keep **FP32 master weights** under FP16 autocast + GradScaler; adapter variants use a frozen FP16 backbone
  with FP32 adapters. Loss is normalized over all valid response tokens of the effective batch (32 sequences).
* L2-SP: `Omega = 0.5 * sum (theta - theta_0)^2` over all unique parameters; its gradient `lambda (theta - theta_0)` is added once per
  update (identical to adding `lambda * Omega` to the loss). `Omega`, `lambda * Omega` and the total loss are logged every update.
* OPLoRA evaluation/diagnostics use the **exported projected factors** (`oplora_projected_factors.pt`); LoRA checkpoints store
  adapter tensors only, so checkpoint sizes are comparable across methods (`results/task1/<variant>/summary.json`).
* Benchmarks run through `lm-evaluation-harness` (GSM8K = `gsm8k_cot`, zero-shot, non-thinking chat template, greedy, 1,024 tokens,
  flexible-extract). Adapters are merged into a fresh FP16 backbone; merged-vs-unmerged logits are checked and logged.
* `offline_analysis` re-reads the saved per-example outputs and never changes a reported score. It adds (i) a GSM8K
  answer-extraction sensitivity check (the stock `flexible-extract` regex also matches a bare `$$`, so responses that end in a
  LaTeX display block can lose their answer; the stock filter is replicated and verified against the stored extractions, then
  re-scored skipping digit-free matches and preferring `\boxed{}`), (ii) paired bootstrap 95% intervals (examples resampled
  jointly, 2,000 resamples, seed 6304) for the controlled comparisons and the changes from the baseline, and (iii) a BoolQ
  yes/no-bias summary (predicted-yes rate, per-class accuracy, balanced accuracy, ROC-AUC of the yes-vs-no log-likelihood gap).
  The notebook reads `offline_analysis.json` and writes `task1_gsm8k_sensitivity`, `task1_comparisons`, `task1_vs_baseline`
  and `task1_boolq_bias` tables. The primary GSM8K metric stays the stock flexible-extract score.
* Disk: a Full FT run writes four checkpoints with optimizer state (~7 GB each). Set `checkpoints.save_optimizer_state: false` to save space
  (resume then becomes unavailable).
* `configs/first_plot_questions.yaml` must be the **file supplied with the assignment**; both its original layout (`paraphrases`/`system`)
  and the layout in this repo are accepted. The first/last plain questions are taken from the file order.
* `inappropriate_content` / `manual_notes` in `qualitative/annotation_subset_42.json` are left for manual annotation, and
  re-running the aggregation keeps any you have already filled in. The automatic `unnecessary_math_framing` flag fires on
  MetaMathQA / GSM8K training artefacts (`The answer is: x`, `#### n`, `\boxed`, LaTeX, `<<calc>>`) as well as math vocabulary;
  `math_framing_signals` records which (`task1_llm_ft/qualitative_flags.py`). `offline_analysis` also counts these over all 24
  responses per condition (`qualitative_math_intrusion`, table `task1_qualitative_math_intrusion`). They are heuristics
  to orient the manual pass, not a replacement for it.

## Task 2: VLM and visual grounding

```bash
python scripts/download_assets.py --what annotations,vqa     # ~290 MB json; the images are fetched individually later
python scripts/run_task2.py --action prepare                 # selects/pairs data, verifies split isolation, writes ids to results/task2/split_ids.json
python scripts/download_assets.py --what images              # only the ~10k selected COCO images
python scripts/run_task2.py --action cache_features          # SigLIP FP16 features (+ caching time / memory in results/task2/caching_stats.json)
python scripts/run_task2.py --action train                   # 4,500 updates, resumable every 250
python scripts/run_task2.py --action evaluate                # 4 image conditions x 2 suites, 4,000 pair-bootstrap resamples
python scripts/run_task2.py --action compress --gpu 1        # optional: 49 visual tokens (outputs to *_49tok)
```

or `python scripts/launch_multi_gpu.py --task 2` (the 49-token run goes to the second GPU when there is one).

* Training/dev/test images come from train2014 / val2014; dev and test sets are image-disjoint and use every image once.
* The connector is calibrated against the Qwen embedding norm of tokens occurring in the *training* data (rule + scale in `results/task2/calibration.json`).
* The visual placeholder id is read from the tokenizer (`<|image_pad|>` = 151655); the prompt contains Qwen's empty `<think>` block (thinking disabled).

## Task 3: test-time compute

```bash
python scripts/download_assets.py --what task3_data,official       # ATML-PA3-Data (Google Drive folder of the assignment PDF) + testmini.json + official prompts
python scripts/run_task3.py --step all --metadata_path data/ATML-PA3-Data/testmini.json   # offline steps (CPU) + live generation (GPU)
python scripts/run_task3.py --step opportunity|scoring|signals|adaptive [--dataset math|mathvista] --metadata_path data/ATML-PA3-Data/testmini.json
python scripts/run_task3.py --step live --dataset math --gpu 0 --model_revision aa8e72537993ba99e69dfaafa59ed015b17504d1
```

* The data folder (`qwen25-math/`, `qwen25vl-mathvista/`, `README.MD`) is the one linked on page 11 of the assignment PDF. The pinned
  `Qwen/Qwen2.5-3B-Instruct` revision above is the one listed in that README. The MATH problem text comes from the `question_text` column.
* Real data is mandatory: missing files / unknown column names / missing MathVista metadata **raise** (override column names in
  `configs/task3_config.yaml -> data.columns`). `--synthetic` exists for smoke tests only and tags the run in `run_info_*.json`.
* The live step appends to `live_generations.jsonl` after every question and resumes after an interruption. Run it on an otherwise idle GPU:
  it records end-to-end runtime and peak memory.
* Scorer: `math-verify` (primary) + a safe numeric / bounded symbolic fallback (large exponents are never expanded; every symbolic
  check has a hard time limit); the version string is stored in `results/task3/run_info_*.json`. The offline steps take ~35 min on CPU.
* Orders: 10 seeded permutations per question (stream derived from the seed and the question id) are saved in `permutations.json`.

## Multi-GPU use

`scripts/launch_multi_gpu.py --task N` runs independent jobs (variants, conditions, the 49-token run, live generation) on every
GPU it finds with one job per GPU and logs to `outputs/logs/`. Each job is also runnable by hand with `--gpu`. Nothing needs more than 24 GB:
Full FT + L2-SP (FP32 weights, gradients, Adam states and theta_0) is the largest at roughly 12-14 GB plus activations.

## Running on a rented GPU box and bringing the results back

```bash
# 1. copy the repo (no .git / data / outputs) to the box, then on the box:
bash /workspace/pa3/scripts/bootstrap_instance.sh      # deps, GPU smoke test, COCO/VQAv2 prep, image download, starts the queue
tmux attach -t night                                   # or: tail -f /workspace/logs/queue_status.txt

# 2. later, on your own machine (Git Bash), pull the small result files + per-example outputs + logs:
bash scripts/pull_results.sh "ssh -p PORT -i path/to/key root@HOST"
```

`scripts/queue_overnight.sh` runs everything in one idempotent pass (every step has a done-marker, training resumes from its latest
checkpoint, a failed step is retried once and never blocks the rest): baseline + adapter runs and their evaluations, Full FT and
L2-SP with the spec checkpoints (4 x ~7 GB incl. optimizer state while training, pruned after their evaluations; needs >= ~60 GB disk,
falls back to lean checkpoints on a smaller disk), Task 2 (cache, train, evaluate), diagnostics / qualitative / aggregation, and the
optional 49-token run last. Task 3 (offline CPU steps + the idle-GPU live generation) is run separately (see above).

## Analysis notebooks (run locally, no GPU / torch needed)

```bash
pip install matplotlib pandas jupyter                  # only these
python notebooks/build_ipynb.py                        # regenerates the .ipynb files from the readable task*_analysis.py sources
jupyter lab notebooks/                                 # open task1_analysis / task2_analysis / task3_analysis
```

Each notebook reads `results/task{1,2,3}/…` and writes figures to `figures/` (PDF + PNG) and tables to `tables/` (`.md` + `.tex`),
covering the "Required Evidence" lists of the assignment for that task. Missing files only print `[not available yet]`, so the
notebooks can be run at any stage. `PA3_RESULTS=<dir>` / `PA3_OUT=<dir>` redirect input / output.

## Repository layout

```text
                                                                                      status
common/            seed, device, normalization                                        original
                   metrics (VQA normalization, rho_k / eta / R(z))                    original (restored)
task1_llm_ft/      dataset, models, l2_sp, trainer, checkpoints, evaluate,
                   diagnostics, __init__                                              original (restored)
                   offline_analysis, qualitative_flags                                original
task2_vlm/         data_prep                                                          original
                   caching, models, dataset, trainer, evaluate, compression, __init__ DRAFT (original not preserved)
task3_test_time/   split, problems                                                    original
                   data_loader, equivalence, opportunity, scoring, signals,
                   adaptive, live_generation, run_task3, __init__                     DRAFT (original not preserved)
scripts/           run_task1.py (restored); launch_multi_gpu.py, smoke_test_gpu.py,
                   bootstrap_instance.sh, queue_overnight.sh, pull_results.sh         original
                   run_task2.py, run_task3.py, download_assets.py                     DRAFT (original not preserved)
configs/           task1_config.yaml (restored), first_plot_questions.yaml            original
                   task2_config.yaml, task3_config.yaml                               DRAFT (see results/*/ for the settings actually used)
tests/             unit tests (mock-torch), written against the original packages     original
```

"Original" was checked by compiling the file and comparing it with the Python bytecode saved from the original working copy
(identical code objects). The drafts for Task 2/3 will not match the tests in `tests/` that target the original modules
(for example `test_task2_metrics.py`, `test_task3_pathological.py`), and `task3_test_time/run_task3.py` (draft) falls back to
synthetic data and placeholder live-generation problems, so **do not run it against `results/`**.
The notebooks and `nb_utils.py` live in the top-level `notebooks/` folder.

## Attributions

* Models: `Qwen/Qwen3-0.6B`, `google/siglip-base-patch16-224`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`.
* Data: MetaMathQA, GSM8K, HellaSwag, BoolQ, ARC-Easy (via lm-evaluation-harness), MS-COCO 2014, VQAv2, MATH, MathVista.
* Methods: LoRA (Hu et al., 2022), OPLoRA (Xiong & Xie), L2-SP (Li et al., 2018), VisConf (Usman et al.), Self-Certainty (Kang et al.),
  Self-Consistency (Wang et al.). Libraries: PyTorch, Hugging Face transformers / PEFT / datasets, lm-evaluation-harness, math-verify.
