# LLM Adaptation, Visual Grounding and Inference Scaling

Results for **Programming Assignment 3** of EE-5102 / CS-6304 *Advanced Topics in Machine Learning* (Fall 2026).
All runs use seed 6304 on a single rented 24 GB GPU (runs of 3–4 November 2026).

| Task | What | Where the results are |
|---|---|---|
| 1 | Qwen3-0.6B on MetaMathQA: Full FT, Full FT + L2-SP, LoRA r=4 / r=16, OPLoRA r=16 (k=128); GSM8K, HellaSwag, BoolQ, ARC-Easy; weight-space and activation diagnostics; qualitative study | `results/task1/` |
| 2 | SigLIP → MLP → Qwen3-0.6B VLM, three-stage curriculum, complementary-pair grounding evaluation, optional 49-token run | `results/task2/`, `results/task2_49tok/` |
| 3 | Test-time compute on the supplied MATH / MathVista rollouts, live generation (50 MATH questions), adaptive stopping | `results/task3/` |

## What is in this repository

```text
results/      machine-readable outputs of the original runs (unmodified): training histories, evaluation reports,
              diagnostics, split / example ids, per-example outputs, Task 3 scoring and adaptive-stopping results
tables/       tables (.md and .tex) generated from results/
figures/      figures (.pdf and .png) generated from results/
notebooks/    analysis scripts / notebooks that read results/ only and write tables/ and figures/ (no GPU needed)
logs_remote/  logs of the real runs on the GPU instance (queue status, per-step training and evaluation logs)
code/         source code, configs and tests (see the status below), with its own README
```

Not included: model weights and optimizer states, cached SigLIP features, COCO / VQAv2 images, and the supplied Task 3
rollout tables (large; they come from the original sources and the assignment's data folder).

Per-example outputs of the Task 1 benchmark runs (extracted GSM8K answers, truncation flags, per-example scores) are in
`results/task1/eval_samples/`; `results/task2/raw_predictions.json` holds the raw Task 2 references and generations.

## Code status — please read

The files in `results/` are the original outputs of the runs, untouched. The source tree that produced them was later
partly overwritten in the working copy by early-draft versions, and **not every original file could be recovered**. This
is what `code/` contains:

| Part | Status |
|---|---|
| Task 1 pipeline: `task1_llm_ft/` (dataset, models, l2_sp, trainer, checkpoints, evaluate, diagnostics), `common/metrics.py` | **Original, restored.** Each file compiles to Python bytecode identical to the compiled files saved from the original working copy. |
| `scripts/run_task1.py`, `configs/task1_config.yaml` | Original, restored from saved copies (the config matches the one recorded in `results/task1/*/training_history.json`). |
| Task 1 analysis (`offline_analysis`, `qualitative_flags`), `notebooks/`, shell scripts, `tests/`, `task2_vlm/data_prep.py`, `task3_test_time/split.py` and `problems.py`, `common/` helpers | Original, unchanged. |
| Task 2: `task2_vlm/` (except `data_prep.py`), `scripts/run_task2.py`, `configs/task2_config.yaml` | **Draft. Not the code that produced the results; the original was not preserved.** |
| Task 3: `task3_test_time/` (except `split.py`, `problems.py`), `scripts/run_task3.py`, `scripts/download_assets.py`, `configs/task3_config.yaml` | **Draft. Not the code that produced the results; the original was not preserved.** |

Consequences:

* For Tasks 2 and 3 the code in this repository **cannot be expected to reproduce the reported numbers**. How those results
  were produced is recorded in the result files themselves: `results/task2/training_summary.json` (stage statistics,
  calibration rule and scale, skipped updates), `results/task2/final_evaluation_report.json` (`eval_settings`),
  `results/task3/run_info_*.json` (command-line arguments, library versions, scorer version, pinned model revision),
  and the logs in `logs_remote/`.
* Part of `tests/` targets the original Task 2 / Task 3 modules and is expected to fail against the drafts.
* `code/task3_test_time/run_task3.py` (draft) falls back to synthetic data and placeholder live-generation problems. Do not run it
  with `--output_dir results/task3`: it would overwrite the real results.

The tables and figures can be regenerated from `results/` alone (`python notebooks/task1_analysis.py`, and likewise for tasks 2 and 3;
they need only numpy, pandas and matplotlib). Task 1's `results/task1/offline_analysis.json` adds sensitivity analyses
(GSM8K answer-extraction check, paired bootstrap intervals, BoolQ yes/no bias) next to the stock lm-evaluation-harness scores; it is computed from `results/task1/eval_samples/`.

## Attributions

* Models: `Qwen/Qwen3-0.6B`, `google/siglip-base-patch16-224`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`.
* Data: MetaMathQA, GSM8K, HellaSwag, BoolQ, ARC-Easy (via lm-evaluation-harness), MS-COCO 2014, VQAv2, MATH, MathVista.
* Methods: LoRA (Hu et al., 2022), OPLoRA (Xiong & Xie), L2-SP (Li et al., 2018), VisConf (Usman et al.), Self-Certainty (Kang et al.),
  Self-Consistency (Wang et al.). Libraries: PyTorch, Hugging Face transformers / PEFT / datasets, lm-evaluation-harness, math-verify.
