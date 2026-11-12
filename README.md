# Advanced Topics in Machine Learning: Programming Assignment 3
## Adaptation, Grounding, and Inference Scaling in Language and Vision Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete experimental framework and codebase for **Programming Assignment 3** of the graduate course **EE-5102 / CS-6304: Advanced Topics in Machine Learning** (Fall 2026).

---

## Overview of Research Tasks

### Task 1: Large Language Model Fine-Tuning & Retention
Investigates the tension between mathematical adaptation (on MetaMathQA) and retention of existing capabilities across standard zero-shot benchmarks (GSM8K, HellaSwag, BoolQ, ARC-Easy). We systematically compare five adaptation strategies starting from `Qwen/Qwen3-0.6B` (`enable_thinking=False`):
1. **Full Fine-Tuning (Full FT)**: Unconstrained parameter updates ($\lambda = 0$).
2. **Full FT with L2-SP Regularization**: Penalizes departures from the starting model ($\lambda = 10^{-3}$, $\Omega(\theta, \theta_0) = \frac{1}{2}\sum_j (\theta_j - \theta_{0,j})^2$).
3. **Low-Rank Adaptation (LoRA-4)**: Rank $r=4$, scaling $\alpha=8$.
4. **Low-Rank Adaptation (LoRA-16)**: Rank $r=16$, scaling $\alpha=32$.
5. **Orthogonal Projection LoRA (OPLoRA-16)**: Rank $r=16$, $\alpha=32$, protection rank $k=128$. Projects adapter updates onto the orthogonal complement of the starting weight's leading singular vectors:
   $$\tilde{B} = B - U_k(U_k^\top B), \quad \tilde{A} = A - (A V_k)V_k^\top, \quad \Delta W = \frac{\alpha}{r}\tilde{B}\tilde{A}$$

Includes weight-space subspace interference $\rho_k$, relative update magnitude $\eta$, activation diagnostic ratios $R(z)$, and qualitative analysis across 24 prompts from `configs/first_plot_questions.yaml`.

### Task 2: Vision-Language Models and Visual Grounding
Builds and trains a SigLIP–Qwen VLM connecting frozen `google/siglip-base-patch16-224` to `Qwen/Qwen3-0.6B` through a two-layer MLP connector ($768 \to 1024 \to 1024$). Evaluates whether multimodal representations exhibit true visual grounding or exploit text priors:
* **Curriculum Training (4,500 total updates)**:
  * *Stage 1 (Caption Alignment)*: 2,250 updates on 18,000 COCO caption pairs (connector only).
  * *Stage 2 (Visual Instruction Tuning)*: 1,500 updates on presence/counting grounding pairs with caption replay (connector + LoRA).
  * *Stage 3 (Natural VQA Specialization)*: 750 updates on filtered VQAv2 complementary pairs with curriculum replay (connector + LoRA).
* **Controlled Grounding Evaluation**: Assesses predictions across 4 image conditions: *Correct*, *Complementary Swap*, *Mismatched*, and *No-Image*, computing Image-Swap Penalty $\Delta\text{NLL}$ and 4,000 paired bootstrap confidence intervals.

### Task 3: Test-Time Compute: Sampling, Selection, and Adaptive Inference
Explores compute allocation strategies at inference time using pre-generated rollouts (`ATML-PA3-Data`):
* **Repeated Sampling**: Candidate budgets $N \in \{1, 2, 4, 8, 16, 32\}$ comparing FirstAns, Self-Consistency (majority voting), and Oracle coverage.
* **Confidence & Quality Scoring**: Mean log-probability $L_i$, Self-Certainty $K_i$, Visual Engagement $V_i$, and candidate-calibrated VisConf $Z_i = V_i C_i$.
* **Rank-Weighted Consensus (RWC)**: Quality-weighted consensus voting $w_i = (N - r_i)/(N - 1)$.
* **Live Generation vs. Sequential Refinement**: Controlled live comparison on 50 MATH evaluation questions with `Qwen/Qwen2.5-3B-Instruct` under a fixed 3,072-token budget.
* **Adaptive Sampling Policies**: Simulates online early stopping comparing agreement-based vs. quality-aware stopping rules.

---

## Repository Structure

```text
pa3/
├── common/                  # Shared math, seed, device, normalization, and metric utilities
│   ├── device.py            # Device detection, GPU memory tracking, and allocation resets
│   ├── metrics.py           # VQA consensus accuracy, bootstrap CIs, rho_k, eta, R(z)
│   ├── normalization.py     # Balanced-brace boxed parsing, text & MathVista normalization
│   └── seed.py              # Global deterministic seed setting (seed 6304)
├── configs/                 # Declarative experiment configurations
│   ├── first_plot_questions.yaml  # 24 qualitative evaluation prompts
│   ├── task1_config.yaml    # Hyperparameters for LLM fine-tuning
│   ├── task2_config.yaml    # Architecture and curriculum settings for VLM
│   └── task3_config.yaml    # Test-time compute parameters and search grids
├── task1_llm_ft/            # Task 1: LLM Fine-Tuning pipeline
│   ├── dataset.py           # MetaMathQA loader, GSM8K decontamination filtering, masking
│   ├── models.py            # LoRA and OPLoRA linear layers, SVD precomputation
│   ├── l2_sp.py             # L2-SP proximity penalty and parameter displacement tracker
│   ├── trainer.py           # FP16 AMP training loop with effective-batch loss normalization
│   ├── diagnostics.py       # Weight-space and probe activation diagnostics
│   └── evaluate.py          # lm-evaluation-harness runner and qualitative generation
├── task2_vlm/               # Task 2: VLM & Visual Grounding pipeline
│   ├── caching.py           # SigLIP FP16 feature extraction and encoder offloading
│   ├── models.py            # 2-layer MLP connector with text-norm calibration
│   ├── dataset.py           # COCO and VQAv2 paired dataset builders
│   ├── trainer.py           # 3-stage curriculum trainer with example-weighted loss
│   ├── evaluate.py          # 4-condition grounding evaluation and pair bootstrap
│   └── compression.py       # Optional 49-token spatial pooling experiment
├── task3_test_time/         # Task 3: Test-Time Compute pipeline
│   ├── data_loader.py       # Parquet ingestion for LLM and VLM rollouts
│   ├── split.py             # Deterministic 200/800 question splitting (seed 6304)
│   ├── equivalence.py       # math-verify and MathVista answer normalization
│   ├── opportunity.py       # Candidate budgets N, oracle coverage, failure breakdown
│   ├── scoring.py           # Log-prob, Self-Certainty, VisConf, and RWC consensus
│   ├── signals.py           # Length correlations and tokenwise fusion covariance ablation
│   ├── live_generation.py   # 50-question live generation (single, parallel, refinement)
│   └── adaptive.py          # Online early-stopping controllers (agreement & quality-aware)
├── scripts/                 # CLI entry points
│   ├── download_assets.py   # Dataset directory setup and asset guidance
│   ├── run_task1.py         # Task 1 runner
│   ├── run_task2.py         # Task 2 runner
│   └── run_task3.py         # Task 3 runner
├── tests/                   # Syntax compilation and mathematical unit tests
│   ├── test_formulas.py     # NumPy-based mathematical verification tests
│   └── test_syntax.py       # Full-repository Python AST / compilation test
└── requirements.txt         # Dependencies
```

---

## Environment Setup & Verification

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Verification Tests
Verify mathematical formulations and compilation integrity without requiring GPU or PyTorch:
```bash
python -m unittest tests.test_formulas
python tests/test_syntax.py
```

### 3. Prepare Datasets
```bash
python scripts/download_assets.py --data_dir ./data
```

---

## Running Experiments

### Task 1: LLM Fine-Tuning
Run training across all five variants or individual variants:
```bash
# Run all variants (Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, OPLoRA-16)
python scripts/run_task1.py --variant all --action train

# Run diagnostics and qualitative prompts
python scripts/run_task1.py --action diagnostics
python scripts/run_task1.py --action qualitative

# Run zero-shot benchmark evaluation
python scripts/run_task1.py --action evaluate
```

### Task 2: Vision-Language Models & Visual Grounding
```bash
# Pre-extract and cache SigLIP FP16 patch features
python scripts/run_task2.py --action cache_features

# Train the 3-stage curriculum
python scripts/run_task2.py --action train

# Run 4-condition grounding and VQA evaluations
python scripts/run_task2.py --action evaluate

# (Optional) Run 49-token visual compression ablation
python scripts/run_task2.py --action compress
```

### Task 3: Test-Time Compute
```bash
# Step 1: Evaluate repeated sampling opportunity
python scripts/run_task3.py --action opportunity

# Step 2: Evaluate confidence metrics and RWC consensus
python scripts/run_task3.py --action scoring

# Step 3: Run signal correlations and tokenwise fusion ablation
python scripts/run_task3.py --action signals

# Step 4: Run live generation comparison on 50 MATH questions
python scripts/run_task3.py --action live_gen

# Step 5: Optimize adaptive stopping policies on dev and evaluate on test
python scripts/run_task3.py --action adaptive
```

---

## Implementation Notes for Single-GPU Execution
* **Memory Management**: Task 1 uses a microbatch of 2 with 16 gradient accumulation steps (effective batch 32) and FP16 AMP. Task 2 unloads the SigLIP vision encoder from GPU memory before language model training begins, training with microbatch 8 and accumulation 2 (effective batch 16).
* **Deterministic Seeds**: All operations adhere to fixed seed `6304`.

---

## Attributions & References
* **Models**: `Qwen/Qwen3-0.6B`, `google/siglip-base-patch16-224`, `Qwen/Qwen2.5-3B-Instruct`, and `Qwen/Qwen2.5-VL-3B-Instruct`.
* **Datasets**: MetaMathQA (Yu et al., 2023), GSM8K (Cobbe et al., 2021), MS-COCO 2014 (Lin et al., 2014), VQAv2 (Goyal et al., 2017), MATH (Hendrycks et al., 2021), MathVista (Lu et al., 2024).
* **Methodologies**:
  * LoRA: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
  * OPLoRA: Xiong and Xie, *Orthogonal Projection Low-Rank Adaptation*, 2024.
  * L2-SP: Li et al., *Explicit Inductive Bias for Transfer Learning with Convolutional Networks*, ICML 2018.
  * VisConf: Usman et al., *VisConf: Quality-Aware Consensus for Best-of-N in Vision-Language Model Reasoning*, 2024.
  * Self-Certainty: Kang et al., *Scalable Best-of-N Selection for Large Language Models via Self-Certainty*, 2024.

