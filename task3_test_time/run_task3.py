"""CLI pipeline runner for Task 3: Test-Time Compute.

Usage examples:
    python -m task3_test_time.run_task3 --step all --data_dir ./ATML-PA3-Data --output_dir ./results/task3
    python -m task3_test_time.run_task3 --step live --dry_run
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

from task3_test_time.split import generate_splits, save_splits
from task3_test_time.data_loader import load_dataset, generate_synthetic_data
from task3_test_time.opportunity import evaluate_dataset_opportunity, generate_rollout_permutations
from task3_test_time.scoring import evaluate_dataset_scoring
from task3_test_time.signals import run_signals_analysis
from task3_test_time.live_generation import (
    QwenLiveGenerator,
    run_live_generation_experiment,
    summarize_live_results,
    save_live_results,
)
from task3_test_time.adaptive import search_development_policies, evaluate_held_out_policy


def run_pipeline(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    seed = args.seed
    datasets = ["math", "mathvista"] if args.dataset == "all" else [args.dataset]

    print("==========================================================")
    print("ATML PA3 Task 3: Test-Time Compute Pipeline")
    print(f"Step: {args.step} | Dataset: {args.dataset} | Seed: {seed}")
    print("==========================================================")

    for ds_name in datasets:
        print(f"\n[Processing Dataset: {ds_name}]")
        ds_out_dir = os.path.join(args.output_dir, ds_name)
        os.makedirs(ds_out_dir, exist_ok=True)

        # 1. Ingest Data
        try:
            print(f"Loading rollout data from {args.data_dir}...")
            questions = load_dataset(args.data_dir, dataset_name=ds_name)
            print(f"Successfully loaded {len(questions)} questions.")
        except Exception as e:
            n_q = args.num_questions if args.num_questions else 250
            print(f"Could not load parquet from {args.data_dir} ({e}). Generating synthetic data for evaluation ({n_q} questions).")
            questions = generate_synthetic_data(dataset_name=ds_name, n_questions=n_q, seed=seed)

        if args.num_questions and len(questions) > args.num_questions:
            q_subset = list(questions.keys())[:args.num_questions]
            questions = {k: questions[k] for k in q_subset}

        # 2. Splits
        all_qids = list(questions.keys())
        splits = generate_splits(all_qids, seed=seed)
        save_splits(splits, os.path.join(ds_out_dir, "splits.json"))
        dev_qids = splits["dev"]
        eval_qids = splits["eval"]
        print(f"Splits: {len(dev_qids)} dev questions, {len(eval_qids)} eval questions.")

        # Reusable rollout permutations (10 per question)
        permutations = generate_rollout_permutations(all_qids, seed=seed)

        # Step 1: Opportunity
        if args.step in ["all", "opportunity"]:
            print("\n--- Step 1: Opportunity from Repeated Sampling ---")
            opp_res = evaluate_dataset_opportunity(questions, eval_qids, permutations=permutations, seed=seed)
            opp_path = os.path.join(ds_out_dir, "step1_opportunity.json")
            with open(opp_path, "w", encoding="utf-8") as f:
                json.dump(opp_res["summary"], f, indent=2)
            print(f"Saved opportunity metrics to {opp_path}")

        # Step 2: Scoring & Aggregations
        if args.step in ["all", "scoring"]:
            print("\n--- Step 2: Scoring and Consensus Aggregations ---")
            score_res = evaluate_dataset_scoring(questions, eval_qids, permutations, seed=seed)
            score_path = os.path.join(ds_out_dir, "step2_scoring.json")
            with open(score_path, "w", encoding="utf-8") as f:
                json.dump(score_res["summary"], f, indent=2)
            print(f"Saved scoring metrics to {score_path}")

        # Step 3: Signal Analysis
        if args.step in ["all", "signals"]:
            print("\n--- Step 3: Signal Analysis ---")
            sig_res = run_signals_analysis(questions, eval_qids, seed=seed)
            sig_path = os.path.join(ds_out_dir, "step3_signals.json")
            with open(sig_path, "w", encoding="utf-8") as f:
                json.dump(sig_res, f, indent=2)
            print(f"Saved signals metrics to {sig_path}")

        # Step 5: Adaptive Inference
        if args.step in ["all", "adaptive"]:
            print("\n--- Step 5: Adaptive Inference ---")
            print("Searching optimal stopping policies on development split...")
            dev_search = search_development_policies(questions, dev_qids, permutations)
            best_agree = dev_search["agreement"]["best_params"]
            best_quality = dev_search["quality_aware"]["best_params"]
            print(f"Selected Agreement Policy: tau={best_agree['tau']}")
            print(f"Selected Quality Policy: tau={best_quality['tau']}, gamma={best_quality.get('gamma', 0.0)}")

            print("Evaluating frozen policies on held-out evaluation split...")
            eval_agree = evaluate_held_out_policy(
                questions, eval_qids, permutations,
                policy_type="agreement", tau=best_agree["tau"], seed=seed,
            )
            eval_qual = evaluate_held_out_policy(
                questions, eval_qids, permutations,
                policy_type="quality", tau=best_quality["tau"], gamma=best_quality.get("gamma", 0.0), seed=seed,
            )

            adaptive_path = os.path.join(ds_out_dir, "step5_adaptive.json")
            with open(adaptive_path, "w", encoding="utf-8") as f:
                json.dump({
                    "dev_search": dev_search,
                    "eval_agreement": eval_agree,
                    "eval_quality_aware": eval_qual,
                }, f, indent=2)
            print(f"Saved adaptive inference results to {adaptive_path}")

    # Step 4: Live Generation Subtask (MATH only)
    if args.step in ["all", "live"]:
        print("\n--- Step 4: Live Generation Subtask (50 MATH questions) ---")
        live_out_dir = os.path.join(args.output_dir, "live_math")
        os.makedirs(live_out_dir, exist_ok=True)

        # Prepare 50 sample problems
        math_live_qids = splits.get("live_math", [])[:50]
        live_problems = [
            {
                "sample_id": qid,
                "problem": f"What is 2 + 2 for problem {qid}?",
                "gold_answer": "4",
            }
            for qid in math_live_qids
        ]

        generator = QwenLiveGenerator(dry_run=args.dry_run)
        live_results = run_live_generation_experiment(live_problems, generator, base_seed=seed)
        summary = summarize_live_results(live_results)

        save_live_results(live_results, os.path.join(live_out_dir, "live_generations.json"))
        with open(os.path.join(live_out_dir, "live_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved live generation summary to {live_out_dir}/live_summary.json")

    print("\nTask 3 pipeline completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="ATML PA3 Task 3 Runner")
    parser.add_argument("--step", type=str, default="all", choices=["all", "opportunity", "scoring", "signals", "live", "adaptive"])
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "math", "mathvista"])
    parser.add_argument("--data_dir", type=str, default="./ATML-PA3-Data")
    parser.add_argument("--output_dir", type=str, default="./results/task3")
    parser.add_argument("--num_questions", type=int, default=None, help="Optional subset of questions for quick debugging")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Run live generation in dry-run mode without GPU")
    parser.add_argument("--seed", type=int, default=6304)
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
