import os
import sys
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description="CLI Runner for Task 3: Test-Time Compute.")
    parser.add_argument("--config", type=str, default="./configs/task3_config.yaml", help="Path to Task 3 YAML config.")
    parser.add_argument("--action", type=str, choices=["opportunity", "scoring", "signals", "live_gen", "adaptive", "all"], default="all", help="Action to execute.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("Task 3: Test-Time Compute (Sampling, Selection & Adaptive Inference)")
    print(f"Action: {args.action}")
    print("=" * 60)

    try:
        from task3_test_time.opportunity import run_opportunity_analysis
        from task3_test_time.scoring import run_scoring_evaluations
        from task3_test_time.signals import run_signal_analysis
        from task3_test_time.live_generation import run_live_generation_experiment
        from task3_test_time.adaptive import run_adaptive_inference
    except ImportError as e:
        print(f"[NOTE] Missing package in current environment: {e}")
        return

    if args.action in ["opportunity", "all"]:
        print("\n>>> Step 1: Evaluating opportunity from repeated sampling (budgets 1..32)...")
        run_opportunity_analysis(cfg)

    if args.action in ["scoring", "all"]:
        print("\n>>> Step 2: Evaluating quality scoring and answer consensus (VisConf, RWC)...")
        run_scoring_evaluations(cfg)

    if args.action in ["signals", "all"]:
        print("\n>>> Step 3: Running signal analysis and tokenwise fusion ablation...")
        run_signal_analysis(cfg)

    if args.action in ["live_gen", "all"]:
        print("\n>>> Step 4: Running live generation comparison on 50 MATH questions...")
        run_live_generation_experiment(cfg)

    if args.action in ["adaptive", "all"]:
        print("\n>>> Step 5: Running adaptive inference optimization and held-out evaluation...")
        run_adaptive_inference(cfg)

if __name__ == "__main__":
    main()

