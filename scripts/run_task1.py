import os
import sys
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description="CLI Runner for Task 1: LLM Fine-Tuning.")
    parser.add_argument("--config", type=str, default="./configs/task1_config.yaml", help="Path to Task 1 YAML config.")
    parser.add_argument("--variant", type=str, choices=["full_ft", "full_ft_l2_sp", "lora_4", "lora_16", "oplora_16", "all"], default="all", help="Fine-tuning variant to train/evaluate.")
    parser.add_argument("--action", type=str, choices=["train", "evaluate", "diagnostics", "qualitative", "all"], default="all", help="Action to execute.")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu).")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print(f"Task 1: Large Language Model Fine-Tuning")
    print(f"Variant: {args.variant} | Action: {args.action}")
    print(f"Model: {cfg['model']['name_or_path']} (Seed: {cfg['model']['seed']})")
    print("=" * 60)

    try:
        from task1_llm_ft.trainer import run_task1_training
        from task1_llm_ft.diagnostics import run_task1_diagnostics
        from task1_llm_ft.evaluate import run_task1_evaluations, run_qualitative_suite
    except ImportError as e:
        print(f"[NOTE] PyTorch / HuggingFace environment not detected on this machine: {e}")
        print("To run actual fine-tuning and evaluation, execute on target GPU machine:")
        print(f"  python scripts/run_task1.py --variant {args.variant} --action {args.action}")
        return

    variants = [args.variant] if args.variant != "all" else ["full_ft", "full_ft_l2_sp", "lora_4", "lora_16", "oplora_16"]

    if args.action in ["train", "all"]:
        for var in variants:
            print(f"\n>>> Starting training for variant: {var}...")
            run_task1_training(var, cfg)

    if args.action in ["diagnostics", "all"]:
        print("\n>>> Running weight-space and activation diagnostics...")
        run_task1_diagnostics(cfg)

    if args.action in ["evaluate", "all"]:
        print("\n>>> Running quantitative zero-shot evaluations...")
        run_task1_evaluations(cfg)

    if args.action in ["qualitative", "all"]:
        print("\n>>> Running qualitative evaluations on first_plot_questions.yaml...")
        run_qualitative_suite(cfg)

if __name__ == "__main__":
    main()

