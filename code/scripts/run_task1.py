"""CLI runner for Task 1 (LLM fine-tuning, retention analysis).

Examples (one 24 GB GPU):
    python scripts/run_task1.py --action train --variant full_ft
    python scripts/run_task1.py --action evaluate --conditions baseline,full_ft_final
    python scripts/run_task1.py --action diagnostics
    python scripts/run_task1.py --action qualitative
    python scripts/run_task1.py --action aggregate
    python scripts/run_task1.py --action offline_analysis   # CPU only: extraction sensitivity, paired CIs, BoolQ bias
Two GPUs: use scripts/launch_multi_gpu.py --task 1 (runs independent jobs in parallel), or pass --gpu N.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VARIANTS = ["full_ft", "full_ft_l2_sp", "lora_4", "lora_16", "oplora_16"]


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI runner for Task 1: LLM fine-tuning.")
    parser.add_argument("--config", default="./configs/task1_config.yaml")
    parser.add_argument("--action", required=True,
                        choices=["train", "evaluate", "diagnostics", "qualitative", "aggregate", "offline_analysis"])
    parser.add_argument("--variant", default="all", choices=VARIANTS + ["all"], help="train: variant(s) to run")
    parser.add_argument("--conditions", default=None,
                        help="evaluate/qualitative/diagnostics: comma-separated conditions (default: all)")
    parser.add_argument("--resume_from", default=None, help="train: checkpoint directory to resume from")
    parser.add_argument("--gpu", type=int, default=None, help="Physical GPU index (sets CUDA_VISIBLE_DEVICES)")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    conditions = args.conditions.split(",") if args.conditions else None

    # Import errors must surface (they used to be swallowed and reported as a missing environment).
    if args.action == "train":
        from task1_llm_ft.trainer import run_task1_training
        variants = VARIANTS if args.variant == "all" else [args.variant]
        if args.resume_from and len(variants) != 1:
            parser.error("--resume_from needs a single --variant")
        for var in variants:
            print(f"\n>>> Training variant: {var}")
            run_task1_training(var, cfg, resume_from=args.resume_from)
    elif args.action == "evaluate":
        from task1_llm_ft.evaluate import run_task1_evaluations
        run_task1_evaluations(cfg, conditions=conditions, aggregate=conditions is None)
    elif args.action == "qualitative":
        from task1_llm_ft.evaluate import run_qualitative_suite
        print(run_qualitative_suite(cfg, conditions=conditions))
    elif args.action == "diagnostics":
        from task1_llm_ft.diagnostics import run_task1_diagnostics
        run_task1_diagnostics(cfg, conditions=conditions)
    elif args.action == "aggregate":
        from task1_llm_ft.evaluate import aggregate_benchmark_results, aggregate_qualitative
        aggregate_benchmark_results(cfg)
        print(aggregate_qualitative(cfg))
    elif args.action == "offline_analysis":
        from task1_llm_ft.offline_analysis import run_offline_analysis
        res = run_offline_analysis(cfg)
        print(f"wrote {os.path.join(cfg.get('results_dir', './results/task1'), 'offline_analysis.json')} "
              f"for conditions: {res['settings']['conditions']}")


if __name__ == "__main__":
    main()
