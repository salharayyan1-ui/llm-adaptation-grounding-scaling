import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="CLI Runner for Task 2: VLM & Visual Grounding.")
    parser.add_argument("--config", type=str, default="./configs/task2_config.yaml", help="Path to Task 2 YAML config.")
    parser.add_argument("--action", type=str, choices=["cache_features", "train", "evaluate", "compress", "all"], default="all", help="Action to execute.")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu).")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print(f"Task 2: Vision-Language Models & Visual Grounding")
    print(f"Action: {args.action}")
    print(f"Vision Encoder: {cfg['models']['vision_encoder']} | LM: {cfg['models']['language_model']}")
    print("=" * 60)

    try:
        from task2_vlm.caching import cache_siglip_features
        from task2_vlm.trainer import run_curriculum_training
        from task2_vlm.evaluate import run_grounding_evaluations
        from task2_vlm.compression import run_compression_experiment
    except ImportError as e:
        print(f"[NOTE] PyTorch / HuggingFace environment not detected on this machine: {e}")
        print("To run actual VLM training and evaluation, execute on target GPU machine:")
        print(f"  python scripts/run_task2.py --action {args.action}")
        return

    if args.action in ["cache_features", "all"]:
        print("\n>>> Caching SigLIP patch features in FP16...")
        cache_siglip_features(cfg)

    if args.action in ["train", "all"]:
        print("\n>>> Running 3-stage curriculum training (4,500 total updates)...")
        run_curriculum_training(cfg)

    if args.action in ["evaluate", "all"]:
        print("\n>>> Running grounding and VQA evaluations across 4 conditions...")
        run_grounding_evaluations(cfg)

    if args.action == "compress":
        print("\n>>> Running optional 49-token visual compression experiment...")
        run_compression_experiment(cfg)

if __name__ == "__main__":
    main()
