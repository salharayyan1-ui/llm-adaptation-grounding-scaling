import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Dataset and asset preparation helper for ATML PA3.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Root directory for dataset storage.")
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    expected_dirs = [
        os.path.join(args.data_dir, "MetaMathQA"),
        os.path.join(args.data_dir, "coco2014"),
        os.path.join(args.data_dir, "vqav2"),
        os.path.join(args.data_dir, "ATML-PA3-Data"),
        os.path.join(args.data_dir, "ATML-PA3-Data", "Qwen2.5-3B-Instruct"),
        os.path.join(args.data_dir, "ATML-PA3-Data", "Qwen2.5-VL-3B-Instruct"),
    ]
    
    for d in expected_dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Ensured directory exists: {d}")
        
    print("\n" + "=" * 60)
    print("Dataset Preparation Instructions:")
    print("=" * 60)
    print("1. Task 1 (MetaMathQA):")
    print("   MetaMathQA will be downloaded automatically via HuggingFace `datasets`")
    print("   (`load_dataset('meta-math/MetaMathQA')`). The first 40,000 eligible examples")
    print("   will be cached locally.")
    print("\n2. Task 2 (COCO 2014 & VQAv2):")
    print("   Place `train2014` and `val2014` image directories and annotations in:")
    print(f"   - {os.path.join(args.data_dir, 'coco2014')}")
    print(f"   - {os.path.join(args.data_dir, 'vqav2')}")
    print("\n3. Task 3 (ATML-PA3-Data):")
    print("   Extract the supplied rollout archive into:")
    print(f"   - {os.path.join(args.data_dir, 'ATML-PA3-Data')}")
    print("   Ensure the following Parquet files exist:")
    print("   - Qwen2.5-3B-Instruct/generations.parquet")
    print("   - Qwen2.5-3B-Instruct/probability_metrics.parquet")
    print("   - Qwen2.5-VL-3B-Instruct/generations.parquet")
    print("   - Qwen2.5-VL-3B-Instruct/probability_metrics.parquet")
    print("   - Qwen2.5-VL-3B-Instruct/attention_metrics.parquet")
    print("=" * 60)

if __name__ == "__main__":
    main()

