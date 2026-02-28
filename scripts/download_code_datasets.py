#!/usr/bin/env python3
"""Download code datasets for training."""
import os
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_code_dataset(dataset_name: str, output_dir: str, split: str = "train", 
                          max_samples: int = None, text_field: str = "content"):
    """Download a code dataset from Hugging Face."""
    print(f"\n[DOWNLOAD] {dataset_name}...")
    
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"{dataset_name.replace('/', '_')}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            count = 0
            for example in tqdm(dataset, desc=f"Processing {dataset_name}"):
                if max_samples and count >= max_samples:
                    break
                
                # Get code content
                code = example.get(text_field, "")
                if code and len(code) > 100:  # Skip very short snippets
                    f.write(code.strip() + "\n\n<|endoftext|>\n\n")
                    count += 1
        
        print(f"[DONE] Saved {count} code samples to {output_file}")
        return count
        
    except Exception as e:
        print(f"[ERROR] {dataset_name}: {e}")
        return 0


def download_tool_calling_datasets(output_dir: str):
    """Download tool calling and function calling datasets."""
    
    print("\n" + "="*50)
    print("Downloading Tool Calling Datasets...")
    print("="*50)
    
    datasets = [
        # Tool calling datasets
        ("Salesforce/xlam-function-calling-60k", "train", 60000, "text"),
        ("gorilla-llm/Berkeley-Function-Calling-Leaderboard", "train", 10000, "text"),
        ("NousResearch/HERMES-function-calling", "train", 50000, "text"),
        ("TIGER-Lab/ToolBench", "train", 50000, "text"),
        
        # API calling
        ("abacusai/APIBench-Android", "train", None, "text"),
        ("abacusai/APIBench-Javascript", "train", None, "text"),
    ]
    
    total = 0
    for dataset_name, split, max_samples, field in datasets:
        count = download_code_dataset(dataset_name, output_dir, split, max_samples, field)
        total += count
    
    return total


def download_code_datasets(output_dir: str):
    """Download general code datasets."""
    
    print("\n" + "="*50)
    print("Downloading Code Datasets...")
    print("="*50)
    
    datasets = [
        # Python code
        ("codeparrot/github-code", "train", 200000, "code"),
        
        # The Stack (huge code dataset)
        ("bigcode/the-stack-dedup", "train", 300000, "content"),
        
        # CodeContests
        ("deepmind/code_contests", "train", 50000, "solution"),
        
        # CommitPack
        ("bigcode/commitpackft", "train", 100000, "text"),
    ]
    
    total = 0
    for dataset_name, split, max_samples, field in datasets:
        try:
            count = download_code_dataset(dataset_name, output_dir, split, max_samples, field)
            total += count
        except Exception as e:
            print(f"[SKIP] {dataset_name}: {e}")
    
    return total


def main():
    parser = argparse.ArgumentParser(description="Download code and tool calling datasets")
    parser.add_argument("--output-dir", default="data/raw/code", help="Output directory")
    parser.add_argument("--type", choices=["code", "tools", "all"], default="all")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_samples = 0
    
    if args.type in ["code", "all"]:
        total_samples += download_code_datasets(args.output_dir)
    
    if args.type in ["tools", "all"]:
        total_samples += download_tool_calling_datasets(args.output_dir)
    
    print(f"\n[TOTAL] Downloaded: {total_samples} code/tool samples")
    print("[DONE]")


if __name__ == "__main__":
    main()
