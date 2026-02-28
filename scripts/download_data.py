#!/usr/bin/env python3
"""Download and prepare training data."""
import os
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_russian_wikipedia(output_dir: str):
    """Download Russian Wikipedia dump."""
    print("Downloading Russian Wikipedia...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    
    output_path = Path(output_dir) / "ru_wikipedia"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save in chunks
    chunk_size = 10000
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset[i:i+chunk_size]
        with open(output_path / f"chunk_{i:06d}.txt", "w", encoding="utf-8") as f:
            for text in chunk["text"]:
                f.write(text + "\n\n")
    
    print(f"Saved {len(dataset)} articles to {output_path}")


def download_russian_common_crawl(output_dir: str, num_samples: int = 1000000):
    """Download Russian Common Crawl sample."""
    print("Downloading Russian Common Crawl...")
    
    # Use mc4 dataset which has Russian web text
    dataset = load_dataset("allenai/c4", "ru", split="train", streaming=True)
    
    output_path = Path(output_dir) / "ru_common_crawl"
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "text.txt", "w", encoding="utf-8") as f:
        for i, example in enumerate(tqdm(dataset, total=num_samples)):
            if i >= num_samples:
                break
            f.write(example["text"] + "\n\n")
    
    print(f"Saved {num_samples} samples to {output_path}")


def download_english_data(output_dir: str, num_samples: int = 500000):
    """Download English SlimPajama sample."""
    print("Downloading English SlimPajama...")
    
    # Use a subset of SlimPajama or similar
    dataset = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
    
    output_path = Path(output_dir) / "en_slimpajama"
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "text.txt", "w", encoding="utf-8") as f:
        for i, example in enumerate(tqdm(dataset, total=num_samples)):
            if i >= num_samples:
                break
            f.write(example["text"] + "\n\n")
    
    print(f"Saved {num_samples} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download training data")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--skip-wiki", action="store_true", help="Skip Wikipedia")
    parser.add_argument("--skip-cc", action="store_true", help="Skip Common Crawl")
    parser.add_argument("--skip-en", action="store_true", help="Skip English data")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.skip_wiki:
        download_russian_wikipedia(args.output_dir)
    
    if not args.skip_cc:
        download_russian_common_crawl(args.output_dir)
    
    if not args.skip_en:
        download_english_data(args.output_dir)
    
    print("Data download complete!")


if __name__ == "__main__":
    main()
