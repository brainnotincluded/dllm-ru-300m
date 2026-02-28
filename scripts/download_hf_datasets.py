#!/usr/bin/env python3
"""Download datasets from Hugging Face and Kaggle for training."""
import os
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_hf_dataset(dataset_name: str, output_dir: str, split: str = "train", max_samples: int = None):
    """Download a Hugging Face dataset."""
    print(f"\n📥 Downloading {dataset_name}...")
    
    try:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        output_file = output_path / f"{dataset_name.replace('/', '_')}.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            count = 0
            for example in tqdm(dataset, desc=f"Processing {dataset_name}"):
                if max_samples and count >= max_samples:
                    break
                
                # Try different text fields
                text = None
                for field in ["text", "content", "article", "sentence", "context"]:
                    if field in example and isinstance(example[field], str):
                        text = example[field]
                        break
                
                if text and len(text) > 50:
                    f.write(text.strip() + "\n\n")
                    count += 1
        
        print(f"✅ Saved {count} samples to {output_file}")
        return count
        
    except Exception as e:
        print(f"❌ Error downloading {dataset_name}: {e}")
        return 0


def download_russian_datasets(output_dir: str):
    """Download Russian datasets from Hugging Face."""
    
    datasets_to_download = [
        # Russian Wikipedia
        ("DataSynGen/RUwiki", "train", None),
        
        # Russian pre-training corpus
        ("NotEvilAI/ruwiki-pretrain-20260102", "train", 100000),
        
        # Russian news
        ("NotEvilAI/ruwikinews-pretrain-20260102", "train", 50000),
        
        # Python code in Russian
        ("DataSynGen/Python_code_RU", "train", None),
    ]
    
    total_samples = 0
    
    for dataset_name, split, max_samples in datasets_to_download:
        count = download_hf_dataset(dataset_name, output_dir, split, max_samples)
        total_samples += count
    
    print(f"\n🎉 Total samples downloaded: {total_samples}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for training")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--source", choices=["huggingface", "kaggle", "all"], default="all")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.source in ["huggingface", "all"]:
        print("\n" + "="*50)
        print("Downloading from Hugging Face...")
        print("="*50)
        download_russian_datasets(args.output_dir)
    
    if args.source in ["kaggle", "all"]:
        print("\n" + "="*50)
        print("Kaggle datasets require manual download.")
        print("Visit: https://www.kaggle.com/datasets?search=russian+text")
        print("="*50)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
