#!/usr/bin/env python3
"""Download simple code datasets."""
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_simple_code(output_dir: str):
    """Download The Stack code dataset."""
    print("\n[DOWNLOAD] bigcode/the-stack-dedup (Python only)...")
    
    try:
        # Load only Python subset
        dataset = load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True
        )
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / "the_stack_python.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            count = 0
            for example in tqdm(dataset, desc="Processing", total=100000):
                if count >= 100000:  # Limit to 100K files
                    break
                
                code = example.get("content", "")
                if code and len(code) > 200:  # Skip very short files
                    # Add language marker
                    f.write(f"<|python|>\n{code.strip()}\n<|endoftext|>\n\n")
                    count += 1
        
        print(f"[DONE] Saved {count} Python files")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        print("[INFO] The Stack requires authentication. Skipping...")


def download_github_code(output_dir: str):
    """Download GitHub code dataset."""
    print("\n[DOWNLOAD] codeparrot/github-code (Python)...")
    
    try:
        dataset = load_dataset(
            "codeparrot/github-code",
            streaming=True,
            split="train"
        )
        
        output_path = Path(output_dir)
        output_file = output_path / "github_python.txt"
        
        with open(output_file, "w", encoding="utf-8") as f:
            count = 0
            for example in tqdm(dataset, desc="Processing", total=50000):
                if count >= 50000:
                    break
                
                code = example.get("code", "")
                lang = example.get("language", "")
                
                # Only Python code
                if lang == "Python" and code and len(code) > 100:
                    f.write(f"<|python|>\n{code.strip()}\n<|endoftext|>\n\n")
                    count += 1
        
        print(f"[DONE] Saved {count} Python files")
        
    except Exception as e:
        print(f"[ERROR] {e}")


def main():
    output_dir = "data/raw/code"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*50)
    print("Downloading Code Datasets")
    print("="*50)
    
    download_github_code(output_dir)
    download_simple_code(output_dir)
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
