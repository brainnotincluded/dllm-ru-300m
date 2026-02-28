#!/usr/bin/env python3
"""Preprocess raw data."""
import argparse
import json
from pathlib import Path
from src.dllm.data.preprocess import preprocess_directory


def main():
    parser = argparse.ArgumentParser(description="Preprocess training data")
    parser.add_argument("--input-dir", default="data/raw", help="Input directory with raw data")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Preprocessing data from {args.input_dir} to {args.output_dir}")
    
    stats = preprocess_directory(args.input_dir, args.output_dir)
    
    # Save statistics
    stats_path = Path(args.output_dir) / "preprocessing_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nPreprocessing complete!")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
