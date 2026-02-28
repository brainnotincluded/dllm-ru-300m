"""Train bilingual tokenizer."""
import os
import json
import argparse
from pathlib import Path
from typing import List
import sentencepiece as spm


def prepare_training_data(input_dir: str, output_file: str, max_samples: int = 1000000):
    """Prepare text file for tokenizer training."""
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    # Collect text from all JSONL files
    for jsonl_file in input_path.rglob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if len(samples) >= max_samples:
                    break
                try:
                    data = json.loads(line)
                    text = data.get("text", "").strip()
                    if len(text) > 50:  # Only use substantial texts
                        samples.append(text)
                except json.JSONDecodeError:
                    continue
        
        if len(samples) >= max_samples:
            break
    
    # Write training text
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample + "\n")
    
    print(f"Prepared {len(samples)} samples for tokenizer training")
    return len(samples)


def train_tokenizer(
    input_file: str,
    output_dir: str,
    vocab_size: int = 32000,
    model_prefix: str = "dllm_bilingual",
    character_coverage: float = 0.9995
):
    """Train SentencePiece tokenizer."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_prefix_path = output_path / model_prefix
    
    # Training arguments
    train_args = {
        "input": input_file,
        "model_prefix": str(model_prefix_path),
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "model_type": "bpe",
        "split_by_whitespace": True,
        "split_by_unicode_script": True,
        "split_by_number": True,
        "max_sentencepiece_length": 16,
        "add_dummy_prefix": True,
        "remove_extra_whitespaces": True,
        "normalization_rule_name": "nmt_nfkc_cf",
        "pad_id": 0,
        "eos_id": 1,
        "unk_id": 2,
        "bos_id": 3,
    }
    
    # Train
    spm.SentencePieceTrainer.train(**train_args)
    
    print(f"Tokenizer trained successfully!")
    print(f"Model: {model_prefix_path}.model")
    print(f"Vocabulary: {model_prefix_path}.vocab")
    
    # Load and display some statistics
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_prefix_path) + ".model")
    
    print(f"\nTokenizer Statistics:")
    print(f"  Vocabulary size: {sp.vocab_size()}")
    print(f"  BOS ID: {sp.bos_id()}")
    print(f"  EOS ID: {sp.eos_id()}")
    print(f"  PAD ID: {sp.pad_id()}")
    print(f"  UNK ID: {sp.unk_id()}")
    
    return str(model_prefix_path) + ".model"


def test_tokenizer(model_path: str):
    """Test the trained tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    test_texts = [
        "Hello, world! This is English text.",
        "Привет, мир! Это русский текст.",
        "Hello мир! Mixed language текст.",
        "print('Hello World')  # Code example",
    ]
    
    print("\nTokenizer Test:")
    for text in test_texts:
        pieces = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        decoded = sp.decode_ids(ids)
        
        print(f"\nText: {text}")
        print(f"Pieces: {pieces[:10]}...")  # Show first 10
        print(f"IDs: {ids[:10]}...")
        print(f"Decoded: {decoded}")
        print(f"Token count: {len(ids)}")


def convert_to_hf_format(sp_model_path: str, output_dir: str):
    """Convert SentencePiece model to HuggingFace format."""
    from transformers import LlamaTokenizer
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load SentencePiece model
    tokenizer = LlamaTokenizer(
        vocab_file=sp_model_path,
        tokenizer_file=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    
    # Save in HuggingFace format
    tokenizer.save_pretrained(output_path)
    
    print(f"\nTokenizer saved to HuggingFace format: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train bilingual tokenizer")
    parser.add_argument("--data-dir", default="data/processed", help="Input data directory")
    parser.add_argument("--output-dir", default="data/tokenizer", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=52256, help="Vocabulary size")
    parser.add_argument("--max-samples", type=int, default=1000000, help="Max training samples")
    
    args = parser.parse_args()
    
    # Prepare training data
    train_file = Path(args.output_dir) / "train_text.txt"
    prepare_training_data(args.data_dir, str(train_file), args.max_samples)
    
    # Train tokenizer
    model_path = train_tokenizer(
        str(train_file),
        args.output_dir,
        vocab_size=args.vocab_size,
        model_prefix="dllm_bilingual"
    )
    
    # Test tokenizer
    test_tokenizer(model_path)
    
    # Convert to HuggingFace format
    convert_to_hf_format(model_path, args.output_dir)
    
    print("\nTokenizer training complete!")


if __name__ == "__main__":
    main()
