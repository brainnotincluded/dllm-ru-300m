"""Dataset for DLLM training."""
import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import random


class TextDataset(Dataset):
    """Dataset for text data."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load all documents
        self.documents = []
        for jsonl_file in self.data_path.rglob("*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        self.documents.append(data["text"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        print(f"Loaded {len(self.documents)} documents")
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        text = self.documents[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text, out_type=int)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            # Pad with EOS token
            tokens = tokens + [self.tokenizer.eos_id()] * (self.max_length - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)
