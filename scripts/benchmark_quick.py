#!/usr/bin/env python3
"""Quick benchmark for RTX 4090 - tests key configurations only."""
import torch
import time
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dllm.model.diffusion import DiffusionLM
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 52256
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    max_position_embeddings: int = 2048
    intermediate_size: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0


def quick_test(batch_size, seq_len, checkpoint, num_iters=5):
    """Quick test of one configuration."""
    config = ModelConfig()
    model = DiffusionLM(config).cuda().to(torch.bfloat16)
    
    if checkpoint:
        # Enable gradient checkpointing
        model.gradient_checkpointing = True
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    dummy = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    
    # Warmup
    for _ in range(2):
        optimizer.zero_grad()
        loss = model(dummy).sum()
        loss.backward()
        optimizer.step()
    
    # Test
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iters):
        optimizer.zero_grad()
        loss = model(dummy).sum()
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return {
        "bs": batch_size,
        "seq": seq_len,
        "checkpoint": checkpoint,
        "time_per_step": elapsed / num_iters,
        "steps_per_sec": num_iters / elapsed,
        "memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


if __name__ == "__main__":
    print("Quick RTX 4090 Benchmark")
    print("="*60)
    
    configs = [
        (1, 2048, True),   # Current config
        (2, 1024, True),   # Smaller seq, larger batch
        (2, 2048, True),   # Try batch 2
        (4, 1024, True),   # Even larger batch
    ]
    
    results = []
    
    for bs, seq, cp in configs:
        try:
            print(f"\nTesting BS={bs}, Seq={seq}, Checkpoint={cp}")
            result = quick_test(bs, seq, cp)
            results.append(result)
            print(f"  Time: {result['time_per_step']:.3f}s/step")
            print(f"  Speed: {result['steps_per_sec']:.2f} steps/sec")
            print(f"  Memory: {result['memory_gb']:.1f} GB")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  [OOM]")
            else:
                raise
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    # Sort by speed
    results.sort(key=lambda x: x['steps_per_sec'], reverse=True)
    
    print(f"{'BS':<5} {'Seq':<6} {'Checkpoint':<12} {'Time(s)':<10} {'Steps/s':<10} {'Memory':<10}")
    print("-"*65)
    
    for r in results:
        print(f"{r['bs']:<5} {r['seq']:<6} {str(r['checkpoint']):<12} "
              f"{r['time_per_step']:<10.3f} {r['steps_per_sec']:<10.2f} "
              f"{r['memory_gb']:<10.1f}")
    
    if results:
        best = results[0]
        print(f"\n✓ BEST CONFIG:")
        print(f"  Batch Size: {best['bs']}")
        print(f"  Sequence: {best['seq']}")
        print(f"  Gradient Checkpointing: {best['checkpoint']}")
        print(f"  Expected Speed: {best['steps_per_sec']:.2f} steps/sec")
        print(f"  Memory: {best['memory_gb']:.1f} GB")
