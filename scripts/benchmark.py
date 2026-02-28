#!/usr/bin/env python3
"""Benchmark different training configurations for RTX 4090."""
import torch
import time
import json
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dllm.model.diffusion import DiffusionLM
from src.dllm.model.model import DLLMModel
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


def benchmark_config(
    batch_size: int,
    seq_length: int = 2048,
    use_gradient_checkpointing: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    num_iterations: int = 10,
):
    """Benchmark a specific configuration."""
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: BS={batch_size}, Seq={seq_length}, "
          f"Checkpoint={use_gradient_checkpointing}, Dtype={dtype}")
    print(f"{'='*60}")
    
    config = ModelConfig()
    model = DiffusionLM(config)
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Enable gradient checkpointing if requested
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Convert to appropriate dtype
    if dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)
    elif dtype == torch.float16:
        model = model.to(torch.float16)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Generate dummy data
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    
    try:
        for _ in range(3):
            optimizer.zero_grad()
            logits = model(dummy_input)
            loss = logits.sum()
            loss.backward()
            optimizer.step()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"[OOM] Configuration failed - Out of memory")
            return None
        raise
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(dummy_input)
        loss = logits.sum()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if i == num_iterations - 1:
            torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Collect metrics
    elapsed = end_time - start_time
    avg_time = elapsed / num_iterations
    throughput = batch_size * num_iterations / elapsed
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters()) / 1e6  # Millions
    
    results = {
        "batch_size": batch_size,
        "seq_length": seq_length,
        "gradient_checkpointing": use_gradient_checkpointing,
        "dtype": str(dtype),
        "avg_step_time": avg_time,
        "throughput": throughput,
        "peak_memory_gb": peak_memory,
        "total_params_m": total_params,
        "steps_per_second": 1.0 / avg_time,
    }
    
    print(f"Average step time: {avg_time:.3f}s")
    print(f"Throughput: {throughput:.1f} tokens/sec")
    print(f"Peak memory: {peak_memory:.1f} GB")
    print(f"Steps/sec: {1.0/avg_time:.2f}")
    
    return results


def run_full_benchmark(output_file: str = "benchmark_results.json"):
    """Run comprehensive benchmark."""
    
    print("="*60)
    print("RTX 4090 Training Configuration Benchmark")
    print("="*60)
    
    device_props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {device_props.name}")
    print(f"VRAM: {device_props.total_memory / 1024**3:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    
    configs_to_test = [
        # (batch_size, seq_length, gradient_checkpointing, dtype)
        (1, 2048, False, torch.bfloat16),
        (1, 2048, True, torch.bfloat16),
        (2, 1024, False, torch.bfloat16),
        (2, 1024, True, torch.bfloat16),
        (2, 2048, True, torch.bfloat16),
        (4, 1024, True, torch.bfloat16),
        (4, 1024, True, torch.float16),
        (1, 4096, True, torch.bfloat16),
    ]
    
    results = []
    
    for batch_size, seq_len, checkpoint, dtype in configs_to_test:
        result = benchmark_config(batch_size, seq_len, checkpoint, dtype)
        if result:
            results.append(result)
    
    # Save results
    output_path = Path(output_file)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Benchmark Results Summary")
    print(f"{'='*60}")
    
    # Sort by throughput
    results_sorted = sorted(results, key=lambda x: x["steps_per_second"], reverse=True)
    
    print("\nTop 5 Configurations by Speed:")
    print(f"{'Rank':<5} {'BS':<4} {'Seq':<6} {'Checkpoint':<12} {'Time(s)':<10} {'Steps/s':<10} {'Memory(GB)':<12}")
    print("-" * 75)
    
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"{i:<5} {r['batch_size']:<4} {r['seq_length']:<6} "
              f"{str(r['gradient_checkpointing']):<12} {r['avg_step_time']:<10.3f} "
              f"{r['steps_per_second']:<10.2f} {r['peak_memory_gb']:<12.1f}")
    
    # Find optimal config (best balance of speed and memory)
    optimal = None
    for r in results_sorted:
        if r["peak_memory_gb"] < 22:  # Leave some headroom
            optimal = r
            break
    
    if optimal:
        print(f"\n{'='*60}")
        print("RECOMMENDED CONFIGURATION")
        print(f"{'='*60}")
        print(f"Batch Size: {optimal['batch_size']}")
        print(f"Sequence Length: {optimal['seq_length']}")
        print(f"Gradient Checkpointing: {optimal['gradient_checkpointing']}")
        print(f"Dtype: {optimal['dtype']}")
        print(f"Expected Speed: {optimal['steps_per_second']:.2f} steps/sec")
        print(f"Memory Usage: {optimal['peak_memory_gb']:.1f} GB")
        print(f"{'='*60}")
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/benchmark_results.json")
    args = parser.parse_args()
    
    run_full_benchmark(args.output)
