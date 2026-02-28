#!/usr/bin/env python3
"""Main training script for DLLM."""
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sentencepiece as spm

from src.dllm.model.diffusion import DiffusionLM, DiffusionTrainer
from src.dllm.data.dataset import TextDataset
from src.dllm.training.utils import MetricsTracker, CheckpointManager, TrainingTimer


def parse_args():
    parser = argparse.ArgumentParser(description="Train DLLM")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--tokenizer-path", type=str, default="data/tokenizer/dllm_bilingual.model", help="Tokenizer path")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--full-model", action="store_true", help="Use full 300M model (default: small test model)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer first to get vocab size
    print("Loading tokenizer...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer_path)
    vocab_size = tokenizer.vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Model config
    from dataclasses import dataclass
    
    @dataclass
    class ModelConfig:
        vocab_size: int = 52256
        hidden_size: int = 1024 if args.full_model else 512
        num_layers: int = 16 if args.full_model else 8  # Reduced from 24 to 16 for speed
        num_heads: int = 16 if args.full_model else 8
        max_position_embeddings: int = 2048
        intermediate_size: int = 4096 if args.full_model else 2048
        rms_norm_eps: float = 1e-6
        rope_theta: float = 10000.0
        dropout: float = 0.0
        gradient_checkpointing: bool = True
    
    config = ModelConfig(vocab_size=vocab_size)
    
    model_size = "300M" if args.full_model else "37M (test)"
    print(f"\nModel configuration: {model_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Vocab size: {config.vocab_size}")
    
    # Initialize model
    print("Initializing model...")
    model = DiffusionLM(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    
    # Load dataset
    print("Loading dataset...")
    max_length = 2048 if args.full_model else 512
    dataset = TextDataset(args.data_dir, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Dataset size: {len(dataset)}")
    
    # Gradient accumulation
    gradient_accumulation_steps = 4  # Effective batch size = batch_size * 4
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
    )
    
    # Initialize trainer
    trainer = DiffusionTrainer(model)
    
    # Initialize tracking
    metrics_tracker = MetricsTracker(log_dir=str(output_dir / "logs"))
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(output_dir / "checkpoints"),
        max_checkpoints=5,
    )
    timer = TrainingTimer()
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        start_step, _, _ = checkpoint_manager.load_checkpoint(args.resume, model, optimizer)
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("="*50 + "\n")
    
    model.train()
    step = start_step
    
    while step < args.max_steps:
        for batch_idx, batch in enumerate(dataloader):
            if step >= args.max_steps:
                break
            
            # Move batch to device
            batch = batch.to(device)
            
            # Training step with gradient accumulation
            loss, metrics = trainer.training_step(batch)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Gradient clipping and optimizer step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # Log metrics
            timing = timer.step()
            metrics.update(timing)
            metrics["loss"] = loss.item()
            metrics_tracker.log_step(metrics, step)
            
            # Print progress
            if step % 10 == 0:
                print(f"Step {step:5d} | Loss: {loss.item():.4f} | "
                      f"Acc: {metrics.get('accuracy', 0):.2%} | "
                      f"Time: {timing['step_time']:.2f}s | "
                      f"Steps/s: {timing['steps_per_second']:.2f}")
            
            # Save checkpoint every 5000 steps (reduced from 100 to avoid slowdown)
            if step % 5000 == 0 and step > 0:
                print(f"Saving checkpoint at step {step}...")
                checkpoint_manager.save_checkpoint(
                    model, optimizer, step, loss.item(), metrics, is_best=False
                )
            
            step += 1
    
    # Save final checkpoint
    checkpoint_manager.save_checkpoint(
        model, optimizer, step, loss.item(), metrics, is_best=True
    )
    
    print(f"\nTraining complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total steps: {step}")


if __name__ == "__main__":
    main()
