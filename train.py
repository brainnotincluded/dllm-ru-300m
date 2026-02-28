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
from transformers import get_cosine_schedule_with_warmup

from src.dllm.config import Config
from src.dllm.model.diffusion import DiffusionLM, DiffusionTrainer
from src.dllm.training.utils import MetricsTracker, CheckpointManager, TrainingTimer


def parse_args():
    parser = argparse.ArgumentParser(description="Train DLLM")
    parser.add_argument("--config", type=str, default="configs/dllm_300m.yaml", help="Config file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    print(f"Loaded config from {args.config}")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    print("Initializing model...")
    model = DiffusionLM(config.model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=config.training.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.max_steps,
    )
    
    # Initialize trainer
    trainer = DiffusionTrainer(model)
    
    # Initialize tracking
    metrics_tracker = MetricsTracker(
        log_dir=str(output_dir / "logs"),
        use_wandb=True,
    )
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(output_dir / "checkpoints"),
        max_checkpoints=10,
    )
    timer = TrainingTimer()
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        start_step, _, _ = checkpoint_manager.load_checkpoint(args.resume, model, optimizer)
    
    # Training loop (simplified - without actual data for now)
    print("\nStarting training...")
    print(f"Max steps: {config.training.max_steps}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # TODO: Add data loading
    print("\nNote: Training data not yet loaded. Add data loading before running full training.")
    print("To test the training loop, use dummy data or implement data pipeline first.")
    
    # Save initial checkpoint
    checkpoint_manager.save_checkpoint(
        model, optimizer, start_step, 0.0, {}, is_best=False
    )
    
    print(f"\nTraining setup complete!")
    print(f"Output directory: {output_dir}")
    print(f"Ready to train once data pipeline is implemented.")


if __name__ == "__main__":
    main()
