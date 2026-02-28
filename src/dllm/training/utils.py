"""Training utilities for DLLM."""
import torch
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import wandb
from torch.utils.tensorboard import SummaryWriter


class MetricsTracker:
    """Track and log training metrics."""
    
    def __init__(self, log_dir: str, use_wandb: bool = True, project_name: str = "dllm-ru-300m"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        self.step = 0
        self.epoch = 0
        
        # TensorBoard
        self.tb_writer = SummaryWriter(self.log_dir / "tensorboard")
        
        # Weights & Biases
        if use_wandb:
            wandb.init(project=project_name, dir=str(self.log_dir))
    
    def log_step(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for a single step."""
        if step is not None:
            self.step = step
        
        # Log to TensorBoard
        for key, value in metrics.items():
            self.tb_writer.add_scalar(key, value, self.step)
        
        # Log to W&B
        if self.use_wandb:
            wandb.log(metrics, step=self.step)
        
        self.step += 1
    
    def log_hyperparams(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self.use_wandb:
            wandb.config.update(params)
    
    def close(self):
        """Close loggers."""
        self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        step: int,
        loss: float,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """Save a checkpoint."""
        checkpoint = {
            "step": step,
            "loss": loss,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:06d}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)
        
        # Remove old checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer=None):
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        step = checkpoint.get("step", 0)
        loss = checkpoint.get("loss", 0.0)
        metrics = checkpoint.get("metrics", {})
        
        print(f"Loaded checkpoint from {checkpoint_path} (step {step}, loss {loss:.4f})")
        
        return step, loss, metrics


class TrainingTimer:
    """Track training time and throughput."""
    
    def __init__(self):
        self.start_time = time.time()
        self.step_times = []
        self.last_step_time = self.start_time
    
    def step(self) -> Dict[str, float]:
        """Record a step and return timing metrics."""
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        
        # Keep only last 100 steps for averaging
        if len(self.step_times) > 100:
            self.step_times.pop(0)
        
        self.last_step_time = current_time
        
        elapsed = current_time - self.start_time
        avg_step_time = sum(self.step_times) / len(self.step_times)
        
        return {
            "step_time": step_time,
            "avg_step_time": avg_step_time,
            "elapsed_time": elapsed,
            "steps_per_second": 1.0 / avg_step_time if avg_step_time > 0 else 0,
        }
    
    def estimate_remaining(self, current_step: int, total_steps: int) -> float:
        """Estimate remaining time in seconds."""
        if len(self.step_times) == 0:
            return 0.0
        
        avg_step_time = sum(self.step_times) / len(self.step_times)
        remaining_steps = total_steps - current_step
        return avg_step_time * remaining_steps
