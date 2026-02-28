#!/usr/bin/env python3
"""Monitor training progress."""
import json
from pathlib import Path
import sys

def monitor_training(log_dir: str = "outputs/logs"):
    """Monitor training progress."""
    log_path = Path(log_dir) / "training_log.jsonl"
    
    if not log_path.exists():
        print("Training log not found. Training may not have started yet.")
        return
    
    # Read last 20 lines
    with open(log_path, "r") as f:
        lines = f.readlines()
    
    if not lines:
        print("Log file is empty.")
        return
    
    # Parse last entry
    try:
        latest = json.loads(lines[-1])
        step = latest.get("step", 0)
        loss = latest.get("loss", 0)
        accuracy = latest.get("accuracy", 0)
        elapsed = latest.get("elapsed_time", 0)
        
        hours = elapsed / 3600
        progress = step / 100000 * 100  # Assuming 100K steps
        
        print("="*50)
        print("DLLM-RU-300M Training Progress")
        print("="*50)
        print(f"Step:        {step:,} / 100,000")
        print(f"Progress:    {progress:.1f}%")
        print(f"Loss:        {loss:.4f}")
        print(f"Accuracy:    {accuracy:.2%}")
        print(f"Elapsed:     {hours:.1f} hours")
        print("="*50)
        
        # Estimate completion
        if step > 0:
            steps_per_second = step / elapsed if elapsed > 0 else 0
            remaining_steps = 100000 - step
            remaining_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
            remaining_hours = remaining_seconds / 3600
            
            print(f"Steps/sec:   {steps_per_second:.2f}")
            print(f"ETA:         {remaining_hours:.1f} hours ({remaining_hours/24:.1f} days)")
        
        print("\nCheckpoints:")
        checkpoint_dir = Path("outputs/checkpoints")
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
            print(f"   Total: {len(checkpoints)} checkpoints")
            if checkpoints:
                latest_cp = checkpoints[-1]
                size_gb = latest_cp.stat().st_size / (1024**3)
                print(f"   Latest: {latest_cp.name} ({size_gb:.1f} GB)")
        
    except Exception as e:
        print(f"Error reading log: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="outputs/logs")
    args = parser.parse_args()
    
    monitor_training(args.log_dir)
