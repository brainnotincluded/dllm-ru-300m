#!/usr/bin/env python3
"""Verify server is ready for training."""
import sys

def check_cuda():
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Test allocation
            x = torch.randn(1000, 1000).cuda()
            print("✓ CUDA test passed")
            return True
    except Exception as e:
        print(f"✗ CUDA error: {e}")
        return False

def check_deepspeed():
    try:
        import deepspeed
        print(f"DeepSpeed: {deepspeed.__version__}")
        print("✓ DeepSpeed available")
        return True
    except ImportError:
        print("✗ DeepSpeed not installed")
        return False

def check_flash_attn():
    try:
        import flash_attn
        print(f"FlashAttention: available")
        print("✓ FlashAttention available")
        return True
    except ImportError:
        print("⚠ FlashAttention not installed (optional)")
        return True  # Optional

if __name__ == "__main__":
    print("Server Verification\n" + "="*50)
    
    cuda_ok = check_cuda()
    ds_ok = check_deepspeed()
    flash_ok = check_flash_attn()
    
    print("\n" + "="*50)
    if cuda_ok and ds_ok:
        print("✓ Server ready for training!")
        sys.exit(0)
    else:
        print("✗ Server needs setup")
        sys.exit(1)
