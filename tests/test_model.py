"""Tests for model components."""
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.dllm.model.layers import RMSNorm, Attention
from src.dllm.model.model import DLLMModel
from src.dllm.config import ModelConfig


def test_rmsnorm():
    """Test RMSNorm layer."""
    print("Testing RMSNorm...")
    norm = RMSNorm(hidden_size=512, eps=1e-6)
    x = torch.randn(2, 10, 512)
    out = norm(x)
    
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("✓ RMSNorm test passed")


def test_attention():
    """Test attention layer."""
    print("Testing Attention...")
    config = ModelConfig(hidden_size=512, num_heads=8)
    attn = Attention(
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
    )
    
    x = torch.randn(2, 10, 512)
    out, _ = attn(x)
    
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("✓ Attention test passed")


def test_model_forward():
    """Test full model forward pass."""
    print("Testing DLLMModel forward pass...")
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=512,
        num_layers=2,
        num_heads=8,
    )
    
    model = DLLMModel(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = model(input_ids)
    
    assert output.logits.shape == (2, 10, config.vocab_size), f"Shape mismatch: {output.logits.shape}"
    print("✓ Model forward test passed")


if __name__ == "__main__":
    test_rmsnorm()
    test_attention()
    test_model_forward()
    
    print("\n🎉 All tests passed!")
