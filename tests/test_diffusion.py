"""Tests for diffusion components."""
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.dllm.model.diffusion import (
    DiffusionLM,
    DiffusionTrainer,
    DiffusionSampler,
    compute_diffusion_loss,
    mask_tokens_for_diffusion,
)
from src.dllm.model.model import DLLMModel
from src.dllm.config import ModelConfig


def test_mask_tokens():
    """Test token masking for diffusion training."""
    print("Testing mask_tokens_for_diffusion...")
    
    batch_size, seq_len = 2, 10
    vocab_size = 100
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Mask 50% of tokens
    masked_ids, mask = mask_tokens_for_diffusion(input_ids, mask_ratio=0.5, vocab_size=vocab_size)
    
    # Check mask ratio approximately correct (allow small variance)
    mask_ratio = mask.float().mean().item()
    assert 0.3 < mask_ratio < 0.7, f"Mask ratio {mask_ratio} not in expected range"
    
    # Check masked positions have mask token
    mask_token_id = vocab_size - 1  # Last token is [MASK]
    masked_positions = masked_ids == mask_token_id
    assert masked_positions.sum() == mask.sum(), "Mask positions don't match"
    
    # Check unmasked positions unchanged
    unmasked = ~mask
    assert torch.all(masked_ids[unmasked] == input_ids[unmasked]), "Unmasked tokens changed"
    
    print(f"✓ Mask tokens test passed (mask ratio: {mask_ratio:.2f})")


def test_diffusion_loss():
    """Test diffusion loss computation."""
    print("Testing compute_diffusion_loss...")
    
    batch_size, seq_len = 2, 10
    vocab_size = 100
    config = ModelConfig(vocab_size=vocab_size, hidden_size=64, num_layers=2, num_heads=4)
    model = DiffusionLM(config)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Compute loss
    loss, metrics = compute_diffusion_loss(model, input_ids, mask_ratio=0.5)
    
    # Check loss is valid
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    assert 'accuracy' in metrics, "Missing accuracy metric"
    assert 0 <= metrics['accuracy'] <= 1, f"Accuracy {metrics['accuracy']} out of range"
    
    print(f"✓ Diffusion loss test passed (loss: {loss.item():.4f}, acc: {metrics['accuracy']:.2%})")


def test_diffusion_lm_forward():
    """Test DiffusionLM forward pass."""
    print("Testing DiffusionLM forward...")
    
    batch_size, seq_len = 2, 10
    vocab_size = 100
    config = ModelConfig(vocab_size=vocab_size, hidden_size=64, num_layers=2, num_heads=4)
    
    diffusion_lm = DiffusionLM(config)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass (no mask_ratio - that's handled in compute_diffusion_loss)
    logits = diffusion_lm(input_ids)
    
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Shape mismatch: {logits.shape}"
    print("✓ DiffusionLM forward test passed")


def test_diffusion_trainer():
    """Test DiffusionTrainer."""
    print("Testing DiffusionTrainer...")
    
    batch_size, seq_len = 2, 10
    vocab_size = 100
    config = ModelConfig(vocab_size=vocab_size, hidden_size=64, num_layers=2, num_heads=4)
    model = DiffusionLM(config)
    
    trainer = DiffusionTrainer(model)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Training step
    loss, metrics = trainer.training_step(input_ids)
    
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    
    print(f"✓ DiffusionTrainer test passed (loss: {loss.item():.4f})")


def test_gradient_flow():
    """Test that gradients flow properly."""
    print("Testing gradient flow...")
    
    batch_size, seq_len = 2, 10
    vocab_size = 100
    config = ModelConfig(vocab_size=vocab_size, hidden_size=64, num_layers=2, num_heads=4)
    model = DiffusionLM(config)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward
    logits = model(input_ids)
    
    # Loss
    loss = logits.sum()
    loss.backward()
    
    # Check gradients
    has_grad = False
    max_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            max_grad = max(max_grad, param.grad.abs().max().item())
        else:
            print(f"  Warning: No gradient for {name}")
    
    assert has_grad, "No gradients found"
    assert max_grad > 0, "All gradients are zero"
    
    print(f"✓ Gradient flow test passed (max grad: {max_grad:.4f})")


if __name__ == "__main__":
    test_mask_tokens()
    test_diffusion_loss()
    test_diffusion_lm_forward()
    test_diffusion_trainer()
    test_gradient_flow()
    
    print("\n🎉 All diffusion tests passed!")
