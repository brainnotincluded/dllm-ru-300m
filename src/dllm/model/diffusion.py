"""Diffusion training and sampling with LLaDA-style logit-space approach."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

from src.dllm.model.layers import RMSNorm, TransformerBlock


class DiffusionLM(nn.Module):
    """
    Diffusion Language Model (LLaDA-style).
    
    Works directly in token logit space:
    - Mask tokens randomly during training
    - Model predicts logits for all positions simultaneously
    - Loss is cross-entropy on masked positions
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Output projection (logits over vocabulary)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len] Token IDs (with [MASK] tokens)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            
        Returns:
            logits: [batch, seq_len, vocab_size] Logits for each position
        """
        batch_size, seq_length = input_ids.shape
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask (allow attending to all positions including masked ones)
        if attention_mask is None:
            # Standard causal mask for autoregressive structure
            # But since we predict all at once, we can use bidirectional attention
            attention_mask = None  # Allow full attention for diffusion
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Transformer layers
        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Output projection (logits)
        logits = self.lm_head(hidden_states)
        
        return logits


def mask_tokens_for_diffusion(
    input_ids: torch.LongTensor,
    mask_ratio: float = 0.5,
    vocab_size: int = 52256,
    mask_token_id: Optional[int] = None,
) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    """
    Mask tokens for diffusion training.
    
    Args:
        input_ids: [batch, seq_len] Input token IDs
        mask_ratio: Fraction of tokens to mask (0.0 to 1.0)
        vocab_size: Total vocabulary size
        mask_token_id: ID for [MASK] token (default: vocab_size - 1)
        
    Returns:
        masked_ids: [batch, seq_len] Input with masked tokens
        mask: [batch, seq_len] Boolean mask (True = masked)
    """
    batch_size, seq_len = input_ids.shape
    
    # Default mask token is last in vocabulary
    if mask_token_id is None:
        mask_token_id = vocab_size - 1
    
    # Create random mask
    mask = torch.rand(batch_size, seq_len, device=input_ids.device) < mask_ratio
    
    # Apply mask
    masked_ids = input_ids.clone()
    masked_ids[mask] = mask_token_id
    
    return masked_ids, mask


def compute_diffusion_loss(
    model: DiffusionLM,
    input_ids: torch.LongTensor,
    mask_ratio: float = 0.5,
    vocab_size: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute diffusion training loss.
    
    Args:
        model: DiffusionLM model
        input_ids: [batch, seq_len] Input token IDs
        mask_ratio: Fraction of tokens to mask
        vocab_size: Vocabulary size (defaults to model.config.vocab_size)
        
    Returns:
        loss: Scalar loss tensor
        metrics: Dictionary with metrics (accuracy, etc.)
    """
    if vocab_size is None:
        vocab_size = model.config.vocab_size
    
    # Mask tokens
    masked_ids, mask = mask_tokens_for_diffusion(input_ids, mask_ratio, vocab_size)
    
    # Forward pass
    logits = model(masked_ids)  # [batch, seq_len, vocab_size]
    
    # Compute loss only on masked positions
    # Reshape for cross-entropy
    logits_masked = logits[mask]  # [num_masked, vocab_size]
    targets = input_ids[mask]  # [num_masked]
    
    if logits_masked.numel() == 0:
        # No masked tokens (edge case)
        return torch.tensor(0.0, device=input_ids.device), {'accuracy': 0.0}
    
    # Cross-entropy loss
    loss = F.cross_entropy(logits_masked, targets)
    
    # Compute accuracy
    predictions = logits_masked.argmax(dim=-1)
    accuracy = (predictions == targets).float().mean().item()
    
    metrics = {
        'accuracy': accuracy,
        'num_masked': mask.sum().item(),
    }
    
    return loss, metrics


class DiffusionTrainer:
    """Trainer for diffusion language model."""
    
    def __init__(
        self,
        model: DiffusionLM,
        mask_ratio: float = 0.5,
        mask_ratio_schedule: str = "constant",  # "constant", "linear", "cosine"
    ):
        self.model = model
        self.mask_ratio = mask_ratio
        self.mask_ratio_schedule = mask_ratio_schedule
        self.step_count = 0
    
    def get_mask_ratio(self) -> float:
        """Get current mask ratio (may vary during training)."""
        if self.mask_ratio_schedule == "constant":
            return self.mask_ratio
        elif self.mask_ratio_schedule == "linear":
            # Start at 0.3, ramp up to mask_ratio
            return min(0.3 + self.step_count * 0.0001, self.mask_ratio)
        elif self.mask_ratio_schedule == "cosine":
            # Cosine schedule from 0.1 to mask_ratio
            import math
            progress = min(self.step_count / 10000, 1.0)
            return 0.1 + (self.mask_ratio - 0.1) * (1 - math.cos(progress * math.pi)) / 2
        else:
            return self.mask_ratio
    
    def training_step(self, input_ids: torch.LongTensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step."""
        mask_ratio = self.get_mask_ratio()
        loss, metrics = compute_diffusion_loss(self.model, input_ids, mask_ratio)
        
        metrics['mask_ratio'] = mask_ratio
        self.step_count += 1
        
        return loss, metrics


class DiffusionSampler:
    """Sampler for generating text from diffusion model."""
    
    def __init__(self, model: DiffusionLM, num_steps: int = 10):
        self.model = model
        self.num_steps = num_steps
        self.mask_token_id = model.config.vocab_size - 1
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_length: int,
        device: torch.device,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples using iterative denoising.
        
        Args:
            batch_size: Number of sequences to generate
            seq_length: Length of each sequence
            device: Device to generate on
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            samples: [batch_size, seq_length] Generated token IDs
        """
        # Start with all masked
        samples = torch.full(
            (batch_size, seq_length),
            self.mask_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Iteratively unmask tokens
        num_masked = seq_length
        for step in range(self.num_steps):
            # Forward pass
            logits = self.model(samples)  # [batch, seq_len, vocab]
            
            # Apply temperature
            logits = logits / temperature
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, probs.size(-1)),
                num_samples=1
            ).view(batch_size, seq_length)
            
            # Find currently masked positions
            mask = samples == self.mask_token_id
            
            # Update masked positions with sampled tokens
            samples = torch.where(mask, sampled, samples)
            
            # Count remaining masked tokens
            num_masked = mask.sum().item()
            if num_masked == 0:
                break
        
        return samples
    
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate continuation from prompt.
        
        Args:
            prompt_ids: [batch, seq_len] Prompt token IDs
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            output_ids: [batch, seq_len + max_new_tokens] Generated tokens
        """
        batch_size, prompt_len = prompt_ids.shape
        device = prompt_ids.device
        vocab_size = self.model.config.vocab_size
        
        # Initialize with prompt + masked tokens
        total_len = prompt_len + max_new_tokens
        samples = torch.full(
            (batch_size, total_len),
            self.mask_token_id,
            dtype=torch.long,
            device=device
        )
        samples[:, :prompt_len] = prompt_ids
        
        # Iteratively unmask
        for step in range(self.num_steps):
            logits = self.model(samples)
            probs = F.softmax(logits / temperature, dim=-1)
            
            # Only sample from positions after prompt
            for i in range(batch_size):
                for pos in range(prompt_len, total_len):
                    if samples[i, pos] == self.mask_token_id:
                        token = torch.multinomial(probs[i, pos], num_samples=1)
                        samples[i, pos] = token
            
            # Check if done
            if (samples[:, prompt_len:] != self.mask_token_id).all():
                break
        
        return samples
