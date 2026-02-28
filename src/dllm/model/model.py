"""Main diffusion LLM model."""
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from dataclasses import dataclass

from src.dllm.model.layers import RMSNorm, TransformerBlock


@dataclass
class ModelOutput:
    """Model output dataclass."""
    logits: torch.Tensor
    past_key_values: Optional[List[Tuple[torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None


class DLLMModel(nn.Module):
    """Diffusion Language Model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Output projection (for diffusion: predict token logits)
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
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        """Forward pass."""
        batch_size, seq_length = input_ids.shape
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=input_ids.device) * float('-inf'),
                diagonal=1
            )
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        
        # Position IDs
        if position_ids is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values else 0
            position_ids = torch.arange(
                past_length, seq_length + past_length,
                dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0)
        
        # Transformer layers
        all_hidden_states = () if output_hidden_states else None
        next_cache = [] if use_cache else None
        
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                next_cache.append(present_key_value)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Output projection
        logits = self.lm_head(hidden_states)
        
        return ModelOutput(
            logits=logits,
            past_key_values=next_cache if use_cache else None,
            hidden_states=all_hidden_states,
        )
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
