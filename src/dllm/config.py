"""Configuration management for DLLM training."""
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 52000
    hidden_size: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    max_position_embeddings: int = 2048
    intermediate_size: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    dropout: float = 0.0


@dataclass
class DiffusionConfig:
    """Diffusion-specific configuration."""
    num_timesteps: int = 1000
    schedule: str = "cosine"  # "cosine", "linear", "learned"
    learned_schedule_dim: int = 256
    flow_matching: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 4
    gradient_accumulation_steps: int = 16
    max_steps: int = 100000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 10
    
    # Data
    train_data_path: str = "data/processed/train"
    eval_data_path: str = "data/processed/eval"
    max_seq_length: int = 2048


@dataclass
class DeepSpeedConfig:
    """DeepSpeed configuration."""
    zero_stage: int = 2
    offload_optimizer: bool = False
    allgather_partitions: bool = True
    allgather_bucket_size: int = 5e8
    overlap_comm: bool = True
    reduce_scatter: bool = True


@dataclass
class Config:
    """Complete configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**data.get('model', {})),
            diffusion=DiffusionConfig(**data.get('diffusion', {})),
            training=TrainingConfig(**data.get('training', {})),
            deepspeed=DeepSpeedConfig(**data.get('deepspeed', {}))
        )
    
    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import yaml
        
        data = {
            'model': self.model.__dict__,
            'diffusion': self.diffusion.__dict__,
            'training': self.training.__dict__,
            'deepspeed': self.deepspeed.__dict__
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
