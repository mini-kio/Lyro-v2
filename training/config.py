from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import os
from pathlib import Path


@dataclass
class GeneratorConfig:
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    d_ff: int = 4096
    d_state: int = 64
    latent_channels: int = 16
    latent_time_steps: int = 128
    flow_steps: int = 50
    cfg_scale: float = 7.5
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 100
    dropout: float = 0.1


@dataclass
class LossConfig:
    flow_matching_weight: float = 1.0
    use_perceptual_loss: bool = False
    perceptual_weight: float = 0.2


@dataclass
class DataConfig:
    dataset_root: str = "dataset/"
    train_metadata: str = "dataset/metadata/train_metadata.jsonl"
    val_metadata: str = "dataset/metadata/val_metadata.jsonl"
    latent_channels: int = 16
    latent_time_steps: int = 128
    max_lyrics_length: int = 256
    num_workers: int = 4
    batch_size: int = 4


@dataclass
class EncoderConfig:
    text_embed_dim: int = 768
    text_hidden_dim: int = 768
    text_num_layers: int = 8
    text_num_heads: int = 12
    text_max_length: int = 256
    pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class TrainingConfig:
    checkpoint_dir: str = "checkpoints/"
    save_interval: int = 10
    eval_interval: int = 5
    device: str = "auto"
    seed: int = 42
    
    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class LyroConfig:
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def validate(self) -> Dict[str, Any]:
        return {
            'valid': True,
            'issues': [],
            'warnings': []
        }


def get_config() -> LyroConfig:
    return LyroConfig()


def create_optimized_config(
    available_memory_gb: float = 14.0,
    target_quality: str = "high"
) -> LyroConfig:
    config = LyroConfig()
    if available_memory_gb < 16:
        config.generator.batch_size = 2
        config.generator.d_model = 768
        config.generator.n_layers = 12
    return config