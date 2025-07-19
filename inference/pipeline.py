import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import time

from models import MultimodalLyroSystem, GeneratorConfig


@dataclass
class GenerationConfig:
    duration: float = 10.0
    num_steps: int = 50
    cfg_scale: float = 7.5
    quality: str = "standard"
    latent_channels: int = 16
    latent_time_steps: int = 128
    
    def apply_quality_preset(self):
        if self.quality == "fast":
            self.num_steps = 20
            self.cfg_scale = 5.0
        elif self.quality == "high":
            self.num_steps = 100
            self.cfg_scale = 10.0


@dataclass
class GenerationInput:
    lyrics: Optional[str] = None
    style: Optional[str] = None
    
    def validate(self) -> List[str]:
        return []


class LyroPipeline:
    def __init__(self, model: MultimodalLyroSystem, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device).eval()
    
    @classmethod
    def from_pretrained(cls, model_checkpoint: str = None, device: Optional[torch.device] = None):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultimodalLyroSystem()
        if model_checkpoint and Path(model_checkpoint).exists():
            model.load_pretrained(model_checkpoint)
        return cls(model=model, device=device)
    
    def generate(
        self,
        input_data: GenerationInput,
        generation_config: GenerationConfig = None
    ) -> torch.Tensor:
        if generation_config is None:
            generation_config = GenerationConfig()
        
        generation_config.apply_quality_preset()
        
        lyrics = [input_data.lyrics] if input_data.lyrics else None
        style = [input_data.style] if input_data.style else None
        
        return self.model.generate_music(
            lyrics=lyrics,
            style=style,
            num_steps=generation_config.num_steps,
            cfg_scale=generation_config.cfg_scale
        )


def create_pipeline(checkpoint: Optional[str] = None, device: str = "auto") -> LyroPipeline:
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return LyroPipeline.from_pretrained(checkpoint, device)


def quick_generate(
    lyrics: str = None,
    style: str = None,
    quality: str = "standard",
    checkpoint: str = None
) -> torch.Tensor:
    pipeline = create_pipeline(checkpoint)
    input_data = GenerationInput(lyrics=lyrics, style=style)
    config = GenerationConfig(quality=quality)
    return pipeline.generate(input_data, config)