# lyro/dcae/config.py - Optimized MusicDCAE-style Configuration
"""
Optimized DCAE Configuration - MusicDCAE-style CNN Model
Focus: Efficient CNN architecture, improved compression ratio, 44.1kHz stereo
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import torch
import os

# Disable torch compile globally
import torch._dynamo
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'


@dataclass
class DCAEConfig:
    """Optimized DCAE Configuration for MusicDCAE-style CNN Model"""
    
    # ==================== Audio Settings ====================
    sample_rate: int = 44100
    audio_duration: float = 1.0  # 1-second clips for efficiency
    
    # ==================== Model Architecture ====================
    # Enhanced compression settings
    latent_channels: int = 8  # Reduced from 16 for better compression
    
    # ConvNeXt-style encoder settings
    encoder_depths: List[int] = field(default_factory=lambda: [2, 2, 6, 2])
    encoder_dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    encoder_drop_path_rate: float = 0.1
    encoder_kernel_sizes: Tuple[int] = field(default_factory=lambda: (7, 11))
    
    # HiFiGAN-style decoder settings
    decoder_upsample_rates: Tuple[int] = field(default_factory=lambda: (8, 8, 2, 2, 2))
    decoder_upsample_kernel_sizes: Tuple[int] = field(default_factory=lambda: (16, 16, 4, 4, 4))
    decoder_resblock_kernel_sizes: Tuple[int] = field(default_factory=lambda: (3, 7, 11))
    decoder_resblock_dilation_sizes: Tuple[Tuple[int]] = field(
        default_factory=lambda: ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    )
    decoder_initial_channel: int = 512
    
    # Mel-spectrogram settings (following MusicDCAE)
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 40.0
    f_max: float = 16000.0
    
    # ==================== Training Settings ====================
    # Optimizer
    learning_rate: float = 1e-4  # Standard for CNN models
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    
    # Training dynamics
    batch_size: int = 8  # Increased for CNN efficiency
    epochs: int = 200
    grad_clip: float = 1.0
    
    # Scheduler
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    
    # ==================== Loss Settings ====================
    # Multi-scale loss weights
    reconstruction_weight: float = 1.0
    spectral_loss_weight: float = 0.5
    perceptual_loss_weight: float = 0.3
    
    # ==================== Data Settings ====================
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Data processing
    min_duration: float = 0.5
    max_duration: float = 2.0
    train_split: float = 0.85
    val_split: float = 0.15
    
    # ==================== Augmentation Settings ====================
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    gain_range: Tuple[float, float] = (-3.0, 3.0)
    
    # ==================== Stability Settings ====================
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = False
    
    # ==================== Checkpoint Settings ====================
    save_interval: int = 20
    keep_last_n: int = 5
    save_best_only: bool = True
    
    # ==================== Validation Settings ====================
    val_check_interval: int = 10
    val_batches_limit: int = 20
    
    # ==================== Logging Settings ====================
    log_interval: int = 100
    sample_interval: int = 50
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration for optimized model"""
        issues = []
        warnings = []
        
        # Compression ratio validation
        total_compression = (
            self.hop_length * 
            (2 ** len(self.decoder_upsample_rates))
        )
        expected_compression_ratio = total_compression / self.latent_channels
        
        if expected_compression_ratio < 50:
            warnings.append(f"Low compression ratio: {expected_compression_ratio:.1f}")
        
        # Memory validation
        estimated_memory = self._estimate_memory_usage()
        if estimated_memory > 12:  # 12GB limit for efficiency
            warnings.append(f"Estimated memory usage {estimated_memory:.1f}GB may be high")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'estimated_memory_gb': estimated_memory,
            'estimated_parameters': self._estimate_parameters(),
            'compression_ratio': expected_compression_ratio
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate GPU memory usage in GB"""
        # CNN model is more memory efficient
        model_memory = 1.5  # ~1.5GB for optimized model
        batch_memory = self.batch_size * 0.3  # ~0.3GB per batch item
        overhead = 0.5  # Reduced overhead
        
        return model_memory + batch_memory + overhead
    
    def _estimate_parameters(self) -> int:
        """Estimate model parameters"""
        # Estimate for optimized CNN model
        encoder_params = sum(dim * 4 for dim in self.encoder_dims) * 1000
        decoder_params = self.decoder_initial_channel * 2000
        
        return encoder_params + decoder_params
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            'latent_channels': self.latent_channels,
            'encoder_depths': self.encoder_depths,
            'encoder_dims': self.encoder_dims,
            'encoder_drop_path_rate': self.encoder_drop_path_rate,
            'encoder_kernel_sizes': self.encoder_kernel_sizes,
            'decoder_upsample_rates': self.decoder_upsample_rates,
            'decoder_upsample_kernel_sizes': self.decoder_upsample_kernel_sizes,
            'decoder_resblock_kernel_sizes': self.decoder_resblock_kernel_sizes,
            'decoder_resblock_dilation_sizes': self.decoder_resblock_dilation_sizes,
            'decoder_initial_channel': self.decoder_initial_channel,
            'n_fft': self.n_fft,
            'win_length': self.win_length,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'sample_rate': self.sample_rate,
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration"""
        return {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'grad_clip': self.grad_clip,
            'mixed_precision': self.mixed_precision,
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data-specific configuration"""
        return {
            'sample_rate': self.sample_rate,
            'audio_duration': self.audio_duration,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'use_augmentation': self.use_augmentation,
            'augmentation_prob': self.augmentation_prob,
        }


@dataclass
class DCAETrainingConfig:
    """Optimized training configuration"""
    
    # Paths
    dataset_root: str = "dataset-dcae/datasets/raw"
    checkpoint_dir: str = "dcae/checkpoints_optimized"
    
    # Training
    resume: Optional[str] = None
    use_wandb: bool = False
    
    # Hardware
    device: str = "cuda"
    
    def __post_init__(self):
        """Post-initialization validation"""
        if not torch.cuda.is_available() and self.device == "cuda":
            print("Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"


# ==================== Factory Functions ====================

def create_optimized_dcae_config(
    audio_duration: float = 1.0,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    use_augmentation: bool = True,
    **kwargs
) -> DCAEConfig:
    """
    Create configuration for Optimized DCAE model
    
    Args:
        audio_duration: Audio duration in seconds
        batch_size: Training batch size
        learning_rate: Learning rate
        use_augmentation: Whether to use augmentation
        **kwargs: Additional config overrides
    
    Returns:
        DCAEConfig for optimized model
    """
    config = DCAEConfig(
        audio_duration=audio_duration,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_augmentation=use_augmentation,
    )
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")
    
    # Validate configuration
    validation = config.validate()
    
    if not validation['valid']:
        print("❌ Configuration validation failed:")
        for issue in validation['issues']:
            print(f"   - {issue}")
        raise ValueError("Configuration validation failed")
    
    if validation['warnings']:
        print("⚠️ Configuration warnings:")
        for warning in validation['warnings']:
            print(f"   - {warning}")
    
    print(f"✅ Optimized DCAE Config Created:")
    print(f"   - Parameters: ~{validation['estimated_parameters']:,}")
    print(f"   - Memory: ~{validation['estimated_memory_gb']:.1f}GB")
    print(f"   - Compression Ratio: {validation['compression_ratio']:.1f}:1")
    print(f"   - Latent Channels: {config.latent_channels}")
    
    return config


def create_fast_dcae_config(
    learning_rate: float = 2e-4,
    batch_size: int = 12,
    **kwargs
) -> DCAEConfig:
    """Create fast training configuration"""
    return create_optimized_dcae_config(
        learning_rate=learning_rate,
        batch_size=batch_size,
        audio_duration=0.75,  # Shorter for speed
        **kwargs
    )


def create_quality_dcae_config(
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    **kwargs
) -> DCAEConfig:
    """Create high quality configuration"""
    return create_optimized_dcae_config(
        learning_rate=learning_rate,
        batch_size=batch_size,
        audio_duration=1.5,  # Longer for quality
        latent_channels=12,  # More channels for quality
        **kwargs
    )


# ==================== Presets ====================

class DCAEPresets:
    """Predefined configuration presets"""
    
    @staticmethod
    def development() -> DCAEConfig:
        """Development preset - fast iteration"""
        return create_optimized_dcae_config(
            audio_duration=0.5,
            batch_size=4,
            epochs=100,
            save_interval=10,
            val_check_interval=5,
        )
    
    @staticmethod
    def production() -> DCAEConfig:
        """Production preset - high quality"""
        return create_optimized_dcae_config(
            audio_duration=1.0,
            batch_size=8,
            epochs=300,
            learning_rate=8e-5,
            augmentation_prob=0.6,
        )
    
    @staticmethod
    def research() -> DCAEConfig:
        """Research preset - comprehensive logging"""
        return create_optimized_dcae_config(
            audio_duration=1.0,
            batch_size=6,
            save_interval=10,
            log_interval=25,
            sample_interval=20,
        )


print("✅ DCAE Configuration - Optimized MusicDCAE-style")
print("Key features:")
print("- CNN-based architecture (no SSM)")
print("- Improved compression ratio")
print("- 44.1kHz stereo support")
print("- Memory efficient design")
print("- Enhanced numerical stability")