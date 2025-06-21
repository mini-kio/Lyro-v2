# lyro/dcae/config.py - Simplified Large Model Configuration
"""
Simplified DCAE Configuration - Large Model Only
Focus: Numerical stability, FP16 enforcement, DDP compatibility
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
    """Simplified DCAE Configuration for Large Model (16 latent channels)"""
    
    # ==================== Audio Settings ====================
    sample_rate: int = 44100
    audio_duration: float = 2.0  # seconds
    
    # ==================== Model Architecture ====================
    # Large model fixed settings
    latent_channels: int = 16
    base_channels: int = 128
    s6_layers: List[int] = field(default_factory=lambda: [3, 4, 4])
    d_state: int = 64
    
    # CQT settings
    cqt_n_bins: int = 96
    cqt_hop_length: int = 512
    
    # Encoder/Decoder channels
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    decoder_channels: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    
    # ==================== Training Settings ====================
    # Optimizer
    learning_rate: float = 8e-5  # Conservative for large model
    weight_decay: float = 0.02
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-6
    
    # Training dynamics
    batch_size: int = 4  # Reduced for large model
    epochs: int = 100
    grad_clip: float = 0.5  # Conservative
    
    # Scheduler
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    
    # ==================== Loss Settings ====================
    # Main loss weights
    reconstruction_weight: float = 1.0
    cqt_loss_weight: float = 1.0
    time_loss_weight: float = 0.1
    
    # ==================== Data Settings ====================
    num_workers: int = 0  # Disabled for DDP safety
    pin_memory: bool = False  # Disabled for DDP safety
    persistent_workers: bool = False
    
    # Data processing
    min_duration: float = 0.5
    max_duration: float = 10.0
    train_split: float = 0.85
    val_split: float = 0.15
    
    # ==================== Augmentation Settings ====================
    use_augmentation: bool = True
    augmentation_prob: float = 0.3  # Conservative
    gain_range: Tuple[float, float] = (-1.0, 1.0)
    
    # ==================== Stability Settings ====================
    # Critical stability flags
    disable_torch_compile: bool = True
    disable_dynamo_tracing: bool = True
    enable_numerical_stability: bool = True
    use_safe_operations: bool = True
    gradient_checkpointing: bool = False  # Disabled to prevent NaN
    
    # Mixed precision
    mixed_precision: str = "fp16"
    force_fp16: bool = True
    
    # ==================== Checkpoint Settings ====================
    save_interval: int = 10
    keep_last_n: int = 3
    save_best_only: bool = True
    
    # ==================== Validation Settings ====================
    val_check_interval: int = 5
    val_batches_limit: int = 10
    
    # ==================== Logging Settings ====================
    log_interval: int = 100
    sample_interval: int = 20
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration for large model"""
        issues = []
        warnings = []
        
        # Model size validation
        if self.latent_channels != 16:
            issues.append(f"Large model requires 16 latent channels, got {self.latent_channels}")
        
        if self.base_channels != 128:
            warnings.append(f"Base channels {self.base_channels} may not be optimal for large model")
        
        # Training stability validation
        if self.learning_rate > 1e-4:
            warnings.append(f"High learning rate {self.learning_rate} may cause instability")
        
        if self.batch_size > 8:
            warnings.append(f"Large batch size {self.batch_size} may cause OOM")
        
        if not self.disable_torch_compile:
            issues.append("torch.compile must be disabled for DDP compatibility")
        
        if self.num_workers > 0:
            warnings.append("num_workers > 0 may cause DDP issues")
        
        if self.gradient_checkpointing:
            warnings.append("Gradient checkpointing may cause NaN issues")
        
        # Memory validation
        estimated_memory = self._estimate_memory_usage()
        if estimated_memory > 20:  # 20GB limit
            warnings.append(f"Estimated memory usage {estimated_memory:.1f}GB may cause OOM")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'estimated_memory_gb': estimated_memory,
            'estimated_parameters': self._estimate_parameters()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate GPU memory usage in GB"""
        # Rough estimation based on model size and batch size
        model_memory = 2.0  # ~2GB for large model
        batch_memory = self.batch_size * 0.5  # ~0.5GB per batch item
        overhead = 1.0  # System overhead
        
        return model_memory + batch_memory + overhead
    
    def _estimate_parameters(self) -> int:
        """Estimate model parameters"""
        # Rough estimation for large model
        base_params = 50_000_000  # Base parameters
        s6_params = sum(self.s6_layers) * 500_000  # S6 layer parameters
        
        return base_params + s6_params
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            'latent_channels': self.latent_channels,
            'base_channels': self.base_channels,
            's6_layers': self.s6_layers,
            'd_state': self.d_state,
            'cqt_n_bins': self.cqt_n_bins,
            'sample_rate': self.sample_rate,
            'force_fp16': True,
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
            'disable_torch_compile': self.disable_torch_compile,
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
    """Simplified training configuration"""
    
    # Paths
    dataset_root: str = "dataset-dcae/datasets/raw"
    checkpoint_dir: str = "dcae/checkpoints_large"
    
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

def create_large_dcae_config(
    audio_duration: float = 2.0,
    batch_size: int = 4,
    learning_rate: float = 8e-5,
    use_augmentation: bool = True,
    **kwargs
) -> DCAEConfig:
    """
    Create configuration for Large DCAE model
    
    Args:
        audio_duration: Audio duration in seconds
        batch_size: Training batch size
        learning_rate: Learning rate
        use_augmentation: Whether to use augmentation
        **kwargs: Additional config overrides
    
    Returns:
        DCAEConfig for large model
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
    
    print(f"✅ Large DCAE Config Created:")
    print(f"   - Parameters: ~{validation['estimated_parameters']:,}")
    print(f"   - Memory: ~{validation['estimated_memory_gb']:.1f}GB")
    print(f"   - Latent Channels: {config.latent_channels}")
    
    return config


def create_stable_training_config(
    learning_rate: float = 5e-5,
    batch_size: int = 2,
    **kwargs
) -> DCAEConfig:
    """Create maximally stable training configuration"""
    return create_large_dcae_config(
        learning_rate=learning_rate,
        batch_size=batch_size,
        grad_clip=0.3,
        augmentation_prob=0.2,
        disable_torch_compile=True,
        gradient_checkpointing=False,
        **kwargs
    )


def create_fast_training_config(
    learning_rate: float = 1e-4,
    batch_size: int = 6,
    **kwargs
) -> DCAEConfig:
    """Create fast training configuration (higher batch size)"""
    return create_large_dcae_config(
        learning_rate=learning_rate,
        batch_size=batch_size,
        audio_duration=1.5,  # Shorter audio for speed
        **kwargs
    )


# ==================== Presets ====================

class DCAEPresets:
    """Predefined configuration presets"""
    
    @staticmethod
    def development() -> DCAEConfig:
        """Development preset - fast iteration"""
        return create_large_dcae_config(
            audio_duration=1.0,
            batch_size=2,
            epochs=50,
            save_interval=5,
            val_check_interval=2,
        )
    
    @staticmethod
    def production() -> DCAEConfig:
        """Production preset - high quality"""
        return create_large_dcae_config(
            audio_duration=2.0,
            batch_size=4,
            epochs=200,
            learning_rate=6e-5,
            augmentation_prob=0.4,
        )
    
    @staticmethod
    def research() -> DCAEConfig:
        """Research preset - comprehensive logging"""
        return create_large_dcae_config(
            audio_duration=2.0,
            batch_size=4,
            save_interval=5,
            log_interval=50,
            sample_interval=10,
        )


print("✅ DCAE Configuration - Large Model Simplified")
print("Key features:")
print("- Large model focused (16 latent channels)")
print("- Numerical stability enforced")
print("- DDP compatibility ensured") 
print("- FP16 optimization")
print("- Comprehensive validation")