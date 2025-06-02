# lyro/dcae/config.py
"""
DCAE Configuration for CQT-SSM Based Models
Updated for Constant-Q Transform and music-optimized processing
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


@dataclass
class DCAEConfig:
    """CQT-SSM DCAE 모델 설정"""
    
    # 오디오 설정
    sample_rate: int = 44100
    
    # CQT 설정 (Constant-Q Transform)
    cqt_hop_length: int = 512
    cqt_n_bins: int = 84  # 7 octaves (C1 to C8)
    cqt_bins_per_octave: int = 12
    cqt_fmin: float = 32.7  # C1 frequency
    cqt_window: str = 'hann'
    
    # Multi-resolution CQT for loss computation
    cqt_hop_lengths: List[int] = field(default_factory=lambda: [256, 512, 1024])
    cqt_n_bins_list: List[int] = field(default_factory=lambda: [72, 84, 96])  # Different octave ranges
    
    # 모델 설정
    compression_factor: int = 8
    latent_channels: int = 8  # vocal: 0-3, inst: 4-7
    
    # 인코더 (CQT-SSM based)
    encoder_base_channels: int = 64
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    encoder_strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    encoder_kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 5, 3])
    
    # 디코더 (Inverse CQT based)
    decoder_base_channels: int = 64
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    decoder_strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    decoder_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 5, 7])
    
    # SSM (State Space Model) 설정
    ssm_d_state: int = 32  # Reduced from 64 for efficiency
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    ssm_layers: List[int] = field(default_factory=lambda: [2, 2, 3, 3, 2])
    use_multiscale_ssm: bool = True
    
    # Memory optimization
    chunk_size: int = 256
    use_checkpointing: bool = True
    memory_efficient: bool = True
    checkpointing_segments: int = 4
    
    # VQ 설정 (선택적)
    use_vector_quantization: bool = False
    vq_num_embeddings: int = 1024
    vq_embedding_dim: int = 8
    vq_beta: float = 0.25
    
    # 손실 가중치 (CQT-based)
    reconstruction_weight: float = 1.0
    cqt_loss_weight: float = 1.0  # Changed from stft_loss_weight
    time_loss_weight: float = 0.1
    vq_weight: float = 0.02
    adversarial_weight: float = 0.1
    
    # CQT 손실 세부 설정
    w_cqt: float = 1.0         # CQT magnitude loss weight
    w_temporal: float = 0.5     # Temporal consistency weight
    w_harmonic: float = 0.3     # Harmonic structure weight
    
    # 학습 설정
    learning_rate: float = 2e-4  # Slightly lower for CQT models
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # 데이터 설정
    batch_size: int = 8  # Conservative for CQT memory usage
    num_workers: int = 4
    epochs: int = 150
    audio_duration: float = 10.0  # seconds
    
    # 체크포인트
    save_interval: int = 5
    keep_last_n: int = 3  # Keep fewer checkpoints for space
    
    # Lossy-to-Lossless Enhancement (updated for CQT)
    lossy_augmentation_prob: float = 0.67
    vbr_range: Tuple[int, int] = (0, 7)
    restoration_loss_weight: float = 0.3
    
    # T-2: Weight Normalization
    use_weight_norm: bool = True
    
    # T-3: EMA Configuration  
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_update_after: int = 100
    ema_update_every: int = 10
    
    # T-1: Augmentation Configuration (Music-optimized)
    use_augmentation: bool = True
    augmentation_prob: float = 0.8
    gain_range: Tuple[float, float] = (-3.0, 3.0)
    pitch_range: Tuple[float, float] = (-0.5, 0.5)  # Conservative for music
    tempo_range: Tuple[float, float] = (0.9, 1.1)
    noise_level: float = 0.005
    reverb_prob: float = 0.3
    eq_prob: float = 0.4
    preserve_musical_structure: bool = True
    harmonic_distortion_prob: float = 0.2
    
    # Harmonic-Percussive Separation
    use_harmonic_percussive: bool = True
    hp_kernel_size_h: int = 17  # Harmonic kernel (vertical)
    hp_kernel_size_p: int = 17  # Percussive kernel (horizontal)
    hp_power: float = 2.0
    hp_margin: float = 1.0
    hp_learnable: bool = True
    
    # Model size presets
    model_size: str = "base"  # "small", "base", "large"
    
    def get_model_config(self) -> Dict:
        """Get model-specific configuration based on size"""
        if self.model_size == "small":
            return {
                "encoder_base_channels": 32,
                "decoder_base_channels": 32,
                "latent_channels": 6,
                "cqt_n_bins": 72,  # 6 octaves
                "ssm_d_state": 16,
                "chunk_size": 128
            }
        elif self.model_size == "base":
            return {
                "encoder_base_channels": 64,
                "decoder_base_channels": 64,
                "latent_channels": 8,
                "cqt_n_bins": 84,  # 7 octaves
                "ssm_d_state": 32,
                "chunk_size": 256
            }
        elif self.model_size == "large":
            return {
                "encoder_base_channels": 96,
                "decoder_base_channels": 96,
                "latent_channels": 12,
                "cqt_n_bins": 96,  # 8 octaves
                "ssm_d_state": 48,
                "chunk_size": 512
            }
        else:
            return {}
    
    def update_for_model_size(self):
        """Update configuration based on model size"""
        config_updates = self.get_model_config()
        for key, value in config_updates.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class CQTSSMConfig:
    """CQT-SSM specific configuration"""
    
    # CQT Transform parameters
    sample_rate: int = 44100
    hop_length: int = 512
    n_bins: int = 84
    bins_per_octave: int = 12
    fmin: float = 32.7  # C1
    window: str = 'hann'
    center: bool = True
    pad_mode: str = 'reflect'
    
    # SSM parameters
    d_model: int = 64
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    dt_rank: Optional[int] = None
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    bias: bool = True
    conv_bias: bool = True
    
    # Memory optimization
    chunk_size: int = 256
    use_checkpointing: bool = True
    memory_efficient: bool = True
    
    # Multi-scale processing
    use_multiscale_ssm: bool = True
    scales: List[int] = field(default_factory=lambda: [1, 2, 4])
    
    # Harmonic-Percussive separation
    use_hp_separation: bool = True
    hp_kernel_size_h: int = 17
    hp_kernel_size_p: int = 17
    hp_learnable: bool = True


@dataclass 
class HiFiGANConfig:
    """HiFiGAN Discriminator 설정 (CQT 호환)"""
    
    # Multi-Period Discriminator
    periods: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    
    # Multi-Scale Discriminator  
    scales: int = 3
    downsample_scales: List[int] = field(default_factory=lambda: [2, 2])
    downsample_kernel_sizes: List[int] = field(default_factory=lambda: [4, 4])
    
    # CQT-aware discriminator settings
    use_cqt_discriminator: bool = True
    cqt_n_bins: int = 84
    cqt_hop_length: int = 512
    
    # 공통 설정
    discriminator_channel_mult: int = 1
    use_spectral_norm: bool = False
    use_weight_norm: bool = True  # Consistent with main model


@dataclass
class TrainingConfig:
    """Enhanced training configuration for CQT-SSM"""
    
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-6
    fused: bool = True  # Use fused optimizer if available
    
    # Scheduler
    scheduler: str = "cosine_annealing"
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    T_0: int = 50  # For cosine annealing with restarts
    T_mult: int = 1
    
    # Training dynamics
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Validation
    val_check_interval: int = 2  # Every N epochs
    val_batches_limit: int = 15  # Limit validation batches
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    save_best_only: bool = False
    
    # Logging
    log_every_n_steps: int = 200
    sample_every_n_epochs: int = 10
    num_samples: int = 4
    
    # Early stopping
    patience: int = 20
    min_delta: float = 1e-4
    
    # CQT-specific training settings
    cqt_loss_start_epoch: int = 0  # When to start CQT loss
    harmonic_loss_start_epoch: int = 5  # When to start harmonic loss
    progressive_training: bool = False  # Progressive resolution training


@dataclass
class DataConfig:
    """Data configuration for CQT-SSM training"""
    
    # Paths
    dataset_root: str = "dataset-dcae/datasets/raw"
    cache_dir: Optional[str] = None
    
    # Audio processing
    sample_rate: int = 44100
    audio_duration: float = 10.0  # seconds
    min_duration: float = 1.0     # minimum duration
    max_duration: float = 30.0    # maximum duration
    
    # Data splitting
    train_split: float = 0.85
    val_split: float = 0.15
    test_split: float = 0.0
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.8
    
    # File handling
    supported_formats: List[str] = field(default_factory=lambda: ['.wav', '.flac', '.mp3', '.m4a'])
    skip_corrupted: bool = True
    normalize_audio: bool = True
    
    # Memory management
    cache_audio: bool = False  # Don't cache for CQT processing
    preload_data: bool = False
    
    # Quality filters
    min_sample_rate: int = 22050
    max_file_size_mb: int = 100
    remove_silence: bool = False


# Convenience function to create complete configuration
def create_cqt_ssm_config(
    model_size: str = "base",
    audio_duration: float = 10.0,
    batch_size: int = 8,
    use_augmentation: bool = True,
    **kwargs
) -> DCAEConfig:
    """
    Create a complete CQT-SSM DCAE configuration
    
    Args:
        model_size: "small", "base", or "large"
        audio_duration: Audio duration in seconds
        batch_size: Training batch size
        use_augmentation: Whether to use audio augmentation
        **kwargs: Additional configuration overrides
    
    Returns:
        Complete DCAEConfig instance
    """
    config = DCAEConfig()
    
    # Set basic parameters
    config.model_size = model_size
    config.audio_duration = audio_duration
    config.batch_size = batch_size
    config.use_augmentation = use_augmentation
    
    # Update for model size
    config.update_for_model_size()
    
    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown configuration parameter: {key}")
    
    return config


# Presets for different use cases
def get_high_quality_config() -> DCAEConfig:
    """Configuration for high-quality music processing"""
    return create_cqt_ssm_config(
        model_size="large",
        audio_duration=15.0,
        batch_size=4,  # Smaller batch for larger model
        cqt_n_bins=96,  # 8 octaves
        ssm_d_state=48,
        chunk_size=512,
        use_augmentation=True,
        preserve_musical_structure=True,
        w_harmonic=0.4,  # Higher harmonic weight
        ema_decay=0.9995  # Slower EMA decay
    )


def get_efficient_config() -> DCAEConfig:
    """Configuration for efficient training/inference"""
    return create_cqt_ssm_config(
        model_size="small",
        audio_duration=8.0,
        batch_size=16,  # Larger batch for smaller model
        cqt_n_bins=72,  # 6 octaves
        ssm_d_state=16,
        chunk_size=128,
        use_augmentation=True,
        preserve_musical_structure=True,
        checkpointing_segments=2
    )


def get_research_config() -> DCAEConfig:
    """Configuration for research and experimentation"""
    return create_cqt_ssm_config(
        model_size="base",
        audio_duration=12.0,
        batch_size=6,
        cqt_n_bins=84,
        use_augmentation=True,
        preserve_musical_structure=False,  # More aggressive augmentation
        harmonic_distortion_prob=0.3,
        save_interval=2,  # Save more frequently
        log_every_n_steps=100  # More frequent logging
    )
