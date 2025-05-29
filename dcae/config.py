# lyro/dcae/config.py
"""
DCAE Configuration
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


@dataclass
class DCAEConfig:
    """DCAE 모델 설정"""
    
    # 오디오 설정
    sample_rate: int = 44100
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: int = 40
    f_max: int = 16000
    
    # 모델 설정
    compression_factor: int = 8
    latent_channels: int = 8  # vocal: 0-3, inst: 4-7
    
    # 인코더
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    encoder_strides: List[int] = field(default_factory=lambda: [2, 2, 2])
    encoder_kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 5])
    
    # 디코더
    decoder_channels: List[int] = field(default_factory=lambda: [128, 64, 32])
    decoder_strides: List[int] = field(default_factory=lambda: [2, 2, 2])
    decoder_kernel_sizes: List[int] = field(default_factory=lambda: [5, 5, 7])
    
    # VQ 설정
    vq_num_embeddings: int = 1024
    vq_embedding_dim: int = 8
    vq_beta: float = 0.25
    
    # 손실 가중치
    reconstruction_weight: float = 1.0
    vq_weight: float = 1.0
    adversarial_weight: float = 0.5
    
    # STFT 손실 설정
    stft_scales: List[int] = field(default_factory=lambda: [2048, 1024, 512, 256, 128])
    
    # 학습 설정
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # 데이터 설정
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 150
    
    # 체크포인트
    save_interval: int = 5
    keep_last_n: int = 5
    
    # Lossy-to-Lossless Enhancement
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
    
    # T-1: Augmentation Configuration
    use_augmentation: bool = True
    augmentation_prob: float = 0.8
    gain_range: Tuple[float, float] = (-3.0, 3.0)
    pitch_range: Tuple[float, float] = (-0.5, 0.5)
    tempo_range: Tuple[float, float] = (0.9, 1.1)
    noise_level: float = 0.005
    reverb_prob: float = 0.3
    eq_prob: float = 0.4


@dataclass
class HiFiGANConfig:
    """HiFiGAN Discriminator 설정"""
    
    # Multi-Period Discriminator
    periods: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    
    # Multi-Scale Discriminator
    scales: int = 3
    downsample_scales: List[int] = field(default_factory=lambda: [2, 2])
    downsample_kernel_sizes: List[int] = field(default_factory=lambda: [4, 4])
    
    # 공통 설정
    discriminator_channel_mult: int = 1
    use_spectral_norm: bool = False