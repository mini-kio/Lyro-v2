# lyro/ssm/config.py
"""
SSM + U-Net Configuration
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class SSMConfig:
    """SSM + U-Net 모델 설정"""
    
    # 오디오 설정
    sample_rate: int = 44100
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: int = 40
    f_max: int = 16000
    
    # 모델 아키텍처
    input_channels: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 256, 256, 512])
    mamba_layers: List[int] = field(default_factory=lambda: [2, 2, 3, 3, 4])
    max_seq_len: int = 6144  # ~5분
    
    # Mamba 설정
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # 조건 임베딩
    task_embedding_dim: int = 512
    time_embedding_dim: int = 512
    lyrics_embedding_dim: int = 1024
    style_embedding_dim: int = 512
    
    # Flow Matching 설정
    flow_steps: int = 8
    flow_sigma: float = 0.0
    integration_method: str = 'euler'
    
    # 품질별 설정
    quality_configs: Dict[str, Dict] = field(default_factory=lambda: {
        'fast': {
            'flow_steps': 4,
            'integration_method': 'euler'
        },
        'standard': {
            'flow_steps': 8,
            'integration_method': 'euler'
        },
        'premium': {
            'flow_steps': 16,
            'integration_method': 'rk4'
        }
    })
    
    # 학습 설정
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    
    # 데이터 설정
    batch_size: int = 4
    num_workers: int = 4
    epochs: int = 150
    
    # Stage별 설정
    stage_configs: Dict[str, Dict] = field(default_factory=lambda: {
        'B': {
            'learning_rate': 1e-3,
            'task_ratios': {'SONG': 1.0, 'INST': 0.0, 'COVER': 0.0},
            'epochs': 50
        },
        'C': {
            'learning_rate': 5e-4,
            'task_ratios': {'SONG': 0.7, 'INST': 0.3, 'COVER': 0.0},
            'epochs': 50
        },
        'D': {
            'learning_rate': 3e-4,
            'task_ratios': {'SONG': 0.5, 'INST': 0.2, 'COVER': 0.3},
            'epochs': 50,
            'icl_max_length': 180  # 3분
        },
        'E': {
            'learning_rate': 1e-4,
            'task_ratios': {'SONG': 0.6, 'INST': 0.15, 'COVER': 0.25},
            'epochs': 20,
            'quality_threshold': 0.8
        }
    })
    
    # Multitask 설정
    adaptive_weights: bool = True
    grad_norm_threshold: float = 1.0
    weight_update_interval: int = 100
    
    # ICL 설정
    icl_max_ref_length: float = 180.0  # 3분
    icl_style_extraction: str = 'hierarchical'
    icl_dual_track: bool = True
    icl_progressive_context: bool = True
    
    # 데이터 설정
    max_audio_length: int = 132300  # 3초 at 44.1kHz
    max_text_length: int = 512


@dataclass
class InferenceConfig:
    """추론 설정"""
    
    # 기본 설정
    device: str = 'cuda'
    dtype: str = 'float16'
    
    # 생성 설정
    default_duration: float = 30.0
    max_duration: float = 300.0
    
    # Flow 설정
    default_flow_steps: int = 8
    default_guidance_scale: float = 1.5
    
    # EOS 설정
    use_early_stopping: bool = True
    eos_penalty: float = 0.1
    min_generation_length: int = 100
    
    # 서버 설정
    server_host: str = '0.0.0.0'
    server_port: int = 8000
    max_concurrent_requests: int = 10
    request_timeout: int = 300  # 5분
    
    # 캐시 설정
    enable_cache: bool = True
    cache_size: int = 100
    cache_ttl: int = 3600  # 1시간