# lyro/ssm/config.py
"""
S6-Optimized SSM + U-Net Configuration
Enhanced configuration for S6 (Mamba-2) State Space Models
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import math


@dataclass
class S6SSMConfig:
    """S6 + U-Net 모델 설정 (Mamba-2 최적화)"""
    
    # 오디오 설정
    sample_rate: int = 44100
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    f_min: int = 40
    f_max: int = 16000
    
    # S6 모델 아키텍처
    input_channels: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 384, 512])
    s6_layers: List[int] = field(default_factory=lambda: [2, 3, 4, 4])
    max_seq_len: int = 8192  # S6는 더 긴 시퀀스 처리 가능
    
    # S6 State Space 설정
    d_state: int = 128  # S6에서 증가된 상태 차원
    d_head: int = 64    # S6 헤드 차원
    d_conv: int = 4
    headdim: int = 64
    ngroups: int = 1
    expand: int = 2
    
    # S6 특화 파라미터
    chunk_size: int = 256              # S6 청킹 크기
    use_mem_eff_path: bool = True      # 메모리 효율적 경로 사용
    A_init_range: Tuple[float, float] = (1, 16)  # S6 A 행렬 초기화 범위
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    
    # S6 최적화 설정
    use_chunked_processing: bool = True    # 청킹 프로세싱 활성화
    adaptive_chunking: bool = True         # 적응적 청킹
    chunk_overlap: int = 16               # 청크 오버랩
    enable_s6_optimizations: bool = True   # S6 최적화 활성화
    
    # 조건 임베딩
    task_embedding_dim: int = 512
    time_embedding_dim: int = 512
    lyrics_embedding_dim: int = 1024
    style_embedding_dim: int = 512
    
    # S6 Flow Matching 설정
    flow_steps: int = 10
    flow_sigma: float = 1e-4
    scheduler_type: str = 's6_cosine'      # S6 최적화된 스케줄러
    solver_type: str = 's6_heun'           # S6 최적화된 솔버
    integration_method: str = 's6_heun'    # 역호환성
    
    # S6 품질별 설정
    quality_configs: Dict[str, Dict] = field(default_factory=lambda: {
        's6_fast': {
            'flow_steps': 6,
            'solver_type': 's6_euler',
            'chunk_size': 128,
            'cfg_scale': 1.2
        },
        's6_standard': {
            'flow_steps': 10,
            'solver_type': 's6_heun',
            'chunk_size': 256,
            'cfg_scale': 1.5
        },
        's6_premium': {
            'flow_steps': 16,
            'solver_type': 's6_adaptive',
            'chunk_size': 512,
            'cfg_scale': 2.0
        }
    })
    
    # S6 학습 설정
    learning_rate: float = 4e-4  # S6에 최적화된 학습률
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    
    # S6 데이터 설정
    batch_size: int = 4  # S6 메모리 효율성으로 더 큰 배치 가능
    num_workers: int = 6  # S6 처리 속도로 더 많은 워커 허용
    epochs: int = 150
    
    # S6 Stage별 설정
    stage_configs: Dict[str, Dict] = field(default_factory=lambda: {
        'B': {
            'learning_rate': 8e-4,  # S6 최적화된 학습률
            'task_ratios': {'SONG': 1.0, 'INST': 0.0, 'COVER': 0.0},
            'epochs': 50,
            'chunk_size': 256,
            'flow_steps': 8
        },
        'C': {
            'learning_rate': 6e-4,
            'task_ratios': {'SONG': 0.7, 'INST': 0.3, 'COVER': 0.0},
            'epochs': 50,
            'chunk_size': 256,
            'flow_steps': 10
        },
        'D': {
            'learning_rate': 4e-4,
            'task_ratios': {'SONG': 0.5, 'INST': 0.2, 'COVER': 0.3},
            'epochs': 50,
            'icl_max_length': 240,  # S6로 더 긴 컨텍스트 처리
            'chunk_size': 384,
            'flow_steps': 12
        },
        'E': {
            'learning_rate': 2e-4,
            'task_ratios': {'SONG': 0.6, 'INST': 0.15, 'COVER': 0.25},
            'epochs': 20,
            'quality_threshold': 0.85,  # S6로 더 높은 품질 기준
            'chunk_size': 512,
            'flow_steps': 16
        }
    })
    
    # S6 Multitask 설정
    adaptive_weights: bool = True
    grad_norm_threshold: float = 1.0
    weight_update_interval: int = 100
    s6_task_balancing: bool = True  # S6 특화 태스크 밸런싱
    
    # S6 ICL 설정
    icl_max_ref_length: float = 240.0  # S6로 더 긴 참조 처리 가능
    icl_style_extraction: str = 's6_hierarchical'  # S6 특화 추출
    icl_dual_track: bool = True
    icl_progressive_context: bool = True
    icl_chunk_processing: bool = True  # S6 청킹으로 긴 참조 처리
    
    # S6 데이터 설정
    max_audio_length: int = 441000  # 10초 (S6로 더 긴 시퀀스 처리)
    max_text_length: int = 768      # S6로 더 긴 텍스트 처리
    
    # S6 메모리 최적화
    memory_efficient: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    torch_compile: bool = True
    compile_mode: str = 'default'
    
    # S6 성능 모니터링
    track_s6_metrics: bool = True
    log_chunk_efficiency: bool = True
    monitor_memory_usage: bool = True
    benchmark_interval: int = 100


@dataclass
class S6InferenceConfig:
    """S6 추론 설정"""
    
    # 기본 설정
    device: str = 'cuda'
    dtype: str = 'float16'
    
    # S6 생성 설정
    default_duration: float = 30.0
    max_duration: float = 600.0  # S6로 더 긴 생성 가능
    
    # S6 Flow 설정
    default_flow_steps: int = 10
    default_guidance_scale: float = 1.5
    scheduler_type: str = 's6_cosine'
    solver_type: str = 's6_heun'
    
    # S6 청킹 설정
    chunk_size: int = 256
    enable_chunked_generation: bool = True
    chunk_overlap: int = 16
    adaptive_chunking: bool = True
    
    # S6 EOS 설정
    use_early_stopping: bool = True
    eos_penalty: float = 0.1
    min_generation_length: int = 100
    s6_sequence_termination: bool = True  # S6 특화 시퀀스 종료
    
    # S6 서버 설정
    server_host: str = '0.0.0.0'
    server_port: int = 8000
    max_concurrent_requests: int = 12  # S6 효율성으로 더 많은 동시 요청
    request_timeout: int = 300
    
    # S6 캐시 설정
    enable_cache: bool = True
    cache_size: int = 150  # S6 메모리 효율성으로 더 큰 캐시
    cache_ttl: int = 3600
    chunk_cache_enabled: bool = True  # S6 청크 캐시
    
    # S6 최적화 설정
    use_torch_compile: bool = True
    compile_mode: str = 'default'
    memory_optimization: bool = True
    batch_generation: bool = True  # S6 배치 생성 지원


@dataclass 
class S6TrainingConfig:
    """S6 학습 전용 설정"""
    
    # S6 옵티마이저 설정
    optimizer: str = "adamw"
    learning_rate: float = 4e-4  # S6 최적화
    betas: Tuple[float, float] = (0.9, 0.95)  # S6에 더 적합
    weight_decay: float = 0.01
    eps: float = 1e-8
    fused: bool = True  # Fused AdamW for S6
    
    # S6 스케줄러 설정
    scheduler: str = "cosine_warmup"
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    warmup_ratio: float = 0.1
    cosine_restarts: bool = True  # S6에 효과적
    T_0: int = 25  # Restart period
    T_mult: int = 1
    
    # S6 학습 동역학
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    torch_compile: bool = True
    compile_mode: str = "default"
    
    # S6 데이터 로딩
    batch_size: int = 4
    num_workers: int = 6  # S6 효율성으로 증가
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 3  # S6 청킹에 최적화
    
    # S6 검증 설정
    val_check_interval: int = 2
    val_batches_limit: int = 20  # S6 효율성으로 증가
    val_chunk_processing: bool = True
    
    # S6 체크포인팅
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    save_best_only: bool = False
    save_s6_metrics: bool = True
    
    # S6 로깅
    log_every_n_steps: int = 100
    sample_every_n_epochs: int = 10
    num_samples: int = 6  # S6로 더 많은 샘플 생성
    log_s6_performance: bool = True
    
    # S6 Early stopping
    patience: int = 25  # S6 안정성으로 증가
    min_delta: float = 1e-4
    monitor_s6_efficiency: bool = True
    
    # S6 특화 학습 설정
    s6_loss_start_epoch: int = 0
    chunk_boundary_smoothing: bool = True
    adaptive_chunk_sizing: bool = True
    progressive_sequence_training: bool = False


@dataclass
class S6DataConfig:
    """S6 데이터 설정"""
    
    # 경로 설정
    dataset_root: str = "dataset-dcae/datasets/raw"
    cache_dir: Optional[str] = None
    
    # S6 오디오 처리
    sample_rate: int = 44100
    audio_duration: float = 10.0  # S6로 더 긴 처리 가능
    min_duration: float = 1.0
    max_duration: float = 60.0    # S6로 대폭 증가
    
    # S6 시퀀스 처리
    max_sequence_length: int = 8192  # S6 긴 시퀀스 지원
    chunk_size: int = 256
    chunk_overlap: int = 16
    adaptive_chunking: bool = True
    
    # 데이터 분할
    train_split: float = 0.85
    val_split: float = 0.15
    test_split: float = 0.0
    
    # S6 증강 설정
    use_augmentation: bool = True
    augmentation_prob: float = 0.8
    s6_temporal_augmentation: bool = True  # S6 특화 시간 증강
    preserve_chunk_boundaries: bool = True
    
    # 파일 처리
    supported_formats: List[str] = field(default_factory=lambda: ['.wav', '.flac', '.mp3', '.m4a', '.ogg'])
    skip_corrupted: bool = True
    normalize_audio: bool = True
    
    # S6 메모리 관리
    cache_audio: bool = False  # S6 청킹으로 캐시 불필요
    preload_data: bool = False
    streaming_processing: bool = True  # S6 스트리밍 처리
    
    # S6 품질 필터
    min_sample_rate: int = 22050
    max_file_size_mb: int = 200  # S6로 더 큰 파일 처리
    remove_silence: bool = False
    min_audio_quality: float = 0.7  # S6 품질 기준
    
    # S6 배치 처리
    dynamic_batching: bool = True  # S6 동적 배칭
    sort_by_length: bool = True    # S6 효율성을 위한 길이별 정렬
    drop_last: bool = True


@dataclass
class S6ModelConfig:
    """S6 모델별 상세 설정"""
    
    # 모델 크기별 설정
    model_size: str = "base"  # "small", "base", "large", "xl"
    
    # S6 아키텍처 설정
    model_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "small": {
            "hidden_dims": [96, 192, 288, 384],
            "s6_layers": [2, 2, 3, 3],
            "d_state": 64,
            "d_head": 32,
            "chunk_size": 128,
            "max_seq_len": 4096
        },
        "base": {
            "hidden_dims": [128, 256, 384, 512],
            "s6_layers": [2, 3, 4, 4],
            "d_state": 128,
            "d_head": 64,
            "chunk_size": 256,
            "max_seq_len": 8192
        },
        "large": {
            "hidden_dims": [256, 512, 768, 1024],
            "s6_layers": [3, 4, 6, 6],
            "d_state": 256,
            "d_head": 128,
            "chunk_size": 512,
            "max_seq_len": 16384
        },
        "xl": {  # S6로 추가 가능한 대형 모델
            "hidden_dims": [384, 768, 1152, 1536],
            "s6_layers": [4, 6, 8, 8],
            "d_state": 384,
            "d_head": 192,
            "chunk_size": 768,
            "max_seq_len": 32768
        }
    })
    
    # S6 초기화 설정
    init_method: str = "xavier_uniform"
    init_gain: float = 1.0
    bias_init: float = 0.0
    
    # S6 정규화 설정
    use_layer_norm: bool = True
    use_group_norm: bool = True
    norm_eps: float = 1e-5
    
    # S6 드롭아웃 설정
    dropout: float = 0.1
    attention_dropout: float = 0.1
    path_dropout: float = 0.0  # Stochastic depth
    
    # S6 활성화 함수
    activation: str = "silu"  # S6에 최적화된 활성화
    
    def get_model_config(self) -> Dict[str, Any]:
        """현재 모델 크기에 대한 설정 반환"""
        return self.model_configs.get(self.model_size, self.model_configs["base"])
    
    def update_for_model_size(self, target_config: dict):
        """모델 크기에 따라 설정 업데이트"""
        config = self.get_model_config()
        for key, value in config.items():
            if hasattr(target_config, key):
                setattr(target_config, key, value)



def create_s6_ssm_config(
    model_size: str = "base",
    stage: str = "B",
    audio_duration: float = 10.0,
    batch_size: int = 4,
    chunk_size: int = None,
    enable_s6_optimizations: bool = True,
    **kwargs
) -> S6SSMConfig:
    """
    완전한 S6-SSM 설정 생성
    
    Args:
        model_size: "small", "base", "large", "xl"
        stage: 학습 단계 "B", "C", "D", "E"
        audio_duration: 오디오 길이 (초)
        batch_size: 배치 크기
        chunk_size: S6 청크 크기 (None이면 자동 설정)
        enable_s6_optimizations: S6 최적화 활성화
        **kwargs: 추가 설정 오버라이드
    
    Returns:
        완전한 S6SSMConfig 인스턴스
    """
    config = S6SSMConfig()
    
    # 기본 파라미터 설정
    config.audio_duration = audio_duration
    config.batch_size = batch_size
    config.enable_s6_optimizations = enable_s6_optimizations
    
    # 모델 크기별 설정 적용
    model_config = S6ModelConfig()
    model_config.model_size = model_size
    model_config.update_for_model_size(config)
    
    # 청크 크기 자동 설정
    if chunk_size is None:
        size_to_chunk = {"small": 128, "base": 256, "large": 512, "xl": 768}
        config.chunk_size = size_to_chunk.get(model_size, 256)
    else:
        config.chunk_size = chunk_size
    
    # 스테이지별 설정 적용
    if stage in config.stage_configs:
        stage_config = config.stage_configs[stage]
        for key, value in stage_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # 추가 오버라이드 적용
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown S6 configuration parameter: {key}")
    
    return config


def get_s6_high_quality_config() -> S6SSMConfig:
    """S6 고품질 음악 처리 설정"""
    return create_s6_ssm_config(
        model_size="large",
        stage="E",
        audio_duration=20.0,
        batch_size=2,  # 대형 모델이므로 작은 배치
        chunk_size=512,
        flow_steps=16,
        d_state=256,
        max_seq_len=16384,
        enable_s6_optimizations=True,
        adaptive_chunking=True,
        use_mem_eff_path=True
    )


def get_s6_efficient_config() -> S6SSMConfig:
    """S6 효율적 학습/추론 설정"""
    return create_s6_ssm_config(
        model_size="small",
        stage="B",
        audio_duration=8.0,
        batch_size=8,  # 작은 모델이므로 큰 배치
        chunk_size=128,
        flow_steps=6,
        d_state=64,
        max_seq_len=4096,
        enable_s6_optimizations=True,
        adaptive_chunking=False,  # 효율성을 위해 고정 청킹
        scheduler_type='s6_linear'
    )


def get_s6_research_config() -> S6SSMConfig:
    """S6 연구 및 실험 설정"""
    return create_s6_ssm_config(
        model_size="base",
        stage="D",
        audio_duration=15.0,
        batch_size=4,
        chunk_size=384,  # 실험을 위한 중간 크기
        flow_steps=12,
        d_state=128,
        max_seq_len=12288,
        enable_s6_optimizations=True,
        adaptive_chunking=True,
        track_s6_metrics=True,
        log_chunk_efficiency=True,
        save_s6_metrics=True
    )


def get_s6_xl_config() -> S6SSMConfig:
    """S6 XL 모델 설정 (최고 품질)"""
    return create_s6_ssm_config(
        model_size="xl",
        stage="E",
        audio_duration=30.0,
        batch_size=1,  # XL 모델이므로 매우 작은 배치
        chunk_size=768,
        flow_steps=20,
        d_state=384,
        max_seq_len=32768,
        enable_s6_optimizations=True,
        adaptive_chunking=True,
        use_mem_eff_path=True,
        gradient_checkpointing=True,
        mixed_precision=True
    )



class S6PerformancePresets:
    """S6 성능 프리셋 관리"""
    
    @staticmethod
    def get_speed_optimized() -> Dict[str, Any]:
        """속도 최적화 S6 설정"""
        return {
            "model_size": "small",
            "chunk_size": 128,
            "flow_steps": 6,
            "solver_type": "s6_euler",
            "scheduler_type": "s6_linear",
            "adaptive_chunking": False,
            "mixed_precision": True,
            "torch_compile": True,
            "compile_mode": "max-autotune"
        }
    
    @staticmethod
    def get_memory_optimized() -> Dict[str, Any]:
        """메모리 최적화 S6 설정"""
        return {
            "chunk_size": 64,
            "gradient_checkpointing": True,
            "memory_efficient": True,
            "batch_size": 1,
            "use_mem_eff_path": True,
            "adaptive_chunking": True,
            "streaming_processing": True
        }
    
    @staticmethod
    def get_quality_optimized() -> Dict[str, Any]:
        """품질 최적화 S6 설정"""
        return {
            "model_size": "large",
            "chunk_size": 512,
            "flow_steps": 16,
            "solver_type": "s6_adaptive",
            "scheduler_type": "s6_advanced",
            "cfg_scale": 2.0,
            "d_state": 256,
            "use_self_conditioning": True
        }
    
    @staticmethod
    def get_balanced() -> Dict[str, Any]:
        """균형잡힌 S6 설정"""
        return {
            "model_size": "base",
            "chunk_size": 256,
            "flow_steps": 10,
            "solver_type": "s6_heun",
            "scheduler_type": "s6_cosine",
            "cfg_scale": 1.5,
            "adaptive_chunking": True,
            "mixed_precision": True
        }



def validate_s6_config(config: S6SSMConfig) -> Tuple[bool, List[str]]:
    """S6 설정 검증"""
    errors = []
    
    # 기본 검증
    if config.chunk_size <= 0:
        errors.append("chunk_size must be positive")
    
    if config.chunk_size > config.max_seq_len:
        errors.append("chunk_size cannot be larger than max_seq_len")
    
    if config.d_state <= 0:
        errors.append("d_state must be positive")
    
    if config.flow_steps <= 0:
        errors.append("flow_steps must be positive")
    
    # S6 특화 검증
    if config.enable_s6_optimizations:
        if config.chunk_size < 64:
            errors.append("S6 optimizations require chunk_size >= 64")
        
        if config.max_seq_len < config.chunk_size * 2:
            errors.append("S6 optimizations require max_seq_len >= chunk_size * 2")
    
    # 메모리 검증
    estimated_memory = (
        config.batch_size * 
        config.max_seq_len * 
        config.hidden_dims[-1] * 
        4  # float32 bytes
    ) / (1024**3)  # GB
    
    if estimated_memory > 24:  # 24GB limit
        errors.append(f"Estimated memory usage ({estimated_memory:.1f}GB) exceeds 24GB limit")
    
    # 성능 검증
    if config.chunk_size % 16 != 0:
        errors.append("chunk_size should be multiple of 16 for optimal S6 performance")
    
    return len(errors) == 0, errors


def optimize_s6_config_for_hardware(
    config: S6SSMConfig, 
    gpu_memory_gb: float,
    num_gpus: int = 1
) -> S6SSMConfig:
    """하드웨어에 맞게 S6 설정 최적화"""
    
    # 메모리 기반 배치 크기 조정
    if gpu_memory_gb <= 8:
        config.batch_size = 1
        config.chunk_size = min(config.chunk_size, 128)
        config.model_size = "small"
    elif gpu_memory_gb <= 16:
        config.batch_size = min(config.batch_size, 2)
        config.chunk_size = min(config.chunk_size, 256)
    elif gpu_memory_gb >= 24:
        config.batch_size = min(config.batch_size * 2, 8)
        config.chunk_size = min(config.chunk_size * 2, 512)
    
    # 멀티 GPU 최적화
    if num_gpus > 1:
        config.batch_size = config.batch_size * num_gpus
        config.num_workers = config.num_workers * num_gpus
    
    # 메모리 최적화 설정
    if gpu_memory_gb <= 12:
        config.gradient_checkpointing = True
        config.memory_efficient = True
        config.use_mem_eff_path = True
    
    return config


print("S6-optimized Configuration system ready!")
print("Key S6 configuration features:")
print("- S6 State Space Model parameters")
print("- Chunk processing configurations") 
print("- Memory optimization settings")
print("- Performance presets for different use cases")
print("- Hardware-specific optimization")
print("- Stage-specific S6 configurations")
print("- Comprehensive validation system")