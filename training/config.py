# lyro/training/config.py
"""
LYRO Generator 설정 관리 (수정됨 - 통합 DCAE + Vocoder 사용)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import os
from pathlib import Path


@dataclass
class GeneratorConfig:
    """Generator 모델 설정 (1.5B 파라미터)"""
    
    # 모델 아키텍처 (1.5B 타겟)
    d_model: int = 1536
    n_layers: int = 24
    n_heads: int = 24
    d_ff: int = 6144
    
    # SSM 설정
    d_state: int = 64
    d_conv: int = 4
    expand_factor: int = 2
    
    # 입출력 설정 (프리트레인된 DCAE와 매칭)
    latent_channels: int = 16
    latent_time_steps: int = 128
    
    # 컨디션 설정
    max_lyrics_length: int = 512
    max_caption_length: int = 256
    lyrics_embed_dim: int = 512
    caption_embed_dim: int = 768
    
    # Flow Matching 설정
    flow_steps: int = 50
    flow_scheduler: str = "cosine"
    sigma_min: float = 1e-4
    sigma_max: float = 1.0
    cfg_scale: float = 7.5
    
    # 훈련 설정
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 100
    warmup_epochs: int = 5
    
    # 기술적 설정
    mixed_precision: str = "fp16"
    gradient_accumulation_steps: int = 8
    grad_clip: float = 1.0
    
    def estimate_parameters(self) -> int:
        """모델 파라미터 수 추정"""
        # Transformer 파라미터
        embedding_params = self.latent_channels * self.d_model
        
        layer_params = (
            4 * self.d_model * self.d_model +  # QKV + output projection
            2 * self.d_model * self.d_ff +     # FFN
            2 * self.d_model +                 # LayerNorm
            self.d_model * self.d_state * 2    # SSM components
        )
        
        transformer_params = self.n_layers * layer_params
        output_params = self.d_model * self.latent_channels
        condition_params = (
            self.lyrics_embed_dim * self.d_model +
            self.caption_embed_dim * self.d_model
        )
        
        total = embedding_params + transformer_params + output_params + condition_params
        return total


@dataclass
class PretrainedDCAEConfig:
    """프리트레인된 DCAE + Vocoder 설정 (수정됨)"""
    
    # 프리트레인된 모델 정보
    model_name: str = "ACE-Step/ACE-Step-v1-3.5B"
    cache_dir: str = "checkpoints"
    
    # Vocoder 설정 (통합됨)
    use_vocoder: bool = True  # 통합 모델에서 Vocoder 사용 여부
    force_download: bool = False
    
    # 오디오 설정
    sample_rate: int = 44100
    channels: int = 2
    
    # 모델 파라미터 (프리트레인된 모델 기준)
    latent_channels: int = 16
    compression_ratio: float = 50.0
    
    # 품질 및 성능 설정
    enable_quality_validation: bool = True
    
    def get_memory_estimate(self) -> Dict[str, float]:
        """메모리 사용량 추정 (수정됨)"""
        estimates = {
            'dcae_encoder': 2.0,  # GB (frozen)
            'dcae_decoder': 0.3,  # GB (frozen)
        }
        
        if self.use_vocoder:
            estimates['integrated_vocoder'] = 0.8  # GB (frozen, 통합됨)
        
        estimates['total_audio_processing'] = sum(estimates.values())
        
        return estimates


@dataclass
class LossConfig:
    """Generator 손실 함수 설정"""
    
    # Flow Matching
    flow_matching_weight: float = 1.0
    
    # REPA Loss (HuBERT 기반)
    use_repa_loss: bool = True
    repa_weight: float = 0.1
    hubert_model: str = "ZhenYe234/hubert_base_general_audio"
    hubert_layers: List[int] = field(default_factory=lambda: [6, 7, 8, 9])
    
    # Perceptual Loss
    use_perceptual_loss: bool = True
    perceptual_weight: float = 0.2
    mel_weight: float = 1.0
    stft_weight: float = 0.5
    
    # Reconstruction Loss
    reconstruction_weight: float = 0.3
    
    # Vocoder-specific losses (수정됨)
    use_vocoder_quality_loss: bool = True
    vocoder_quality_weight: float = 0.05
    
    # Dynamic loss weighting
    adaptive_loss_weighting: bool = True
    loss_weight_update_interval: int = 100  # steps


@dataclass
class DataConfig:
    """데이터 설정"""
    
    # 데이터셋 경로
    dataset_root: str = "dataset/"
    train_metadata: str = "dataset/metadata/train_metadata.jsonl"
    val_metadata: str = "dataset/metadata/val_metadata.jsonl"
    test_metadata: Optional[str] = "dataset/metadata/test_metadata.jsonl"
    
    # 오디오 설정 (통합 DCAE + Vocoder와 매칭)
    sample_rate: int = 44100
    max_audio_length: int = 441000  # 10초
    min_audio_length: int = 44100   # 1초
    audio_duration: float = 10.0
    
    # 텍스트 설정
    max_lyrics_length: int = 512
    max_caption_length: int = 256
    
    # 태스크 비율
    task_ratios: Dict[str, float] = field(default_factory=lambda: {
        'SONG': 0.7,   # 가사 + 오디오
        'INST': 0.2,   # 캡션 + 오디오
        'COVER': 0.1   # 참조 + 가사 + 오디오
    })
    
    # 데이터 로딩
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # 증강 (통합 모델 고려)
    use_augmentation: bool = True
    augmentation_prob: float = 0.3
    vocoder_compatible_augmentation: bool = True


@dataclass
class EncoderConfig:
    """조건 인코더 설정"""
    
    # 가사 인코더
    lyrics_vocab_size: int = 32000
    lyrics_embed_dim: int = 512
    lyrics_hidden_dim: int = 512
    lyrics_num_layers: int = 6
    lyrics_num_heads: int = 8
    
    # 캡션 인코더
    caption_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    caption_embed_dim: int = 768
    caption_freeze: bool = True
    
    # 참조 오디오 인코더 (통합 DCAE + Vocoder 사용)
    reference_embed_dim: int = 512
    reference_pooling: str = "attention"
    reference_use_vocoder_features: bool = True


@dataclass
class TrainingConfig:
    """훈련 설정"""
    
    # 체크포인트
    checkpoint_dir: str = "checkpoints/"
    save_interval: int = 10
    eval_interval: int = 5
    keep_best: int = 5
    
    # 로깅
    use_wandb: bool = False
    wandb_project: str = "lyro"
    log_interval: int = 100
    
    # Vocoder 품질 모니터링 (수정됨)
    monitor_vocoder_quality: bool = True
    quality_check_interval: int = 50  # steps
    quality_threshold: float = 0.7  # 품질 임계값
    
    # 하드웨어 (통합 모델 메모리 사용량 고려)
    device: str = "auto"
    world_size: int = 1
    distributed: bool = False
    max_memory_usage_gb: float = 20.0  # GPU 메모리 한계 (통합 모델 포함)
    
    # 기타
    seed: int = 42
    deterministic: bool = False
    
    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class LyroConfig:
    """LYRO 전체 설정 (수정됨 - 통합 DCAE + Vocoder)"""
    
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    dcae: PretrainedDCAEConfig = field(default_factory=PretrainedDCAEConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def validate(self) -> Dict[str, Any]:
        """설정 검증 (수정됨)"""
        issues = []
        warnings = []
        
        # Generator와 DCAE 호환성 확인
        if self.dcae.latent_channels != self.generator.latent_channels:
            issues.append(
                f"DCAE latent_channels ({self.dcae.latent_channels}) != "
                f"Generator latent_channels ({self.generator.latent_channels})"
            )
        
        # 통합 Vocoder 설정 검증
        if self.dcae.use_vocoder:
            if not self.dcae.cache_dir:
                warnings.append("Cache directory not specified")
            
            if self.generator.batch_size > 2:
                warnings.append("Large batch size with Vocoder may cause OOM")
        
        # 메모리 추정 (통합 모델 포함)
        generator_memory = self._estimate_total_memory()
        
        if generator_memory > self.training.max_memory_usage_gb:
            issues.append(
                f"Estimated memory usage {generator_memory:.1f}GB exceeds limit "
                f"{self.training.max_memory_usage_gb}GB"
            )
        
        # 파라미터 수 확인
        generator_params = self.generator.estimate_parameters()
        if generator_params < 1_400_000_000 or generator_params > 1_600_000_000:
            warnings.append(
                f"Generator parameters {generator_params:,} not close to 1.5B target"
            )
        
        # 통합 모델 품질 설정 검증
        if self.dcae.use_vocoder and not self.training.monitor_vocoder_quality:
            warnings.append("Vocoder quality monitoring disabled - may miss quality issues")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'estimated_memory_gb': generator_memory,
            'generator_parameters': generator_params,
            'dcae_compression_ratio': self.dcae.compression_ratio,
            'vocoder_enabled': self.dcae.use_vocoder,
            'model_type': 'Integrated DCAE + Vocoder',
        }
    
    def _estimate_total_memory(self) -> float:
        """전체 메모리 추정 (수정됨 - 통합 DCAE + Vocoder, GB)"""
        # Generator 메모리 (1.5B * 2 bytes for fp16)
        generator_memory = 1.5 * 2 / 1000  # ~3GB
        
        # 통합 DCAE + Vocoder 메모리
        audio_processing_memory = sum(self.dcae.get_memory_estimate().values())
        
        # 배치 메모리
        batch_memory = (
            self.generator.batch_size *
            self.generator.latent_channels *
            self.generator.latent_time_steps *
            4  # float32
        ) / (1024**3)
        
        # Attention 메모리
        attention_memory = (
            self.generator.batch_size *
            self.generator.n_heads *
            self.generator.latent_time_steps ** 2 *
            4
        ) / (1024**3)
        
        # 통합 모델 오버헤드
        integration_overhead = 0.3  # 통합으로 인한 추가 메모리
        
        total = (
            generator_memory + 
            audio_processing_memory + 
            (batch_memory + attention_memory) * 2 +  # 훈련시 2배
            integration_overhead
        )
        
        return total
    
    def optimize_for_memory(self, target_memory_gb: float = 18.0):
        """메모리 제약에 맞춰 설정 최적화 (수정됨)"""
        current_memory = self._estimate_total_memory()
        
        if current_memory <= target_memory_gb:
            return  # 이미 메모리 한계 내
        
        reduction_needed = current_memory - target_memory_gb
        
        # 1. 배치 크기 감소
        if reduction_needed > 1.0 and self.generator.batch_size > 1:
            self.generator.batch_size = max(1, self.generator.batch_size - 1)
            self.generator.gradient_accumulation_steps *= 2
        
        # 2. Generator 모델 크기 축소
        if reduction_needed > 2.0:
            self.generator.d_model = min(1024, self.generator.d_model)
            self.generator.n_layers = min(16, self.generator.n_layers)
        
        # 3. 마지막 수단: Vocoder 비활성화
        if self._estimate_total_memory() > target_memory_gb:
            self.dcae.use_vocoder = False
            print("Warning: Disabled Vocoder due to memory constraints")
    
    @classmethod
    def from_yaml(cls, path: str) -> "LyroConfig":
        """YAML 파일에서 설정 로드"""
        import yaml
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_yaml(self, path: str):
        """설정을 YAML 파일로 저장"""
        import yaml
        from dataclasses import asdict
        
        config_dict = asdict(self)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def create_preset(cls, preset: str) -> "LyroConfig":
        """미리 정의된 설정 생성 (수정됨 - 통합 모델)"""
        
        if preset == "development":
            # 개발용 설정 (빠른 반복)
            config = cls()
            config.generator.batch_size = 1
            config.generator.epochs = 30
            config.generator.d_model = 768  # 작은 모델
            config.generator.n_layers = 12
            config.training.save_interval = 5
            config.dcae.use_vocoder = True  # 통합 모델이므로 유지
            return config
        
        elif preset == "production":
            # 프로덕션 설정 (고품질)
            config = cls()
            config.generator.epochs = 150
            config.training.use_wandb = True
            config.dcae.use_vocoder = True
            config.training.monitor_vocoder_quality = True
            return config
        
        elif preset == "small_gpu":
            # 작은 GPU용 설정 (메모리 최적화)
            config = cls()
            config.generator.batch_size = 1
            config.generator.d_model = 1024
            config.generator.n_layers = 16
            config.generator.gradient_accumulation_steps = 16
            config.dcae.use_vocoder = True  # 통합 모델
            config.training.max_memory_usage_gb = 16.0
            return config
        
        elif preset == "integrated_test":
            # 통합 모델 테스트용 설정
            config = cls()
            config.generator.batch_size = 1
            config.generator.epochs = 10
            config.dcae.use_vocoder = True
            config.training.monitor_vocoder_quality = True
            config.training.quality_check_interval = 10
            config.loss.use_vocoder_quality_loss = True
            return config
        
        else:
            raise ValueError(f"Unknown preset: {preset}")


# 전역 설정 인스턴스
_global_config: Optional[LyroConfig] = None


def get_config() -> LyroConfig:
    """전역 설정 가져오기"""
    global _global_config
    if _global_config is None:
        _global_config = LyroConfig()
    return _global_config


def set_config(config: LyroConfig):
    """전역 설정 설정"""
    global _global_config
    _global_config = config


def load_config(path: str) -> LyroConfig:
    """설정 파일 로드 및 전역 설정"""
    config = LyroConfig.from_yaml(path)
    set_config(config)
    return config


def create_optimized_config(
    available_memory_gb: float = 18.0,
    target_quality: str = "high",
    enable_vocoder: bool = True
) -> LyroConfig:
    """메모리와 품질 요구사항에 맞춰 최적화된 설정 생성 (수정됨)"""
    
    # 기본 설정으로 시작
    config = LyroConfig()
    
    # 통합 Vocoder 설정
    config.dcae.use_vocoder = enable_vocoder
    
    # 메모리 최적화
    config.training.max_memory_usage_gb = available_memory_gb
    config.optimize_for_memory(available_memory_gb)
    
    return config


if __name__ == "__main__":
    # 설정 검증 테스트 (수정됨 - 통합 DCAE + Vocoder)
    config = LyroConfig()
    validation = config.validate()
    
    print("LYRO Configuration Validation (Integrated DCAE + Vocoder):")
    print(f"Valid: {validation['valid']}")
    
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    print(f"Estimated memory: {validation['estimated_memory_gb']:.1f}GB")
    print(f"Generator parameters: {validation['generator_parameters']:,}")
    print(f"DCAE compression ratio: {validation['dcae_compression_ratio']:.1f}:1")
    print(f"Vocoder enabled: {validation['vocoder_enabled']}")
    print(f"Model type: {validation['model_type']}")
    
    # 메모리 최적화 테스트
    print("\nTesting memory optimization for 16GB GPU:")
    optimized_config = create_optimized_config(
        available_memory_gb=16.0,
        target_quality="standard",
        enable_vocoder=True
    )
    
    optimized_validation = optimized_config.validate()
    print(f"Optimized memory usage: {optimized_validation['estimated_memory_gb']:.1f}GB")
    print(f"Vocoder enabled: {optimized_validation['vocoder_enabled']}")
    print(f"Model type: {optimized_validation['model_type']}")