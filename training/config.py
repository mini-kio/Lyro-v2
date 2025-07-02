# lyro/training/config.py
"""
LYRO Generator 설정 관리 (DCAE는 프리트레인 모델 사용)
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
    
    # 입력/출력 설정 (프리트레인된 DCAE와 매칭)
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
    """프리트레인된 DCAE 설정"""
    
    # 프리트레인된 모델 정보
    model_name: str = "ACE-Step/ACE-Step-v1-3.5B"
    subfolder: str = "music_dcae_f8c8"
    cache_dir: str = "checkpoints"
    
    # 오디오 설정
    sample_rate: int = 44100
    channels: int = 2
    
    # 모델 파라미터 (프리트레인된 모델 기준)
    latent_channels: int = 16
    compression_ratio: float = 50.0


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


@dataclass
class DataConfig:
    """데이터 설정"""
    
    # 데이터셋 경로
    dataset_root: str = "dataset/"
    train_metadata: str = "dataset/metadata/train_metadata.jsonl"
    val_metadata: str = "dataset/metadata/val_metadata.jsonl"
    test_metadata: Optional[str] = "dataset/metadata/test_metadata.jsonl"
    
    # 오디오 설정 (프리트레인된 DCAE와 매칭)
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
    
    # 증강
    use_augmentation: bool = True
    augmentation_prob: float = 0.3


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
    
    # 참조 오디오 인코더
    reference_embed_dim: int = 512
    reference_pooling: str = "attention"


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
    
    # 하드웨어
    device: str = "auto"
    world_size: int = 1
    distributed: bool = False
    
    # 기타
    seed: int = 42
    deterministic: bool = False
    
    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class LyroConfig:
    """LYRO 전체 설정 (Generator 중심)"""
    
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    dcae: PretrainedDCAEConfig = field(default_factory=PretrainedDCAEConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def validate(self) -> Dict[str, Any]:
        """설정 검증"""
        issues = []
        warnings = []
        
        # Generator와 DCAE 호환성 확인
        if self.dcae.latent_channels != self.generator.latent_channels:
            issues.append(
                f"DCAE latent_channels ({self.dcae.latent_channels}) != "
                f"Generator latent_channels ({self.generator.latent_channels})"
            )
        
        # 메모리 추정
        generator_memory = self._estimate_generator_memory()
        
        if generator_memory > 24:  # 24GB 한계
            warnings.append(f"Estimated memory usage {generator_memory:.1f}GB may exceed GPU limits")
        
        # 파라미터 수 확인
        generator_params = self.generator.estimate_parameters()
        if generator_params < 1_400_000_000 or generator_params > 1_600_000_000:
            warnings.append(
                f"Generator parameters {generator_params:,} not close to 1.5B target"
            )
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'estimated_memory_gb': generator_memory,
            'generator_parameters': generator_params,
            'dcae_compression_ratio': self.dcae.compression_ratio,
        }
    
    def _estimate_generator_memory(self) -> float:
        """Generator 메모리 추정 (GB)"""
        # 모델 메모리 (1.5B * 2 bytes for fp16)
        model_memory = 1.5 * 2 / 1000  # ~3GB
        
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
        
        return model_memory + (batch_memory + attention_memory) * 2
    
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
        """미리 정의된 설정 생성"""
        
        if preset == "development":
            # 개발용 설정 (빠른 반복)
            config = cls()
            config.generator.batch_size = 1
            config.generator.epochs = 30
            config.generator.d_model = 768  # 작은 모델
            config.generator.n_layers = 12
            config.training.save_interval = 5
            return config
        
        elif preset == "production":
            # 프로덕션 설정 (고품질)
            config = cls()
            config.generator.epochs = 150
            config.training.use_wandb = True
            return config
        
        elif preset == "small_gpu":
            # 작은 GPU용 설정
            config = cls()
            config.generator.batch_size = 1
            config.generator.d_model = 1024
            config.generator.n_layers = 16
            config.generator.gradient_accumulation_steps = 16
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


if __name__ == "__main__":
    # 설정 검증 테스트
    config = LyroConfig()
    validation = config.validate()
    
    print("LYRO Configuration Validation:")
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