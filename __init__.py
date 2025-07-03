# lyro/__init__.py (새로 생성)
"""
LYRO: 텍스트 기반 음악 생성 시스템
"""

__version__ = "1.0.0"

from . import data
from . import models  
from . import inference
from . import training
from . import utils

__all__ = ['data', 'models', 'inference', 'training', 'utils']

# ===================================

# lyro/data/__init__.py (수정)
"""
LYRO Data Package
Unified data loading and processing
"""

from .dataset import LyroDataset, create_lyro_datasets
from .processor import LyroCollator, LyroTokenizer, DataProcessor, ProcessorConfig

__all__ = [
    'LyroDataset',
    'LyroCollator', 
    'LyroTokenizer',
    'DataProcessor',
    'ProcessorConfig',
    'create_lyro_datasets'
]


def create_data_pipeline(config):
    """
    Create complete data pipeline
    
    Args:
        config: Data configuration
        
    Returns:
        dict: Data pipeline components
    """
    # Create tokenizer
    tokenizer = LyroTokenizer(
        vocab_size=config.tokenizer.vocab_size,
        max_length=config.tokenizer.max_length
    )
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_lyro_datasets(
        train_metadata=config.train_metadata,
        val_metadata=config.val_metadata,
        test_metadata=config.test_metadata,
        dataset_root=config.dataset_root,
        tokenizer=tokenizer,
        **config.dataset_kwargs
    )
    
    # Create collator
    collator = LyroCollator(
        tokenizer=tokenizer,
        max_audio_length=config.max_audio_length,
        max_text_length=config.max_text_length,
        **config.collator_kwargs
    )
    
    return {
        'tokenizer': tokenizer,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'collator': collator
    }

# ===================================

# lyro/inference/__init__.py (수정)
"""
LYRO 추론 패키지
음악 생성 파이프라인과 생성기
"""

from .pipeline import (
    LyroPipeline,
    GenerationConfig,
    GenerationInput,
    create_pipeline,
    quick_generate
)

from .generator import (
    LyroGenerator,
    SampleGenerator,
    BatchGenerator
)

__all__ = [
    # Pipeline
    'LyroPipeline',
    'GenerationConfig', 
    'GenerationInput',
    'create_pipeline',
    'quick_generate',
    
    # Generators
    'LyroGenerator',
    'SampleGenerator',
    'BatchGenerator',
]


def create_simple_pipeline(
    dcae_checkpoint: str,
    generator_checkpoint: str,
    device: str = "auto"
):
    """간단한 파이프라인 생성 헬퍼"""
    from training.config import LyroConfig
    
    # 기본 설정
    config = LyroConfig()
    
    # 디바이스 설정
    if device == "auto":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 파이프라인 생성
    pipeline = LyroPipeline.from_pretrained(
        dcae_model_name=dcae_checkpoint,
        generator_checkpoint=generator_checkpoint,
        device=device
    )
    
    return pipeline

# ===================================

# lyro/models/__init__.py (수정)
"""
LYRO Models Package (프리트레인된 DCAE + Generator 훈련)
"""

from .dcae import PretrainedDCAE, create_dcae_model
from .generator import LyroGenerator, GeneratorConfig, create_lyro_generator
from .encoders import CaptionEncoder, LyricsEncoder, ReferenceEncoder
from .losses import (
    FlowMatchingLoss, 
    REPALoss, 
    ReconstructionLoss, 
    PerceptualLoss,
    CombinedLoss
)
from .sampling import FlowMatchingSampler

__all__ = [
    # Pretrained DCAE
    'PretrainedDCAE',
    'create_dcae_model',
    
    # Generator
    'LyroGenerator',
    'GeneratorConfig',
    'create_lyro_generator',
    
    # Encoders
    'CaptionEncoder',
    'LyricsEncoder', 
    'ReferenceEncoder',
    
    # Losses (Generator용)
    'FlowMatchingLoss',
    'REPALoss',
    'ReconstructionLoss',
    'PerceptualLoss',
    'CombinedLoss',
    
    # Sampling
    'FlowMatchingSampler',
]


def create_lyro_models(config):
    """
    LYRO 모델 스위트 생성 (프리트레인된 DCAE + 훈련할 Generator)
    
    Args:
        config: 모델 설정
        
    Returns:
        dict: 모델 딕셔너리
    """
    models = {}
    
    # 프리트레인된 DCAE (훈련하지 않음)
    models['dcae'] = create_dcae_model(
        model_name=config.dcae.model_name,
        cache_dir=config.dcae.cache_dir,
        sample_rate=config.dcae.sample_rate
    )
    
    # Generator (훈련 대상)
    models['generator'] = create_lyro_generator(
        config=GeneratorConfig(
            latent_channels=config.generator.latent_channels,
            latent_time_steps=config.generator.latent_time_steps,
            d_model=config.generator.d_model,
            n_layers=config.generator.n_layers
        )
    )
    
    # 조건 인코더들
    models['caption_encoder'] = CaptionEncoder(
        model_name=config.encoder.caption_model,
        output_dim=config.encoder.caption_embed_dim
    )
    
    models['lyrics_encoder'] = LyricsEncoder(
        vocab_size=config.encoder.lyrics_vocab_size,
        embed_dim=config.encoder.lyrics_embed_dim,
        max_length=config.encoder.max_lyrics_length
    )
    
    models['reference_encoder'] = ReferenceEncoder(
        dcae_model=models['dcae']
    )
    
    return models
