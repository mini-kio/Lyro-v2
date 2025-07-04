# lyro/inference/__init__.py
"""
LYRO 추론 패키지 (수정됨 - Generator 전용)
Latent Vector 생성 파이프라인과 생성기
"""

from .pipeline import (
    LyroPipeline,
    GenerationConfig,
    GenerationInput
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
    'create_simple_pipeline',
    'quick_generate',
    
    # Generators
    'LyroGenerator',
    'SampleGenerator',
    'BatchGenerator',
]


def create_simple_pipeline(
    generator_checkpoint: str,
    device: str = "auto"
):
    """간단한 파이프라인 생성 헬퍼 (수정됨 - Generator 전용)"""
    from training.config import LyroConfig
    
    # 기본 설정
    config = LyroConfig()
    
    # 디바이스 설정
    if device == "auto":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generator 전용 파이프라인 생성
    pipeline = LyroPipeline.from_pretrained(
        generator_checkpoint=generator_checkpoint,
        cache_dir="checkpoints",
        device=device
    )
    
    return pipeline


def quick_generate(
    generator_checkpoint: str,
    lyrics: str = None,
    caption: str = None,
    duration: float = 10.0,
    output_path: str = None
):
    """빠른 생성 헬퍼 함수 (수정됨 - Latent 출력)"""
    # 파이프라인 생성
    pipeline = create_simple_pipeline(generator_checkpoint)
    
    # 입력 준비
    generation_input = GenerationInput(
        task="SONG" if lyrics else "INST",
        lyrics=lyrics,
        caption=caption
    )
    
    # 생성 설정
    generation_config = GenerationConfig(
        duration=duration,
        quality="standard"
    )
    
    # 생성
    result = pipeline.generate(generation_input, generation_config)
    
    # 저장
    if output_path:
        pipeline.save_latents(
            latents=result['latents'],
            output_path=output_path,
            metadata=result.get('metadata')
        )
    
    return result