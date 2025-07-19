#!/usr/bin/env python3
"""
LYRO v2 - Multimodal Music Generation System
오디오-가사 정렬 + TTS 보조 학습 + 음악 생성 통합 시스템
"""

__version__ = "2.0.0"
__author__ = "LYRO Team"
__description__ = "Multimodal Music Generation with Audio-Lyrics Alignment"

# Core Components (주석 처리 - 패키지 내부에서만 import 가능)
# from .models.multimodal_lyro import MultimodalLyroSystem
# from .models.generator import GeneratorConfig
# from .models.encoders import UnifiedTextEncoder
# from .models.alignment import AudioLyricsAligner
# from .training.multimodal_trainer import MultimodalTrainer
# from .training.config import LyroConfig

# # Inference
# from .inference.pipeline import MultimodalLyroPipeline, GenerationConfig, GenerationInput
# from .inference.pipeline import create_pipeline, quick_generate

# # Utilities
# from .utils.audio import AudioProcessor
# from .utils.metrics import MetricCalculator
# from .data.processor import DataProcessor

__all__ = [
    # Core
    'MultimodalLyroSystem',
    'GeneratorConfig',
    'UnifiedTextEncoder', 
    'AudioLyricsAligner',
    'MultimodalTrainer',
    'LyroConfig',
    
    # Inference
    'MultimodalLyroPipeline',
    'GenerationConfig',
    'GenerationInput',
    'create_pipeline',
    'quick_generate',
    
    # Utilities
    'AudioProcessor',
    'MetricCalculator',
    'DataProcessor',
]

def get_model_info():
    """모델 정보 반환"""
    return {
        'version': __version__,
        'description': __description__,
        'components': {
            'text_encoder': 'Unified (Lyrics + Style with Cross-attention)',
            'alignment_system': 'MERT-v1-330M + mHuBERT-147',
            'generator': 'SSM + Flow Matching (800M parameters)',
            'tts_integration': 'Alignment-guided auxiliary training',
            'training_pipeline': '4-stage progressive (pre-training → alignment → joint → fine-tuning)'
        },
        'features': [
            'Audio-lyrics alignment with confidence scoring',
            'Multi-tag lyric structure support ([section:instrument:mood])',
            'Cross-modal attention (text ↔ audio)',
            'Dynamic vocabulary building',
            'TTS-enhanced training pipeline',
            'Multi-task generation (SONG/INST/COVER)',
            'Multilingual support (147 languages via mHuBERT)',
            'Real-time alignment feedback'
        ],
        'architecture_flow': 'Text + Audio → Alignment → Cross-attention → Generator → Music'
    }

def create_multimodal_system(
    generator_config: GeneratorConfig = None,
    training_mode: str = "joint"
) -> MultimodalLyroSystem:
    """
    멀티모달 LYRO 시스템 생성 헬퍼
    
    Args:
        generator_config: Generator 설정
        training_mode: 훈련 모드 (joint, separate, alignment_only)
    
    Returns:
        MultimodalLyroSystem 인스턴스
    """
    if generator_config is None:
        generator_config = GeneratorConfig()
    
    return MultimodalLyroSystem(
        generator_config=generator_config,
        training_mode=training_mode
    )

def create_inference_pipeline(
    model_checkpoint: str = None,
    device: str = "auto"
) -> MultimodalLyroPipeline:
    """
    추론 파이프라인 생성 헬퍼
    
    Args:
        model_checkpoint: 모델 체크포인트 경로
        device: 디바이스 설정
    
    Returns:
        MultimodalLyroPipeline 인스턴스
    """
    return create_pipeline(
        model_checkpoint=model_checkpoint,
        device=device
    )

# 버전 호환성 체크
def check_compatibility():
    """시스템 호환성 체크"""
    import torch
    import numpy as np
    
    compatibility = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'multimodal_ready': True,
        'recommended_torch': '>=1.12.0',
        'recommended_cuda': '>=11.6'
    }
    
    if torch.cuda.is_available():
        compatibility['cuda_version'] = torch.version.cuda
        compatibility['gpu_count'] = torch.cuda.device_count()
        compatibility['gpu_memory'] = [
            torch.cuda.get_device_properties(i).total_memory // (1024**3) 
            for i in range(torch.cuda.device_count())
        ]
    
    return compatibility

# 시스템 정보 출력
if __name__ == "__main__":
    print("🎵 LYRO v2 - Multimodal Music Generation System")
    print("=" * 60)
    
    model_info = get_model_info()
    print(f"Version: {model_info['version']}")
    print(f"Description: {model_info['description']}")
    print()
    
    print("Components:")
    for component, description in model_info['components'].items():
        print(f"  • {component}: {description}")
    print()
    
    print("Features:")
    for feature in model_info['features']:
        print(f"  ✓ {feature}")
    print()
    
    print("Architecture Flow:")
    print(f"  {model_info['architecture_flow']}")
    print()
    
    # 호환성 체크
    compat = check_compatibility()
    print("System Compatibility:")
    print(f"  • PyTorch: {compat['torch_version']}")
    print(f"  • CUDA Available: {compat['cuda_available']}")
    if compat['cuda_available']:
        print(f"  • CUDA Version: {compat.get('cuda_version', 'Unknown')}")
        print(f"  • GPU Count: {compat.get('gpu_count', 0)}")
        if 'gpu_memory' in compat:
            for i, mem in enumerate(compat['gpu_memory']):
                print(f"  • GPU {i} Memory: {mem}GB")
    print(f"  • Multimodal Ready: {compat['multimodal_ready']}")