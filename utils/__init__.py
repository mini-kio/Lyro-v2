# lyro/utils/__init__.py
"""
LYRO 유틸리티 패키지
오디오 처리, 메트릭 계산, 공통 함수들
"""

from .audio import (
    AudioProcessor, 
    AudioAugmentation, 
    AudioAnalyzer,
    AudioConfig,
    ensure_audio_format,
    detect_audio_quality
)

from .metrics import (
    MetricCalculator,
    MetricResult,
    AudioQualityMetrics,
    PerceptualMetrics,
    MusicMetrics,
    TextAudioAlignmentMetrics,
    CompressionMetrics
)

__all__ = [
    # Audio utilities
    'AudioProcessor',
    'AudioAugmentation', 
    'AudioAnalyzer',
    'AudioConfig',
    'ensure_audio_format',
    'detect_audio_quality',
    
    # Metrics
    'MetricCalculator',
    'MetricResult',
    'AudioQualityMetrics',
    'PerceptualMetrics',
    'MusicMetrics', 
    'TextAudioAlignmentMetrics',
    'CompressionMetrics',
]


def get_audio_processor(sample_rate: int = 44100) -> AudioProcessor:
    """기본 오디오 프로세서 생성"""
    config = AudioConfig(sample_rate=sample_rate)
    return AudioProcessor(config)


def get_metric_calculator(sample_rate: int = 44100) -> MetricCalculator:
    """기본 메트릭 계산기 생성"""
    return MetricCalculator(sample_rate)


def safe_load_audio(path, **kwargs):
    """안전한 오디오 로딩 래퍼"""
    processor = get_audio_processor()
    try:
        return processor.load_audio(path, **kwargs)
    except Exception as e:
        print(f"Failed to load audio {path}: {e}")
        return None, None


def safe_save_audio(audio, path, **kwargs):
    """안전한 오디오 저장 래퍼"""
    processor = get_audio_processor()
    try:
        processor.save_audio(audio, path, **kwargs)
        return True
    except Exception as e:
        print(f"Failed to save audio {path}: {e}")
        return False