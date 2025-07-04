"""
LYRO 생성 파이프라인 (수정됨 - Generator 전용)
Latent Vector 생성에 특화된 파이프라인
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import logging
import time
import warnings

from models.generator import LyroGenerator, GeneratorConfig, create_lyro_generator
from models.sampling import FlowMatchingSampler
from data.processor import DataProcessor, ProcessorConfig
from utils.audio import AudioProcessor
from utils.metrics import MetricCalculator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """생성 설정 (수정됨 - Latent 출력)"""
    # 기본 설정
    duration: float = 10.0  # 초
    sample_rate: int = 44100
    
    # Flow Matching + CFG 설정
    num_steps: int = 50
    cfg_scale: float = 15.0
    cfg_min_scale: float = 3.0
    cfg_active_ratio: float = 0.5
    
    # 품질 설정
    quality: str = "standard"  # fast, standard, high
    
    # Latent 설정
    latent_channels: int = 16
    latent_time_steps: int = 128
    
    # 시드 설정
    seed: Optional[int] = None
    
    # 출력 설정
    output_format: str = "latent"  # latent, numpy
    normalize: bool = True
    
    def apply_quality_preset(self):
        """품질 프리셋 적용"""
        if self.quality == "fast":
            self.num_steps = 20
            self.cfg_scale = 7.5
        elif self.quality == "high":
            self.num_steps = 100
            self.cfg_scale = 20.0
        # standard는 기본값 유지


@dataclass
class GenerationInput:
    """생성 입력"""
    task: str = "SONG"  # SONG, INST, COVER
    lyrics: Optional[str] = None
    caption: Optional[str] = None
    reference_latents: Optional[Union[torch.Tensor, np.ndarray]] = None
    genre: Optional[List[str]] = None
    mood: Optional[str] = None
    tempo: Optional[str] = None
    
    def validate(self) -> List[str]:
        """입력 검증"""
        errors = []
        
        if self.task == "SONG" and not self.lyrics:
            errors.append("SONG task requires lyrics")
        
        if self.task == "COVER" and self.reference_latents is None:
            errors.append("COVER task requires reference latents")
        
        if self.task == "INST" and not (self.caption or self.genre):
            errors.append("INST task requires caption or genre information")
        
        return errors


class LyroPipeline:
    """
    LYRO Latent 생성 파이프라인 (수정됨 - Generator 전용)
    """
    
    def __init__(
        self,
        generator_model: LyroGenerator,
        data_processor: DataProcessor,
        device: Optional[torch.device] = None
    ):
        self.generator_model = generator_model
        self.data_processor = data_processor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델을 디바이스로 이동 및 eval 모드
        self.generator_model = self.generator_model.to(self.device).eval()
        
        # 유틸리티 초기화
        self.audio_processor = AudioProcessor()
        self.metric_calculator = MetricCalculator(44100)
        
        logger.info(f"LYRO Latent Pipeline initialized on {self.device}")
        logger.info(f"Generator parameters: {self._count_generator_parameters():,}")
    
    @classmethod
    def from_pretrained(
        cls,
        generator_checkpoint: str,
        cache_dir: str = "checkpoints",
        device: Optional[torch.device] = None
    ) -> "LyroPipeline":
        """
        프리트레인된 Generator로부터 파이프라인 생성
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generator 로드
        if generator_checkpoint and Path(generator_checkpoint).exists():
            from models.generator import load_pretrained_generator
            generator_model = load_pretrained_generator(generator_checkpoint, device=device)
        else:
            # 새로운 Generator 생성
            generator_config = GeneratorConfig()
            generator_model = create_lyro_generator(generator_config)
            logger.warning("No generator checkpoint provided, using randomly initialized model")
        
        # 데이터 처리기
        data_processor = DataProcessor()
        
        return cls(
            generator_model=generator_model,
            data_processor=data_processor,
            device=device
        )
    
    def generate(
        self,
        input_data: GenerationInput,
        generation_config: GenerationConfig = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Latent Vector 생성 메인 함수
        """
        if generation_config is None:
            generation_config = GenerationConfig()
        
        generation_config.apply_quality_preset()
        
        # 입력 검증
        errors = input_data.validate()
        if errors:
            raise ValueError(f"Input validation failed: {errors}")
        
        # 시드 설정
        if generation_config.seed is not None:
            torch.manual_seed(generation_config.seed)
            np.random.seed(generation_config.seed)
        
        start_time = time.time()
        
        if verbose:
            print(f"🎵 Starting {input_data.task} latent generation...")
            print(f"   Duration: {generation_config.duration}s")
            print(f"   Quality: {generation_config.quality}")
            print(f"   CFG Scale: {generation_config.cfg_scale}")
            print(f"   Output: Latent vectors ({generation_config.latent_channels} channels)")
        
        # 조건 준비
        conditions = self._prepare_conditions(input_data)
        
        # 잠재 벡터 생성
        with torch.no_grad():
            generated_latents = self._generate_latents(conditions, generation_config, verbose)
        
        # 후처리
        processed_latents = self._post_process_latents(generated_latents, generation_config)
        
        generation_time = time.time() - start_time
        
        # 품질 분석
        quality_metrics = self._analyze_latent_quality(processed_latents)
        
        # 결과 구성
        result = {
            'latents': processed_latents,
            'latent_shape': processed_latents.shape,
            'duration': generation_config.duration,
            'generation_time': generation_time,
            'config': generation_config,
            'input': input_data,
            'quality_metrics': quality_metrics,
            'metadata': {
                'model': 'LYRO-Generator',
                'version': '1.0 (Latent-Only)',
                'timestamp': time.time(),
                'generator_parameters': self._count_generator_parameters(),
                'pipeline_components': {
                    'generator': 'SSM + Flow Matching',
                    'output_format': 'Latent Vectors',
                    'audio_synthesis': 'External Vocoder Required'
                }
            }
        }
        
        if verbose:
            print(f"✅ Latent generation completed in {generation_time:.2f}s")
            print(f"   Latent shape: {processed_latents.shape}")
            if quality_metrics:
                print(f"   Quality score: {quality_metrics.get('overall_quality', 'N/A'):.3f}")
        
        return result
    
    def _prepare_conditions(self, input_data: GenerationInput) -> Dict[str, Any]:
        """조건 준비"""
        conditions = {}
        
        # 태스크 타입
        conditions['task_type'] = input_data.task
        
        # 가사 처리
        if input_data.lyrics:
            lyrics_data = self.data_processor.process_text_only(input_data.lyrics, 'lyrics')
            conditions['lyrics'] = lyrics_data['tokens'].unsqueeze(0).to(self.device)
            conditions['lyrics_mask'] = lyrics_data['mask'].unsqueeze(0).to(self.device)
        else:
            conditions['lyrics'] = None
            conditions['lyrics_mask'] = None
        
        # 캡션 처리
        if input_data.caption:
            conditions['captions'] = [input_data.caption]
        elif input_data.genre or input_data.mood or input_data.tempo:
            # 장르/무드/템포로부터 캡션 생성
            caption_parts = []
            
            if input_data.genre:
                genre_str = ", ".join(input_data.genre)
                caption_parts.append(f"This is a {genre_str} song")
            else:
                caption_parts.append("This is a music piece")
            
            if input_data.mood:
                caption_parts.append(f"with a {input_data.mood} mood")
            
            if input_data.tempo:
                caption_parts.append(f"at a {input_data.tempo} tempo")
            
            caption = " ".join(caption_parts) + "."
            conditions['captions'] = [caption]
        else:
            conditions['captions'] = None
        
        # 참조 latent 처리
        if input_data.reference_latents is not None:
            reference_tensor = self._load_reference_latents(input_data.reference_latents)
            conditions['reference_audio'] = reference_tensor
        else:
            conditions['reference_audio'] = None
        
        return conditions
    
    def _generate_latents(
        self, 
        conditions: Dict[str, Any], 
        config: GenerationConfig,
        verbose: bool = True
    ) -> torch.Tensor:
        """잠재 벡터 생성"""
        # 타겟 형태 계산
        shape = (1, config.latent_channels, config.latent_time_steps)
        
        # CFG 생성
        if config.quality == "fast":
            # 빠른 생성 (CFG 없음)
            generated_latents = self.generator_model.generate_fast(
                shape=shape,
                num_steps=config.num_steps,
                device=self.device,
                **conditions
            )
        else:
            # 고품질 생성 (CFG 포함)
            generated_latents = self.generator_model.generate_with_cfg(
                shape=shape,
                cfg_scale=config.cfg_scale,
                num_steps=config.num_steps,
                device=self.device,
                verbose=verbose,
                **conditions
            )
        
        return generated_latents
    
    def _post_process_latents(self, latents: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """잠재 벡터 후처리"""
        # 길이 조정
        target_length = config.latent_time_steps
        current_length = latents.shape[-1]
        
        if current_length != target_length:
            if current_length > target_length:
                # 크롭
                start_idx = (current_length - target_length) // 2
                latents = latents[..., start_idx:start_idx + target_length]
            else:
                # 패딩
                pad_length = target_length - current_length
                latents = torch.nn.functional.pad(latents, (0, pad_length))
        
        # 정규화
        if config.normalize:
            # Latent 정규화 (각 채널별)
            for c in range(latents.shape[1]):
                channel_data = latents[:, c, :]
                mean = torch.mean(channel_data)
                std = torch.std(channel_data)
                if std > 0:
                    latents[:, c, :] = (channel_data - mean) / std
        
        # 범위 클리핑
        latents = torch.clamp(latents, -5.0, 5.0)
        
        return latents
    
    def _analyze_latent_quality(self, latents: torch.Tensor) -> Dict[str, float]:
        """잠재 벡터 품질 분석"""
        try:
            quality_metrics = {}
            
            # 통계적 특성
            mean_val = torch.mean(latents).item()
            std_val = torch.std(latents).item()
            quality_metrics['mean'] = mean_val
            quality_metrics['std'] = std_val
            
            # 다이나믹 레인지
            min_val = torch.min(latents).item()
            max_val = torch.max(latents).item()
            dynamic_range = max_val - min_val
            quality_metrics['dynamic_range'] = dynamic_range
            
            # 채널별 분산
            channel_vars = torch.var(latents, dim=-1).mean(dim=0)
            quality_metrics['channel_variance'] = torch.mean(channel_vars).item()
            
            # 시간적 변화
            temporal_diff = torch.diff(latents, dim=-1)
            temporal_variation = torch.mean(torch.abs(temporal_diff)).item()
            quality_metrics['temporal_variation'] = temporal_variation
            
            # 종합 품질 점수
            quality_score = (
                min(std_val / 2.0, 1.0) * 0.3 +  # 적절한 분산
                min(dynamic_range / 10.0, 1.0) * 0.3 +  # 충분한 다이나믹 레인지
                min(temporal_variation * 10, 1.0) * 0.4  # 적절한 시간적 변화
            )
            quality_metrics['overall_quality'] = quality_score
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Quality analysis failed: {e}")
            return {'overall_quality': 0.5}
    
    def _load_reference_latents(self, reference: Union[torch.Tensor, np.ndarray]) -> Optional[torch.Tensor]:
        """참조 latent 로드"""
        try:
            if isinstance(reference, np.ndarray):
                reference_tensor = torch.from_numpy(reference).float()
            elif isinstance(reference, torch.Tensor):
                reference_tensor = reference.float()
            else:
                logger.error(f"Unsupported reference type: {type(reference)}")
                return None
            
            # 차원 정규화
            if reference_tensor.dim() == 2:
                reference_tensor = reference_tensor.unsqueeze(0)  # (C, T) -> (1, C, T)
            elif reference_tensor.dim() == 1:
                reference_tensor = reference_tensor.unsqueeze(0).unsqueeze(0)  # (T,) -> (1, 1, T)
            
            # 디바이스로 이동
            reference_tensor = reference_tensor.to(self.device)
            
            return reference_tensor
            
        except Exception as e:
            logger.error(f"Failed to load reference latents: {e}")
            return None
    
    def _count_generator_parameters(self) -> int:
        """Generator 파라미터 수 계산"""
        try:
            if hasattr(self.generator_model, 'count_parameters'):
                return self.generator_model.count_parameters()
            else:
                return sum(p.numel() for p in self.generator_model.parameters() if p.requires_grad)
        except:
            return 0
    
    def save_latents(
        self,
        latents: torch.Tensor,
        output_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """잠재 벡터 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # numpy 형태로 변환하여 저장
            latents_np = latents.cpu().numpy()
            
            # .npy 형태로 저장
            np.save(str(output_path.with_suffix('.npy')), latents_np)
            
            # 메타데이터 저장 (JSON)
            if metadata:
                metadata_path = output_path.with_suffix('.json')
                import json
                
                # 추가 메타데이터
                metadata['latent_shape'] = list(latents_np.shape)
                metadata['latent_dtype'] = str(latents_np.dtype)
                metadata['generator_only'] = True
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Latents saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save latents: {e}")
            return False
    
    def load_latents(self, latent_path: Union[str, Path]) -> torch.Tensor:
        """잠재 벡터 로드"""
        latent_path = Path(latent_path)
        
        if not latent_path.exists():
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        
        try:
            # .npy 파일 로드
            latents_np = np.load(str(latent_path))
            latents = torch.from_numpy(latents_np).float().to(self.device)
            
            logger.info(f"Latents loaded from {latent_path}, shape: {latents.shape}")
            return latents
            
        except Exception as e:
            logger.error(f"Failed to load latents: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'generator': {
                'parameters': self._count_generator_parameters(),
                'architecture': 'SSM + Flow Matching'
            },
            'device': str(self.device),
            'pipeline_version': '1.0 (Generator-Only)',
            'output_format': 'Latent Vectors',
            'architecture_flow': 'Text → Generator → Latent Vectors',
            'audio_synthesis': 'External Vocoder Required'
        }


def create_pipeline(
    generator_checkpoint: Optional[str] = None,
    cache_dir: str = "checkpoints",
    device: str = "auto"
) -> LyroPipeline:
    """
    간단한 파이프라인 생성 헬퍼
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return LyroPipeline.from_pretrained(
        generator_checkpoint=generator_checkpoint,
        cache_dir=cache_dir,
        device=device
    )


def quick_generate(
    lyrics: str = None,
    caption: str = None,
    task: str = "SONG",
    duration: float = 10.0,
    quality: str = "standard",
    output_path: str = None,
    generator_checkpoint: str = None
) -> Dict[str, Any]:
    """
    빠른 latent 생성 헬퍼 함수
    """
    # 파이프라인 생성
    pipeline = create_pipeline(
        generator_checkpoint=generator_checkpoint
    )
    
    # 입력 준비
    generation_input = GenerationInput(
        task=task,
        lyrics=lyrics,
        caption=caption
    )
    
    # 생성 설정
    generation_config = GenerationConfig(
        duration=duration,
        quality=quality
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


if __name__ == "__main__":
    # 테스트
    print("Testing LYRO Generator-Only Pipeline...")
    
    # Generator 전용 파이프라인으로 latent 생성 테스트
    result = quick_generate(
        lyrics="Walking down the street tonight, the stars are shining bright",
        duration=5.0,
        quality="standard",
        output_path="test_latents.npy"
    )
    
    print(f"Latent generation completed!")
    print(f"Latent shape: {result['latents'].shape}")
    print(f"Generation time: {result['generation_time']:.2f}s")
    print(f"Quality score: {result['quality_metrics']['overall_quality']:.3f}")
    print(f"Note: Use external vocoder to convert latents to audio")