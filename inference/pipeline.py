# lyro/inference/pipeline.py
"""
LYRO 추론 파이프라인
DCAE + Generator를 연결한 완전한 음악 생성 파이프라인
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import logging
import time
import warnings

# LYRO 모듈
from ..models.dcae import DCAE
from ..models.generator import LyroGenerator, GeneratorConfig
from ..models.encoders import MultiModalEncoder
from ..data.tokenizer import LyroTokenizer
from ..utils.audio import AudioProcessor, ensure_audio_format
from ..utils.metrics import MetricCalculator
from ..training.config import LyroConfig

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """생성 설정"""
    # 기본 설정
    duration: float = 10.0  # 초
    sample_rate: int = 44100
    
    # Flow Matching 설정
    num_steps: int = 50
    cfg_scale: float = 7.5
    guidance_scale: float = 1.0
    
    # 품질 설정
    quality: str = "standard"  # fast, standard, high
    
    # 시드 설정
    seed: Optional[int] = None
    
    # 출력 설정
    output_format: str = "wav"  # wav, mp3, flac
    normalize: bool = True
    
    def apply_quality_preset(self):
        """품질 프리셋 적용"""
        if self.quality == "fast":
            self.num_steps = 25
            self.cfg_scale = 3.0
        elif self.quality == "high":
            self.num_steps = 100
            self.cfg_scale = 10.0
        # standard는 기본값 유지


@dataclass
class GenerationInput:
    """생성 입력"""
    # 태스크 타입
    task: str = "SONG"  # SONG, INST, COVER
    
    # 텍스트 조건
    lyrics: Optional[str] = None
    caption: Optional[str] = None
    
    # 오디오 조건
    reference_audio: Optional[Union[str, Path, torch.Tensor]] = None
    
    # 스타일 조건
    genre: Optional[List[str]] = None
    mood: Optional[str] = None
    tempo: Optional[str] = None
    
    # 고급 설정
    structure: Optional[str] = None  # verse-chorus-verse 등
    key: Optional[str] = None
    
    def validate(self) -> List[str]:
        """입력 검증"""
        errors = []
        
        if self.task == "SONG" and not self.lyrics:
            errors.append("SONG task requires lyrics")
        
        if self.task == "COVER" and not self.reference_audio:
            errors.append("COVER task requires reference audio")
        
        if self.task == "INST" and not self.caption:
            errors.append("INST task requires caption or genre information")
        
        return errors


class LyroPipeline:
    """
    LYRO 통합 생성 파이프라인
    DCAE와 Generator를 연결하여 완전한 음악 생성 시스템 제공
    """
    
    def __init__(
        self,
        dcae_model: DCAE,
        generator_model: LyroGenerator,
        tokenizer: LyroTokenizer,
        config: LyroConfig,
        device: Optional[torch.device] = None
    ):
        self.dcae_model = dcae_model
        self.generator_model = generator_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델들을 디바이스로 이동 및 eval 모드
        self.dcae_model = self.dcae_model.to(self.device).eval()
        self.generator_model = self.generator_model.to(self.device).eval()
        
        # 유틸리티 초기화
        self.audio_processor = AudioProcessor()
        self.metric_calculator = MetricCalculator(config.data.sample_rate)
        
        # 품질 프리셋
        self.quality_presets = {
            "fast": {"num_steps": 25, "cfg_scale": 3.0},
            "standard": {"num_steps": 50, "cfg_scale": 7.5},
            "high": {"num_steps": 100, "cfg_scale": 10.0}
        }
        
        logger.info("LYRO Pipeline initialized successfully")
    
    @classmethod
    def from_checkpoints(
        cls,
        dcae_checkpoint: Union[str, Path],
        generator_checkpoint: Union[str, Path],
        config: LyroConfig,
        device: Optional[torch.device] = None
    ) -> "LyroPipeline":
        """
        체크포인트에서 파이프라인 로드
        
        Args:
            dcae_checkpoint: DCAE 체크포인트 경로
            generator_checkpoint: Generator 체크포인트 경로
            config: 설정
            device: 디바이스
        
        Returns:
            초기화된 파이프라인
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # DCAE 로드
        from ..models.dcae import create_dcae
        dcae_model = create_dcae(
            sample_rate=config.dcae.sample_rate,
            latent_channels=config.dcae.latent_channels,
            target_compression_ratio=config.dcae.estimate_compression_ratio()
        )
        
        dcae_checkpoint_data = torch.load(dcae_checkpoint, map_location=device)
        dcae_model.load_state_dict(dcae_checkpoint_data['model_state_dict'])
        
        # Generator 로드
        generator_config = GeneratorConfig(
            d_model=config.generator.d_model,
            n_layers=config.generator.n_layers,
            n_heads=config.generator.n_heads,
            latent_channels=config.dcae.latent_channels,
            latent_time_steps=config.dcae.target_time_steps
        )
        
        from ..models.generator import create_lyro_generator
        generator_model = create_lyro_generator(generator_config)
        
        generator_checkpoint_data = torch.load(generator_checkpoint, map_location=device)
        generator_model.load_state_dict(generator_checkpoint_data['model_state_dict'])
        
        # 토크나이저
        tokenizer = LyroTokenizer()
        
        return cls(
            dcae_model=dcae_model,
            generator_model=generator_model,
            tokenizer=tokenizer,
            config=config,
            device=device
        )
    
    def generate(
        self,
        input_data: GenerationInput,
        generation_config: GenerationConfig = None
    ) -> Dict[str, Any]:
        """
        음악 생성 메인 함수
        
        Args:
            input_data: 생성 입력
            generation_config: 생성 설정
        
        Returns:
            생성 결과 딕셔너리
        """
        # 설정 기본값
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
        
        # 조건 준비
        conditions = self._prepare_conditions(input_data, generation_config)
        
        # 잠재 벡터 생성
        with torch.no_grad():
            generated_latents = self._generate_latents(conditions, generation_config)
        
        # 오디오 디코딩
        with torch.no_grad():
            generated_audio = self._decode_to_audio(generated_latents)
        
        # 후처리
        processed_audio = self._post_process_audio(generated_audio, generation_config)
        
        generation_time = time.time() - start_time
        
        # 결과 구성
        result = {
            'audio': processed_audio,
            'sample_rate': generation_config.sample_rate,
            'duration': generation_config.duration,
            'latents': generated_latents,
            'generation_time': generation_time,
            'config': generation_config,
            'input': input_data,
            'metadata': {
                'model': 'LYRO',
                'version': '1.0',
                'timestamp': time.time(),
                'parameters': self.generator_model.count_parameters()
            }
        }
        
        # 품질 메트릭 (참조 오디오가 있는 경우)
        if input_data.reference_audio is not None:
            reference_tensor = self._load_reference_audio(input_data.reference_audio)
            if reference_tensor is not None:
                metrics = self.metric_calculator.compute_all_metrics(
                    target_audio=reference_tensor,
                    generated_audio=processed_audio,
                    text=input_data.lyrics or input_data.caption
                )
                result['quality_metrics'] = metrics
        
        logger.info(f"Generation completed in {generation_time:.2f}s")
        
        return result
    
    def _prepare_conditions(self, input_data: GenerationInput, config: GenerationConfig) -> Dict[str, Any]:
        """조건 준비"""
        conditions = {}
        
        # 가사 처리
        if input_data.lyrics:
            lyrics_tokens = self.tokenizer.encode_lyrics(input_data.lyrics)
            conditions['lyrics'] = torch.tensor([lyrics_tokens], device=self.device)
            conditions['lyrics_mask'] = torch.ones_like(conditions['lyrics'], dtype=torch.bool)
        
        # 캡션 처리 (MusicCaps 스타일)
        if input_data.caption:
            conditions['captions'] = [input_data.caption]
        elif input_data.genre or input_data.mood or input_data.tempo:
            # 장르/무드/템포로부터 캡션 생성
            caption_parts = []
            
            if input_data.genre:
                if len(input_data.genre) == 1:
                    caption_parts.append(f"This is a {input_data.genre[0]} song")
                else:
                    caption_parts.append(f"This is a {', '.join(input_data.genre)} song")
            else:
                caption_parts.append("This is a music piece")
            
            if input_data.mood:
                caption_parts.append(f"with a {input_data.mood} mood")
            
            if input_data.tempo:
                caption_parts.append(f"at a {input_data.tempo} tempo")
            
            caption = ", ".join(caption_parts) + "."
            conditions['captions'] = [caption]
        
        # 참조 오디오 처리
        if input_data.reference_audio:
            reference_tensor = self._load_reference_audio(input_data.reference_audio)
            if reference_tensor is not None:
                # DCAE로 인코딩
                with torch.no_grad():
                    ref_latents, _ = self.dcae_model.encode(reference_tensor.unsqueeze(0))
                conditions['reference_audio'] = ref_latents
        
        # 태스크 타입
        conditions['task_type'] = input_data.task
        
        return conditions
    
    def _generate_latents(self, conditions: Dict[str, Any], config: GenerationConfig) -> torch.Tensor:
        """잠재 벡터 생성"""
        # 타겟 형태 계산
        target_samples = int(config.sample_rate * config.duration)
        target_latent_time = int(target_samples / (config.sample_rate / self.config.dcae.target_time_steps))
        
        shape = (
            1,  # batch_size
            self.config.dcae.latent_channels,
            target_latent_time
        )
        
        # 생성
        generated_latents = self.generator_model.generate(
            shape=shape,
            lyrics=conditions.get('lyrics'),
            lyrics_mask=conditions.get('lyrics_mask'),
            captions=conditions.get('captions'),
            reference_audio=conditions.get('reference_audio'),
            task_type=conditions.get('task_type'),
            num_steps=config.num_steps,
            cfg_scale=config.cfg_scale,
            device=self.device
        )
        
        return generated_latents
    
    def _decode_to_audio(self, latents: torch.Tensor) -> torch.Tensor:
        """잠재 벡터를 오디오로 디코딩"""
        try:
            decoded_audio = self.dcae_model.decode(latents)
            return decoded_audio.squeeze(0)  # 배치 차원 제거
        except Exception as e:
            logger.error(f"Audio decoding failed: {e}")
            # 폴백: 노이즈 오디오
            target_samples = int(10 * self.config.data.sample_rate)
            return torch.randn(2, target_samples, device=self.device) * 0.1
    
    def _post_process_audio(self, audio: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """오디오 후처리"""
        # 길이 조정
        target_samples = int(config.sample_rate * config.duration)
        audio = ensure_audio_format(
            audio=audio,
            target_channels=2,
            target_length=target_samples,
            sample_rate=config.sample_rate
        )
        
        # 정규화
        if config.normalize:
            audio = self.audio_processor.normalize_audio(audio, method="peak")
        
        # 클리핑 방지
        audio = torch.clamp(audio, -0.95, 0.95)
        
        return audio
    
    def _load_reference_audio(self, reference: Union[str, Path, torch.Tensor]) -> Optional[torch.Tensor]:
        """참조 오디오 로드"""
        try:
            if isinstance(reference, torch.Tensor):
                return ensure_audio_format(reference, target_channels=2)
            
            # 파일에서 로드
            audio, sr = self.audio_processor.load_audio(
                path=reference,
                target_sr=self.config.data.sample_rate,
                normalize=True
            )
            
            return ensure_audio_format(audio, target_channels=2)
            
        except Exception as e:
            logger.error(f"Failed to load reference audio: {e}")
            return None
    
    def save_audio(
        self,
        audio: torch.Tensor,
        output_path: Union[str, Path],
        sample_rate: int = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """오디오 저장"""
        output_path = Path(output_path)
        sample_rate = sample_rate or self.config.data.sample_rate
        
        # 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 오디오 저장
        self.audio_processor.save_audio(
            audio=audio,
            path=output_path,
            sample_rate=sample_rate
        )
        
        # 메타데이터 저장 (JSON)
        if metadata:
            metadata_path = output_path.with_suffix('.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Audio saved to {output_path}")
    
    def batch_generate(
        self,
        inputs: List[GenerationInput],
        generation_config: GenerationConfig = None,
        output_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """배치 생성"""
        results = []
        
        for i, input_data in enumerate(inputs):
            try:
                logger.info(f"Generating {i+1}/{len(inputs)}: {input_data.task}")
                
                result = self.generate(input_data, generation_config)
                results.append(result)
                
                # 출력 디렉토리가 있으면 저장
                if output_dir:
                    output_path = output_dir / f"generated_{i:03d}.wav"
                    self.save_audio(
                        audio=result['audio'],
                        output_path=output_path,
                        metadata=result.get('metadata')
                    )
                
            except Exception as e:
                logger.error(f"Failed to generate {i+1}: {e}")
                results.append({'error': str(e)})
        
        return results
    
    def interpolate(
        self,
        input_a: GenerationInput,
        input_b: GenerationInput,
        num_steps: int = 5,
        generation_config: GenerationConfig = None
    ) -> List[Dict[str, Any]]:
        """두 조건 간 보간 생성"""
        if generation_config is None:
            generation_config = GenerationConfig()
        
        results = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            
            # 조건 보간 (간단한 구현)
            interpolated_input = GenerationInput(
                task=input_a.task,
                lyrics=input_a.lyrics if alpha < 0.5 else input_b.lyrics,
                caption=self._interpolate_text(input_a.caption, input_b.caption, alpha),
                genre=input_a.genre if alpha < 0.5 else input_b.genre,
                mood=self._interpolate_text(input_a.mood, input_b.mood, alpha),
                tempo=input_a.tempo if alpha < 0.5 else input_b.tempo
            )
            
            result = self.generate(interpolated_input, generation_config)
            result['interpolation_alpha'] = alpha
            results.append(result)
        
        return results
    
    def _interpolate_text(self, text_a: Optional[str], text_b: Optional[str], alpha: float) -> Optional[str]:
        """텍스트 보간 (간단한 구현)"""
        if not text_a and not text_b:
            return None
        if not text_a:
            return text_b
        if not text_b:
            return text_a
        
        # 간단한 선택 기반 보간
        return text_a if alpha < 0.5 else text_b
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'dcae': {
                'parameters': sum(p.numel() for p in self.dcae_model.parameters()),
                'latent_channels': self.config.dcae.latent_channels,
                'compression_ratio': self.config.dcae.estimate_compression_ratio()
            },
            'generator': {
                'parameters': self.generator_model.count_parameters(),
                'd_model': self.config.generator.d_model,
                'n_layers': self.config.generator.n_layers
            },
            'total_parameters': (
                sum(p.numel() for p in self.dcae_model.parameters()) +
                self.generator_model.count_parameters()
            )
        }


class StreamingPipeline:
    """실시간 스트리밍 생성 파이프라인"""
    
    def __init__(self, base_pipeline: LyroPipeline, chunk_duration: float = 1.0):
        self.base_pipeline = base_pipeline
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * base_pipeline.config.data.sample_rate)
        
        # 스트리밍 상태
        self.current_context = None
        self.generated_chunks = []
    
    def start_stream(self, input_data: GenerationInput, generation_config: GenerationConfig = None):
        """스트리밍 시작"""
        self.current_context = self.base_pipeline._prepare_conditions(input_data, generation_config or GenerationConfig())
        self.generated_chunks = []
    
    def generate_chunk(self) -> torch.Tensor:
        """다음 청크 생성"""
        if self.current_context is None:
            raise RuntimeError("Stream not started. Call start_stream() first.")
        
        # 간단한 구현: 짧은 길이로 생성
        config = GenerationConfig(duration=self.chunk_duration, num_steps=25)
        
        # 컨텍스트 기반 생성
        latents = self.base_pipeline._generate_latents(self.current_context, config)
        audio_chunk = self.base_pipeline._decode_to_audio(latents)
        
        self.generated_chunks.append(audio_chunk)
        
        return audio_chunk
    
    def get_full_audio(self) -> torch.Tensor:
        """전체 생성된 오디오 반환"""
        if not self.generated_chunks:
            return torch.empty(2, 0)
        
        return torch.cat(self.generated_chunks, dim=-1)


def create_pipeline_from_config(config_path: Union[str, Path]) -> LyroPipeline:
    """설정 파일에서 파이프라인 생성"""
    from ..training.config import LyroConfig
    
    config = LyroConfig.from_yaml(config_path)
    
    # 체크포인트 경로는 설정에서 가져오거나 추론
    dcae_checkpoint = Path(config.training.checkpoint_dir) / "dcae_best.pt"
    generator_checkpoint = Path(config.training.checkpoint_dir) / "generator_best.pt"
    
    return LyroPipeline.from_checkpoints(
        dcae_checkpoint=dcae_checkpoint,
        generator_checkpoint=generator_checkpoint,
        config=config
    )
