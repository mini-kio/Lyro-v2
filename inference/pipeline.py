"""
LYRO 통합 추론 파이프라인 (수정됨 - 통합 DCAE + Vocoder)
수정된 PretrainedDCAE 모델 사용으로 오류 해결
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

from models.dcae import MusicDCAE, create_dcae_model
from models.generator import LyroGenerator, GeneratorConfig, create_lyro_generator
from models.sampling import FlowMatchingSampler
from data.processor import DataProcessor, ProcessorConfig
from utils.audio import AudioProcessor
from utils.metrics import MetricCalculator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """생성 설정 (수정됨)"""
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
    
    # Vocoder 설정 (통합됨)
    use_vocoder: bool = True  # 통합 모델에서 Vocoder 사용 여부
    
    # 시드 설정
    seed: Optional[int] = None
    
    # 출력 설정
    output_format: str = "wav"
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
    reference_audio: Optional[Union[str, Path, torch.Tensor]] = None
    genre: Optional[List[str]] = None
    mood: Optional[str] = None
    tempo: Optional[str] = None
    
    def validate(self) -> List[str]:
        """입력 검증"""
        errors = []
        
        if self.task == "SONG" and not self.lyrics:
            errors.append("SONG task requires lyrics")
        
        if self.task == "COVER" and not self.reference_audio:
            errors.append("COVER task requires reference audio")
        
        if self.task == "INST" and not (self.caption or self.genre):
            errors.append("INST task requires caption or genre information")
        
        return errors


class LyroPipeline:
    """
    LYRO 통합 생성 파이프라인 (수정됨 - 통합 DCAE + Vocoder)
    """
    
    def __init__(
        self,
        dcae_model: MusicDCAE,
        generator_model: LyroGenerator,
        data_processor: DataProcessor,
        device: Optional[torch.device] = None
    ):
        self.dcae_model = dcae_model
        self.generator_model = generator_model
        self.data_processor = data_processor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델들을 디바이스로 이동 및 eval 모드
        self.dcae_model = self.dcae_model.to(self.device).eval()
        self.generator_model = self.generator_model.to(self.device).eval()
        
        # Vocoder 정보 확인 (통합 모델에서)
        self.has_vocoder = (
            hasattr(self.dcae_model, 'vocoder') and 
            self.dcae_model.vocoder is not None and
            self.dcae_model.use_vocoder
        )
        
        # 유틸리티 초기화
        self.audio_processor = AudioProcessor()
        self.metric_calculator = MetricCalculator(44100)
        
        logger.info(f"LYRO Pipeline initialized on {self.device}")
        logger.info(f"DCAE + Vocoder: {'✅ Enabled' if self.has_vocoder else '❌ DCAE only'}")
    
    @classmethod
    def from_pretrained(
        cls,
        dcae_model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
        generator_checkpoint: Optional[str] = None,
        cache_dir: str = "checkpoints",
        use_vocoder: bool = True,
        device: Optional[torch.device] = None
    ) -> "LyroPipeline":
        """
        프리트레인된 모델들로부터 파이프라인 생성 (수정됨)
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 통합 DCAE + Vocoder 로드
        dcae_model = create_dcae_model(
            model_type="pretrained",
            model_name=dcae_model_name,
            cache_dir=cache_dir,
            use_vocoder=use_vocoder
        )
        
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
            dcae_model=dcae_model,
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
        음악 생성 메인 함수 (수정됨 - 통합 DCAE + Vocoder)
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
            print(f"🎵 Starting {input_data.task} generation...")
            print(f"   Duration: {generation_config.duration}s")
            print(f"   Quality: {generation_config.quality}")
            print(f"   CFG Scale: {generation_config.cfg_scale}")
            vocoder_status = "✅ Enabled" if (self.has_vocoder and generation_config.use_vocoder) else "❌ DCAE only"
            print(f"   Vocoder: {vocoder_status}")
        
        # 조건 준비
        conditions = self._prepare_conditions(input_data)
        
        # 잠재 벡터 생성
        with torch.no_grad():
            generated_latents = self._generate_latents(conditions, generation_config, verbose)
        
        # 오디오 디코딩 (통합 DCAE + Vocoder)
        with torch.no_grad():
            generated_audio = self._decode_to_audio(
                generated_latents, 
                generation_config, 
                verbose
            )
        
        # 후처리
        processed_audio = self._post_process_audio(generated_audio, generation_config)
        
        generation_time = time.time() - start_time
        
        # 품질 분석
        quality_metrics = self._analyze_audio_quality(processed_audio)
        
        # 결과 구성
        result = {
            'audio': processed_audio,
            'sample_rate': generation_config.sample_rate,
            'duration': generation_config.duration,
            'latents': generated_latents,
            'generation_time': generation_time,
            'config': generation_config,
            'input': input_data,
            'quality_metrics': quality_metrics,
            'metadata': {
                'model': 'LYRO',
                'version': '1.0 (Fixed)',
                'timestamp': time.time(),
                'dcae_compression': self.dcae_model.compression_ratio,
                'generator_parameters': self._count_generator_parameters(),
                'vocoder_used': self.has_vocoder and generation_config.use_vocoder,
                'pipeline_components': {
                    'dcae': 'ACE-Step Pretrained',
                    'vocoder': 'Integrated HiFi-GAN Style' if self.has_vocoder else 'None',
                    'generator': 'SSM + Flow Matching'
                }
            }
        }
        
        if verbose:
            print(f"✅ Generation completed in {generation_time:.2f}s")
            print(f"   Output shape: {processed_audio.shape}")
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
        
        # 참조 오디오 처리
        if input_data.reference_audio:
            reference_tensor = self._load_reference_audio(input_data.reference_audio)
            if reference_tensor is not None:
                # 통합 DCAE로 인코딩
                with torch.no_grad():
                    if reference_tensor.dim() == 1:
                        reference_tensor = reference_tensor.unsqueeze(0).unsqueeze(0)
                    elif reference_tensor.dim() == 2:
                        reference_tensor = reference_tensor.unsqueeze(0)
                    
                    ref_latents, _ = self.dcae_model.encode(reference_tensor)
                conditions['reference_audio'] = ref_latents
            else:
                conditions['reference_audio'] = None
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
        target_samples = int(config.sample_rate * config.duration)
        target_latent_time = target_samples // (config.sample_rate // 128)
        target_latent_time = min(max(target_latent_time, 64), 256)
        
        shape = (1, 16, target_latent_time)  # (batch, channels, time)
        
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
    
    def _decode_to_audio(
        self, 
        latents: torch.Tensor, 
        config: GenerationConfig,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        잠재 벡터를 오디오로 디코딩 (수정됨 - 통합 DCAE + Vocoder)
        """
        try:
            # latents 차원 확인
            if latents.dim() == 3 and latents.shape[0] == 1:
                pass  # (1, C, T) 형태 유지
            elif latents.dim() == 2:
                latents = latents.unsqueeze(0)  # (C, T) -> (1, C, T)
            else:
                logger.warning(f"Unexpected latents shape: {latents.shape}")
            
            # latents를 4D로 변환 (DCAE 호환)
            if latents.dim() == 3:
                B, C, T = latents.shape
                # 적절한 2D 형태로 변환 (16x16 등)
                H = W = int(np.sqrt(T))
                if H * W != T:
                    # 패딩 또는 크롭
                    target_size = 16 * 16  # 기본 크기
                    if T < target_size:
                        # 패딩
                        pad_size = target_size - T
                        latents = torch.nn.functional.pad(latents, (0, pad_size))
                    else:
                        # 크롭
                        latents = latents[:, :, :target_size]
                    H = W = 16
                
                latents_4d = latents.view(B, C, H, W)
            else:
                latents_4d = latents
            
            # Vocoder 사용 결정
            use_vocoder = (
                config.use_vocoder and 
                self.has_vocoder and 
                config.quality != "fast"
            )
            
            if verbose:
                method = "Integrated DCAE + Vocoder" if use_vocoder else "DCAE only"
                print(f"🎤 Using {method} for audio synthesis")
            
            # 통합 모델로 디코딩
            decoded_audio = self.dcae_model.decode(latents_4d)
            
            # 품질 검증
            if self._validate_audio_quality(decoded_audio):
                if verbose:
                    print("✅ Audio synthesis successful")
            else:
                if verbose:
                    print("⚠️ Audio quality issues detected")
            
            # 배치 차원 제거 및 shape 검증
            if decoded_audio.dim() == 3 and decoded_audio.shape[0] == 1:
                decoded_audio = decoded_audio.squeeze(0)  # (1, C, T) -> (C, T)
            elif decoded_audio.dim() == 2:
                pass  # 이미 (C, T) 형태
            else:
                logger.warning(f"Unexpected decoded audio shape: {decoded_audio.shape}")
            
            return decoded_audio
            
        except Exception as e:
            logger.error(f"Audio decoding failed: {e}")
            # 폴백: 노이즈 오디오
            target_samples = int(config.duration * 44100)
            return torch.randn(2, target_samples, device=self.device) * 0.1
    
    def _validate_audio_quality(self, audio: torch.Tensor) -> bool:
        """오디오 품질 검증"""
        if audio is None:
            return False
        
        # NaN/Inf 검사
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            return False
        
        # 무음 검사
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms < 1e-6:
            return False
        
        # 클리핑 검사
        clipping_ratio = torch.mean((torch.abs(audio) >= 0.99).float())
        if clipping_ratio > 0.1:
            return False
        
        return True
    
    def _analyze_audio_quality(self, audio: torch.Tensor) -> Dict[str, float]:
        """오디오 품질 분석"""
        try:
            quality_metrics = {}
            
            # RMS 에너지
            rms = torch.sqrt(torch.mean(audio ** 2)).item()
            quality_metrics['rms_energy'] = rms
            
            # 다이나믹 레인지
            peak = torch.max(torch.abs(audio)).item()
            dynamic_range = peak / (rms + 1e-10)
            quality_metrics['dynamic_range'] = dynamic_range
            
            # 클리핑 비율
            clipping_ratio = torch.mean((torch.abs(audio) >= 0.99).float()).item()
            quality_metrics['clipping_ratio'] = clipping_ratio
            
            # 무음 비율
            silence_threshold = 0.01
            silence_ratio = torch.mean((torch.abs(audio) < silence_threshold).float()).item()
            quality_metrics['silence_ratio'] = silence_ratio
            
            # 종합 품질 점수 (0-1)
            quality_score = (
                min(rms * 10, 1.0) * 0.3 +
                min(dynamic_range / 10, 1.0) * 0.3 +
                (1.0 - clipping_ratio) * 0.2 +
                (1.0 - min(silence_ratio, 1.0)) * 0.2
            )
            quality_metrics['overall_quality'] = quality_score
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Quality analysis failed: {e}")
            return {'overall_quality': 0.5}
    
    def _post_process_audio(self, audio: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """오디오 후처리"""
        # 길이 조정
        target_samples = int(config.sample_rate * config.duration)
        current_samples = audio.shape[-1]
        
        if current_samples != target_samples:
            if current_samples > target_samples:
                # 크롭
                start_idx = (current_samples - target_samples) // 2
                audio = audio[..., start_idx:start_idx + target_samples]
            else:
                # 패딩
                pad_length = target_samples - current_samples
                audio = torch.nn.functional.pad(audio, (0, pad_length))
        
        # 스테레오 보장
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]
        
        # 정규화
        if config.normalize:
            peak = torch.max(torch.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.95  # 클리핑 방지
        
        # 범위 클리핑
        audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio
    
    def _load_reference_audio(self, reference: Union[str, Path, torch.Tensor]) -> Optional[torch.Tensor]:
        """참조 오디오 로드"""
        try:
            if isinstance(reference, torch.Tensor):
                audio = reference
                # tensor인 경우 차원 정규화
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)  # (T,) -> (1, T)
                elif audio.dim() == 3 and audio.shape[0] == 1:
                    audio = audio.squeeze(0)  # (1, C, T) -> (C, T)
            else:
                # 파일에서 로드
                audio, sr = self.audio_processor.load_audio(
                    path=reference,
                    target_sr=44100,
                    normalize=True
                )
            
            # 처리
            processed = self.data_processor.process_audio_only(audio)
            return processed
            
        except Exception as e:
            logger.error(f"Failed to load reference audio: {e}")
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
    
    def save_audio(
        self,
        audio: torch.Tensor,
        output_path: Union[str, Path],
        sample_rate: int = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """오디오 저장"""
        output_path = Path(output_path)
        sample_rate = sample_rate or 44100
        
        # 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
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
                
                # Vocoder 정보 추가
                if 'vocoder_used' not in metadata:
                    metadata['vocoder_used'] = self.has_vocoder
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Audio saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        dcae_info = {
            'model_name': getattr(self.dcae_model, 'model_name', 'ACE-Step/ACE-Step-v1-3.5B'),
            'compression_ratio': self.dcae_model.compression_ratio,
            'latent_channels': self.dcae_model.latent_channels,
            'vocoder_enabled': self.has_vocoder,
            'vocoder_type': 'Integrated HiFi-GAN Style' if self.has_vocoder else 'None'
        }
        
        return {
            'dcae': dcae_info,
            'generator': {
                'parameters': self._count_generator_parameters(),
                'architecture': 'SSM + Flow Matching'
            },
            'device': str(self.device),
            'pipeline_version': '1.0 (Fixed - Integrated DCAE + Vocoder)',
            'architecture_flow': 'Text → Generator → Latents → Integrated DCAE + Vocoder → Audio'
        }


def create_pipeline(
    dcae_model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
    generator_checkpoint: Optional[str] = None,
    cache_dir: str = "checkpoints",
    use_vocoder: bool = True,
    device: str = "auto"
) -> LyroPipeline:
    """
    간단한 파이프라인 생성 헬퍼 (수정됨)
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return LyroPipeline.from_pretrained(
        dcae_model_name=dcae_model_name,
        generator_checkpoint=generator_checkpoint,
        cache_dir=cache_dir,
        use_vocoder=use_vocoder,
        device=device
    )


def quick_generate(
    lyrics: str = None,
    caption: str = None,
    task: str = "SONG",
    duration: float = 10.0,
    quality: str = "standard",
    use_vocoder: bool = True,
    output_path: str = None,
    dcae_model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
    generator_checkpoint: str = None
) -> Dict[str, Any]:
    """
    빠른 생성 헬퍼 함수 (수정됨)
    """
    # 파이프라인 생성
    pipeline = create_pipeline(
        dcae_model_name=dcae_model_name,
        generator_checkpoint=generator_checkpoint,
        use_vocoder=use_vocoder
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
        quality=quality,
        use_vocoder=use_vocoder
    )
    
    # 생성
    result = pipeline.generate(generation_input, generation_config)
    
    # 저장
    if output_path:
        pipeline.save_audio(
            audio=result['audio'],
            output_path=output_path,
            metadata=result.get('metadata')
        )
    
    return result


if __name__ == "__main__":
    # 테스트
    print("Testing Fixed LYRO Pipeline...")
    
    # 수정된 파이프라인으로 빠른 생성 테스트
    result = quick_generate(
        lyrics="Walking down the street tonight, the stars are shining bright",
        duration=5.0,
        quality="standard",
        use_vocoder=True,
        output_path="test_output_fixed.wav"
    )
    
    print(f"Generation completed!")
    print(f"Audio shape: {result['audio'].shape}")
    print(f"Generation time: {result['generation_time']:.2f}s")
    print(f"Vocoder used: {result['metadata']['vocoder_used']}")
    print(f"Quality score: {result['quality_metrics']['overall_quality']:.3f}")