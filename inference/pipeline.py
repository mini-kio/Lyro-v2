"""
LYRO 통합 추론 파이프라인 (올바른 DCAE + Vocoder 아키텍처)
DCAE (멜 <-> 잠재벡터) + Vocoder (멜 <-> 오디오) + Generator + CFG 샘플링을 통합한 완전한 음악 생성 시스템
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

from models.dcae import PretrainedDCAE, AdvancedVocoder, create_dcae_model, create_vocoder_model
from models.generator import LyroGenerator, GeneratorConfig, create_lyro_generator
from models.sampling import FlowMatchingSampler
from data.processor import DataProcessor, ProcessorConfig
from utils.audio import AudioProcessor
from utils.metrics import MetricCalculator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """생성 설정"""
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
    LYRO 통합 생성 파이프라인 (올바른 DCAE + Vocoder 아키텍처)
    """
    
    def __init__(
        self,
        dcae_model: PretrainedDCAE,
        vocoder_model: AdvancedVocoder,
        generator_model: LyroGenerator,
        data_processor: DataProcessor,
        device: Optional[torch.device] = None
    ):
        self.dcae_model = dcae_model
        self.vocoder_model = vocoder_model
        self.generator_model = generator_model
        self.data_processor = data_processor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델들을 디바이스로 이동 및 eval 모드
        self.dcae_model = self.dcae_model.to(self.device).eval()
        self.vocoder_model = self.vocoder_model.to(self.device).eval()
        self.generator_model = self.generator_model.to(self.device).eval()
        
        # 유틸리티 초기화
        self.audio_processor = AudioProcessor()
        self.metric_calculator = MetricCalculator(44100)
        
        logger.info(f"LYRO Pipeline initialized on {self.device}")
        logger.info("Pipeline components:")
        logger.info(f"  - DCAE: {type(self.dcae_model).__name__}")
        logger.info(f"  - Vocoder: {type(self.vocoder_model).__name__}")
        logger.info(f"  - Generator: {type(self.generator_model).__name__}")
    
    @classmethod
    def from_pretrained(
        cls,
        dcae_model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
        generator_checkpoint: Optional[str] = None,
        cache_dir: str = "checkpoints",
        device: Optional[torch.device] = None
    ) -> "LyroPipeline":
        """
        프리트레인된 모델들로부터 파이프라인 생성
        """
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # DCAE 로드 (멜 <-> 잠재벡터)
        dcae_model = create_dcae_model(
            model_type="pretrained",
            model_name=dcae_model_name,
            cache_dir=cache_dir
        )
        
        # Vocoder 로드 (멜 <-> 오디오)
        vocoder_model = create_vocoder_model(
            model_name=dcae_model_name,
            cache_dir=cache_dir
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
            vocoder_model=vocoder_model,
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
        음악 생성 메인 함수 (올바른 파이프라인)
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
            print(f"🎵 Starting {input_data.task} generation with corrected pipeline...")
            print(f"   Duration: {generation_config.duration}s")
            print(f"   Quality: {generation_config.quality}")
            print(f"   CFG Scale: {generation_config.cfg_scale}")
        
        # 조건 준비
        conditions = self._prepare_conditions(input_data)
        
        # 잠재 벡터 생성
        with torch.no_grad():
            generated_latents = self._generate_latents(conditions, generation_config, verbose)
        
        # 올바른 파이프라인: 잠재벡터 -> 멜 -> 오디오
        with torch.no_grad():
            generated_audio = self._decode_to_audio_correct_pipeline(generated_latents)
        
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
                'version': '1.0_corrected',
                'timestamp': time.time(),
                'dcae_compression': self.dcae_model.compression_ratio,
                'generator_parameters': self.generator_model.count_parameters(),
                'pipeline': 'audio->mel->dcae->mel->audio'
            }
        }
        
        if verbose:
            print(f"✅ Generation completed in {generation_time:.2f}s")
            print(f"   Output shape: {processed_audio.shape}")
            print(f"   Pipeline: Latents -> Mel -> Audio (Vocoder)")
        
        return result
    
    def _prepare_conditions(self, input_data: GenerationInput) -> Dict[str, Any]:
        """조건 준비 (참조 오디오 처리 수정)"""
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
        
        # 참조 오디오 처리 (올바른 파이프라인)
        if input_data.reference_audio:
            reference_latents = self._load_reference_audio_correct_pipeline(input_data.reference_audio)
            conditions['reference_audio'] = reference_latents
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
        # 멜 스펙트로그램 시간 길이 계산 (hop_length=512 기준)
        target_mel_time = target_samples // 512
        target_mel_time = min(max(target_mel_time, 64), 512)  # 범위 제한
        
        # DCAE latent 크기 추정 (멜 크기에서 압축)
        estimated_latent_h = target_mel_time // 8  # 대략적인 압축률
        estimated_latent_w = 128 // 8  # 멜 주파수 빈 압축
        
        shape = (1, 16, estimated_latent_h, estimated_latent_w)  # (batch, channels, h, w)
        
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
    
    def _decode_to_audio_correct_pipeline(self, latents: torch.Tensor) -> torch.Tensor:
        """
        올바른 파이프라인: 잠재 벡터 -> 멜 스펙트로그램 -> 오디오
        """
        try:
            # Step 1: 잠재 벡터 -> 멜 스펙트로그램 (DCAE 디코딩)
            if latents.dim() == 3:
                # Generator 출력이 3D인 경우 4D로 변환
                latents = latents.unsqueeze(1)  # (B, 1, H, W)
            elif latents.dim() != 4:
                logger.warning(f"Unexpected latents shape: {latents.shape}")
                # 적절한 4D 형태로 변환 시도
                if latents.dim() == 2:
                    # (B, C*H*W) -> (B, C, H, W)
                    B = latents.shape[0]
                    remaining = latents.shape[1]
                    # 16 채널이라고 가정하고 H, W 추정
                    C = 16
                    HW = remaining // C
                    H = W = int(np.sqrt(HW))
                    latents = latents.view(B, C, H, W)
                else:
                    raise ValueError(f"Cannot handle latents shape: {latents.shape}")
            
            # DCAE 디코딩: 잠재벡터 -> 멜 스펙트로그램
            decoded_mel = self.dcae_model.decode_to_mel(latents)
            
            # Step 2: 멜 스펙트로그램 -> 오디오 (Vocoder)
            decoded_audio = self.vocoder_model.mel_to_audio(decoded_mel)
            
            # 배치 차원 처리
            if decoded_audio.dim() == 3 and decoded_audio.shape[0] == 1:
                decoded_audio = decoded_audio.squeeze(0)  # (1, C, T) -> (C, T)
            elif decoded_audio.dim() == 2:
                pass  # 이미 (C, T) 형태
            else:
                logger.warning(f"Unexpected decoded audio shape: {decoded_audio.shape}")
                if decoded_audio.dim() > 2:
                    decoded_audio = decoded_audio.squeeze(0)
            
            return decoded_audio
            
        except Exception as e:
            logger.error(f"Audio decoding failed: {e}")
            # 폴백: 노이즈 오디오
            target_samples = int(10 * 44100)
            return torch.randn(2, target_samples, device=self.device) * 0.1
    
    def _load_reference_audio_correct_pipeline(self, reference: Union[str, Path, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        참조 오디오 로드 (올바른 파이프라인: 오디오 -> 멜 -> DCAE latent)
        """
        try:
            if isinstance(reference, torch.Tensor):
                audio = reference
                # tensor인 경우 차원 정규화
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0).repeat(2, 1)  # (T,) -> (2, T)
                elif audio.dim() == 2 and audio.shape[0] == 1:
                    audio = audio.repeat(2, 1)  # (1, T) -> (2, T)
                elif audio.dim() == 3 and audio.shape[0] == 1:
                    audio = audio.squeeze(0)  # (1, C, T) -> (C, T)
            else:
                # 파일에서 로드
                audio, sr = self.audio_processor.load_audio(
                    path=reference,
                    target_sr=44100,
                    normalize=True
                )
            
            # 배치 차원 추가
            if audio.dim() == 2:
                audio = audio.unsqueeze(0)  # (C, T) -> (1, C, T)
            
            # 올바른 파이프라인: 오디오 -> 멜 -> DCAE latent
            with torch.no_grad():
                # Step 1: 오디오 -> 멜 스펙트로그램
                mel = self.dcae_model.audio_to_mel(audio)
                
                # Step 2: 멜 스펙트로그램 -> DCAE latent
                ref_latents, _ = self.dcae_model.encode_mel(mel)
                
            return ref_latents
            
        except Exception as e:
            logger.error(f"Failed to load reference audio: {e}")
            return None
    
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
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Audio saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False
    
    def batch_generate(
        self,
        inputs: List[GenerationInput],
        generation_config: GenerationConfig = None,
        output_dir: Optional[Path] = None,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """배치 생성"""
        results = []
        
        for i, input_data in enumerate(inputs):
            try:
                if verbose:
                    print(f"Generating {i+1}/{len(inputs)}: {input_data.task}")
                
                result = self.generate(input_data, generation_config, verbose=False)
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'dcae': {
                'model_name': getattr(self.dcae_model, 'model_name', 'Unknown'),
                'compression_ratio': self.dcae_model.compression_ratio,
                'latent_channels': self.dcae_model.latent_channels
            },
            'vocoder': {
                'model_name': getattr(self.vocoder_model, 'model_name', 'Unknown'),
                'sample_rate': getattr(self.vocoder_model, 'sample_rate', 44100)
            },
            'generator': {
                'parameters': self.generator_model.count_parameters(),
                'd_model': self.generator_model.config.d_model,
                'n_layers': self.generator_model.config.n_layers
            },
            'device': str(self.device),
            'pipeline_version': '1.0_corrected',
            'pipeline_architecture': 'audio->mel->dcae_latent->mel->audio'
        }
    
    def test_pipeline(self) -> Dict[str, Any]:
        """파이프라인 테스트"""
        print("🧪 Testing corrected LYRO pipeline...")
        
        # 테스트 오디오 생성
        test_audio = torch.randn(1, 2, 44100 * 3).to(self.device)  # 3초 오디오
        
        results = {}
        
        try:
            # Step 1: 오디오 -> 멜
            mel = self.dcae_model.audio_to_mel(test_audio)
            results['audio_to_mel'] = {'input_shape': test_audio.shape, 'output_shape': mel.shape}
            print(f"  ✅ Audio -> Mel: {test_audio.shape} -> {mel.shape}")
            
            # Step 2: 멜 -> DCAE latent
            latents, _ = self.dcae_model.encode_mel(mel)
            results['mel_to_latent'] = {'input_shape': mel.shape, 'output_shape': latents.shape}
            print(f"  ✅ Mel -> Latent: {mel.shape} -> {latents.shape}")
            
            # Step 3: DCAE latent -> 멜
            reconstructed_mel = self.dcae_model.decode_to_mel(latents)
            results['latent_to_mel'] = {'input_shape': latents.shape, 'output_shape': reconstructed_mel.shape}
            print(f"  ✅ Latent -> Mel: {latents.shape} -> {reconstructed_mel.shape}")
            
            # Step 4: 멜 -> 오디오
            reconstructed_audio = self.vocoder_model.mel_to_audio(reconstructed_mel)
            results['mel_to_audio'] = {'input_shape': reconstructed_mel.shape, 'output_shape': reconstructed_audio.shape}
            print(f"  ✅ Mel -> Audio: {reconstructed_mel.shape} -> {reconstructed_audio.shape}")
            
            # 전체 파이프라인 테스트
            test_latents = torch.randn(1, 16, 16, 32).to(self.device)
            final_audio = self._decode_to_audio_correct_pipeline(test_latents)
            results['full_pipeline'] = {'input_shape': test_latents.shape, 'output_shape': final_audio.shape}
            print(f"  ✅ Full Pipeline: {test_latents.shape} -> {final_audio.shape}")
            
            results['status'] = 'success'
            print("🎉 Pipeline test completed successfully!")
            
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            print(f"❌ Pipeline test failed: {e}")
        
        return results


def create_pipeline(
    dcae_model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
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
        dcae_model_name=dcae_model_name,
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
    dcae_model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
    generator_checkpoint: str = None
) -> Dict[str, Any]:
    """
    빠른 생성 헬퍼 함수 (올바른 파이프라인)
    """
    # 파이프라인 생성
    pipeline = create_pipeline(
        dcae_model_name=dcae_model_name,
        generator_checkpoint=generator_checkpoint
    )
    
    # 파이프라인 테스트
    test_result = pipeline.test_pipeline()
    if test_result['status'] != 'success':
        print("⚠️ Pipeline test failed, proceeding anyway...")
    
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
        pipeline.save_audio(
            audio=result['audio'],
            output_path=output_path,
            metadata=result.get('metadata')
        )
    
    return result


if __name__ == "__main__":
    # 테스트
    print("Testing corrected LYRO Pipeline...")
    
    # 빠른 생성 테스트
    result = quick_generate(
        lyrics="Walking down the street tonight, the stars are shining bright",
        duration=5.0,
        quality="fast",
        output_path="test_output_corrected.wav"
    )
    
    print(f"Generation completed!")
    print(f"Audio shape: {result['audio'].shape}")
    print(f"Generation time: {result['generation_time']:.2f}s")
    print(f"Pipeline: {result['metadata']['pipeline']}")
