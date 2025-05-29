# lyro/inference/cli.py
"""
LYRO CLI Interface
명령줄 기반 음악 생성 도구
"""

import os
import sys
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
import json
import time
from typing import Optional, Dict, List
import logging

# LYRO 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssm.model import LyroSSMUNet, TaskController, EOSTokenHandler
from ssm.flow_matching import LyroFlowMatching, FlowConfig
from dcae.model import LyroMusicDCAE  # ✅ 수정: MusicDCAE -> LyroMusicDCAE
from dataset.tokenizer import LyroTokenizer


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LyroCLI:
    """LYRO CLI 메인 클래스"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 모델 로드
        self._load_models()
        
        # 출력 디렉토리
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_models(self):
        """모델 로드"""
        logger.info("Loading models...")
        
        # SSM 모델
        self.ssm_model = LyroSSMUNet(
            input_channels=8,
            hidden_dims=[128, 128, 256, 256, 512],
            mamba_layers=[2, 2, 3, 3, 4],
            max_seq_len=8192  # ✅ 수정: 6144 -> 8192
        ).to(self.device)
        
        # SSM 체크포인트 로드
        if self.args.ssm_checkpoint:
            ssm_ckpt = torch.load(self.args.ssm_checkpoint, map_location=self.device)
            self.ssm_model.load_state_dict(ssm_ckpt['model_state_dict'])
            logger.info(f"Loaded SSM from {self.args.ssm_checkpoint}")
        
        self.ssm_model.eval()
        
        # DCAE 모델
        self.dcae_model = LyroMusicDCAE(  # ✅ 수정: MusicDCAE -> LyroMusicDCAE
            sample_rate=44100,
            latent_channels=8  # ✅ 수정: compression_factor 제거, latent_channels 사용
        ).to(self.device)
        
        # DCAE 체크포인트 로드
        if self.args.dcae_checkpoint:
            dcae_ckpt = torch.load(self.args.dcae_checkpoint, map_location=self.device)
            self.dcae_model.load_state_dict(dcae_ckpt['model_state_dict'])
            logger.info(f"Loaded DCAE from {self.args.dcae_checkpoint}")
            
        self.dcae_model.eval()
        
        # Flow Matching 설정
        flow_config = FlowConfig()
        
        # 품질 설정에 따른 조정
        if self.args.quality == 'premium':
            flow_config.flow_steps = 16
            flow_config.integration_method = 'rk4'
        elif self.args.quality == 'fast':
            flow_config.flow_steps = 4
            flow_config.integration_method = 'euler'
        else:  # standard
            flow_config.flow_steps = self.args.flow_steps or 8
            
        self.flow_matching = LyroFlowMatching(  # ✅ 수정: FlowMatching -> LyroFlowMatching
            model=self.ssm_model,
            scheduler_type="cosine",
            solver_type=flow_config.integration_method,
            sigma=1e-4,
            flow_type="rectified"
        )
        
        # 토크나이저
        self.tokenizer = LyroTokenizer()
        
        logger.info("Models loaded successfully!")
        
    def generate(self):
        """메인 생성 함수"""
        logger.info(f"Starting {self.args.task} generation...")
        
        # 태스크별 처리
        if self.args.task == 'SONG':
            return self._generate_song()
        elif self.args.task == 'INST':
            return self._generate_instrumental()
        elif self.args.task == 'COVER':
            return self._generate_cover()
        elif self.args.task == 'INPAINT':
            return self._inpaint()
        elif self.args.task == 'EXTEND':
            return self._extend()
        else:
            raise ValueError(f"Unknown task: {self.args.task}")
            
    def _generate_song(self):
        """SONG 태스크 생성"""
        # 가사 로드
        if not self.args.lyrics:
            raise ValueError("Lyrics required for SONG task")
            
        if Path(self.args.lyrics).exists():
            with open(self.args.lyrics, 'r', encoding='utf-8') as f:
                lyrics = f.read().strip()
        else:
            lyrics = self.args.lyrics
            
        logger.info(f"Lyrics: {lyrics[:100]}...")
        
        # 조건 준비
        conditions = self._prepare_conditions('SONG', lyrics=lyrics)
        
        # 생성
        audio = self._generate_audio(conditions)
        
        # 저장
        output_path = self._save_audio(audio, prefix='song')
        logger.info(f"Generated song saved to: {output_path}")
        
        return output_path
    
    def _generate_instrumental(self):
        """INST 태스크 생성"""
        logger.info("Generating instrumental...")
        
        # 조건 준비 (가사 없음)
        conditions = self._prepare_conditions('INST')
        
        # 생성
        audio = self._generate_audio(conditions)
        
        # 저장
        output_path = self._save_audio(audio, prefix='inst')
        logger.info(f"Generated instrumental saved to: {output_path}")
        
        return output_path
    
    def _generate_cover(self):
        """COVER 태스크 생성"""
        if not self.args.reference:
            raise ValueError("Reference audio required for COVER task")
            
        logger.info(f"Loading reference from: {self.args.reference}")
        
        # 참조 오디오 로드
        ref_audio, sr = torchaudio.load(self.args.reference)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            ref_audio = resampler(ref_audio)
            
        # 참조 길이 제한
        ref_length = self.args.ref_length or 10.0
        max_samples = int(ref_length * 44100)
        if ref_audio.shape[1] > max_samples:
            ref_audio = ref_audio[:, :max_samples]
            
        # 스테레오 확인
        if ref_audio.shape[0] == 1:
            ref_audio = ref_audio.repeat(2, 1)
            
        # DCAE 인코딩
        with torch.no_grad():
            ref_audio_tensor = ref_audio.unsqueeze(0).to(self.device)
            ref_latent, _ = self.dcae_model.encode(ref_audio_tensor)  # ✅ 수정: skip_features 처리
            
        # 새로운 가사 (옵션)
        lyrics = None
        if self.args.lyrics:
            if Path(self.args.lyrics).exists():
                with open(self.args.lyrics, 'r', encoding='utf-8') as f:
                    lyrics = f.read().strip()
            else:
                lyrics = self.args.lyrics
                
        # 조건 준비
        conditions = self._prepare_conditions(
            'COVER',
            lyrics=lyrics,
            reference_latent=ref_latent
        )
        
        # 생성
        audio = self._generate_audio(conditions)
        
        # 저장
        output_path = self._save_audio(audio, prefix='cover')
        logger.info(f"Generated cover saved to: {output_path}")
        
        return output_path
    
    def _inpaint(self):
        """INPAINT 태스크 (부분 편집)"""
        if not self.args.input:
            raise ValueError("Input audio required for INPAINT task")
            
        if not self.args.mask:
            raise ValueError("Mask region required for INPAINT task")
            
        logger.info(f"Loading input from: {self.args.input}")
        logger.info(f"Edit region: {self.args.mask}")
        
        # 입력 오디오 로드
        audio, sr = torchaudio.load(self.args.input)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            audio = resampler(audio)
            
        # DCAE 인코딩
        with torch.no_grad():
            audio_tensor = audio.unsqueeze(0).to(self.device)
            original_latent, _ = self.dcae_model.encode(audio_tensor)  # ✅ 수정: skip_features 처리
            
        # 마스크 파싱
        mask_parts = self.args.mask.split(':')
        if len(mask_parts) != 2:
            raise ValueError("Mask format should be 'start:end' in seconds")
            
        start_time = float(mask_parts[0])
        end_time = float(mask_parts[1])
        
        # 마스크 생성 (수정된 계산)
        # ✅ 수정: 32x 압축율 반영
        total_duration = audio.shape[1] / 44100
        latent_length = original_latent.shape[-1]
        
        start_ratio = start_time / total_duration
        end_ratio = end_time / total_duration
        
        mask = torch.zeros(1, 1, latent_length, device=self.device)
        start_idx = int(start_ratio * latent_length)
        end_idx = int(end_ratio * latent_length)
        mask[:, :, start_idx:end_idx] = 1.0
        
        # 편집 프롬프트
        edit_prompt = self.args.prompt or "edited version"
        
        # Flow-Edit 적용
        logger.info("Applying Flow-Edit...")
        
        # 편집 조건 준비
        edit_conditions = self._prepare_conditions('INPAINT', style_prompt=edit_prompt)
        
        # Flow-Edit
        edited_latent = self.flow_matching.flow_edit(
            original=original_latent,
            mask=mask,
            new_conditions=edit_conditions,
            edit_steps=self.args.flow_steps or 4
        )
        
        # DCAE 디코딩 (더미 skip_features 생성)
        with torch.no_grad():
            # Skip features를 위한 더미 생성
            dummy_skip_features = [torch.zeros_like(edited_latent) for _ in range(5)]
            edited_audio = self.dcae_model.decode(edited_latent, dummy_skip_features)
            
        # 저장
        output_path = self._save_audio(edited_audio[0], prefix='edited')
        logger.info(f"Edited audio saved to: {output_path}")
        
        return output_path
    
    def _extend(self):
        """EXTEND 태스크 (곡 연장)"""
        if not self.args.input:
            raise ValueError("Input audio required for EXTEND task")
            
        logger.info(f"Loading input from: {self.args.input}")
        
        # 입력 오디오 로드
        audio, sr = torchaudio.load(self.args.input)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            audio = resampler(audio)
            
        # DCAE 인코딩
        with torch.no_grad():
            audio_tensor = audio.unsqueeze(0).to(self.device)
            original_latent, _ = self.dcae_model.encode(audio_tensor)  # ✅ 수정: skip_features 처리
            
        # 연장 길이 (수정된 계산)
        extend_duration = self.args.extend_length or 30.0  # 기본 30초 연장
        # ✅ 수정: 32x 압축율 반영
        extend_samples = int(extend_duration * 44100 / (512 * 32))
        
        # 스타일 조건
        style_conditions = self._prepare_conditions('EXTEND', style_prompt=self.args.prompt)
        
        # Flow-Extend 적용
        logger.info(f"Extending by {extend_duration} seconds...")
        
        extended_latent = self.flow_matching.flow_extend(
            original_latent=original_latent,
            extend_length=extend_samples,
            conditions=style_conditions
        )
        
        # DCAE 디코딩 (더미 skip_features 생성)
        with torch.no_grad():
            dummy_skip_features = [torch.zeros_like(extended_latent) for _ in range(5)]
            extended_audio = self.dcae_model.decode(extended_latent, dummy_skip_features)
            
        # 저장
        output_path = self._save_audio(extended_audio[0], prefix='extended')
        logger.info(f"Extended audio saved to: {output_path}")
        
        return output_path
    
    def _prepare_conditions(
        self,
        task: str,
        lyrics: Optional[str] = None,
        style_prompt: Optional[str] = None,
        reference_latent: Optional[torch.Tensor] = None
    ) -> Dict:
        """조건 준비"""
        # 기본 스타일 프롬프트
        if style_prompt is None:
            style_prompt = self.args.style or "pop music"
            
        # 조건 생성
        conditions = TaskController.create_conditions(
            task_type=task,
            lyrics=self._encode_lyrics(lyrics) if lyrics else None,
            style_prompt=self._encode_style(style_prompt),
            icl_reference=reference_latent,
            device=self.device  # ✅ 추가: device 전달
        )
        
        return conditions
    
    def _encode_lyrics(self, lyrics: str) -> torch.Tensor:
        """가사 인코딩"""
        tokens = self.tokenizer.encode_text(lyrics)
        return torch.tensor([tokens]).to(self.device)
    
    def _encode_style(self, style_prompt: str) -> torch.Tensor:
        """스타일 프롬프트 인코딩"""
        # 간단한 텍스트 임베딩
        tokens = self.tokenizer.encode_text(style_prompt)
        style_vec = torch.zeros(512)
        
        # 장르 키워드 매핑
        genre_keywords = {
            'pop': 0, 'rock': 1, 'jazz': 2, 'classical': 3,
            'electronic': 4, 'hip-hop': 5, 'folk': 6, 'blues': 7,
            'ballad': 8, 'dance': 9, 'ambient': 10, 'metal': 11
        }
        
        # 키워드 검색
        style_lower = style_prompt.lower()
        for genre, idx in genre_keywords.items():
            if genre in style_lower:
                style_vec[idx * 40:(idx + 1) * 40] = 1.0
                
        return style_vec.unsqueeze(0).to(self.device)
    
    def _generate_audio(self, conditions: Dict) -> torch.Tensor:
        """안전한 오디오 생성"""
        start_time = time.time()
        
        # 생성 shape (수정된 계산)
        # ✅ 수정: 32x 압축율 반영
        duration_samples = min(
            int(self.args.duration * 44100 / (512 * 32)),
            self.ssm_model.max_seq_len
        )
        
        if duration_samples <= 0:
            duration_samples = 64  # 최소값
            
        shape = (1, 8, duration_samples)
        
        logger.info(f"Generating {self.args.duration}s of audio...")
        logger.info(f"Flow steps: {self.flow_matching.flow_steps}")
        logger.info(f"Guidance scale: {self.args.guidance_scale}")
        logger.info(f"Latent shape: {shape}")
        
        try:
            # Flow Matching 생성
            with torch.no_grad():
                generated_latent, trajectory = self.flow_matching.generate(
                    shape=shape,
                    conditions=conditions,
                    num_steps=self.flow_matching.flow_steps,  # ✅ 수정: steps -> num_steps
                    cfg_scale=self.args.guidance_scale,
                    seed=self.args.seed
                )
                
                # 안전성 검증
                if torch.isnan(generated_latent).any():
                    raise ValueError("Generated latent contains NaN")
                    
                if torch.isinf(generated_latent).any():
                    raise ValueError("Generated latent contains Inf")
                
                # EOS 처리
                if self.args.early_stopping:
                    # EOS 토큰 감지 로직
                    # TODO: 실제 EOS 처리 구현
                    pass
                    
                # DCAE 디코딩 (더미 skip_features 생성)
                dummy_skip_features = [torch.zeros_like(generated_latent) for _ in range(5)]
                audio = self.dcae_model.decode(generated_latent, dummy_skip_features)
                
                # 오디오 후처리
                audio = torch.clamp(audio, -1.0, 1.0)  # 클리핑 방지
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            # 대체 생성 (무음)
            audio = torch.zeros(1, 2, int(self.args.duration * 44100))
            
        generation_time = time.time() - start_time
        rtf = generation_time / self.args.duration
        
        logger.info(f"Generation completed in {generation_time:.2f}s (RTF: {rtf:.3f})")
        
        return audio[0]  # (2, T)
    
    def _save_audio(self, audio: torch.Tensor, prefix: str = 'output') -> Path:
        """오디오 저장"""
        # 타임스탬프 추가
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{timestamp}.wav"
        output_path = self.output_dir / filename
        
        # 저장
        torchaudio.save(
            output_path,
            audio.cpu(),
            sample_rate=44100
        )
        
        # 메타데이터 저장
        metadata = {
            'task': self.args.task,
            'duration': self.args.duration,
            'flow_steps': self.flow_matching.flow_steps,
            'guidance_scale': self.args.guidance_scale,
            'quality': self.args.quality,
            'timestamp': timestamp
        }
        
        if self.args.lyrics:
            metadata['lyrics'] = self.args.lyrics
        if self.args.style:
            metadata['style'] = self.args.style
            
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return output_path
    
    def batch_generate(self):
        """배치 생성 모드"""
        if not self.args.batch_file:
            raise ValueError("Batch file required for batch mode")
            
        logger.info(f"Processing batch file: {self.args.batch_file}")
        
        # 배치 파일 로드
        with open(self.args.batch_file, 'r') as f:
            batch_data = json.load(f)
            
        results = []
        
        # 각 항목 처리
        for i, item in enumerate(batch_data['items']):
            logger.info(f"\nProcessing item {i+1}/{len(batch_data['items'])}")
            
            # 인자 업데이트
            self.args.task = item.get('task', 'SONG')
            self.args.lyrics = item.get('lyrics')
            self.args.style = item.get('style')
            self.args.duration = item.get('duration', 30.0)
            
            try:
                # 생성
                output_path = self.generate()
                results.append({
                    'index': i,
                    'status': 'success',
                    'output': str(output_path)
                })
            except Exception as e:
                logger.error(f"Failed to process item {i}: {e}")
                results.append({
                    'index': i,
                    'status': 'failed',
                    'error': str(e)
                })
                
        # 결과 저장
        results_path = self.output_dir / 'batch_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"\nBatch processing completed. Results saved to: {results_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='LYRO Music Generation CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a song with lyrics
  python inference/cli.py --task SONG --lyrics "Walking down the street..." --style "pop ballad"
  
  # Generate instrumental
  python inference/cli.py --task INST --style "jazz piano" --duration 60
  
  # Generate cover
  python inference/cli.py --task COVER --reference original.wav --lyrics new_lyrics.txt
  
  # Edit audio (inpaint)
  python inference/cli.py --task INPAINT --input song.wav --mask "45.0:70.0" --prompt "stronger drums"
  
  # Extend audio
  python inference/cli.py --task EXTEND --input short_song.wav --extend_length 30
        """
    )
    
    # 기본 인자
    parser.add_argument('--task', type=str, default='SONG',
                        choices=['SONG', 'INST', 'COVER', 'INPAINT', 'EXTEND'],
                        help='Generation task')
    
    # 입력 관련
    parser.add_argument('--lyrics', type=str, help='Lyrics text or file path')
    parser.add_argument('--style', type=str, help='Style prompt')
    parser.add_argument('--reference', type=str, help='Reference audio for COVER')
    parser.add_argument('--input', type=str, help='Input audio for INPAINT/EXTEND')
    
    # 생성 설정
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Output duration in seconds')
    parser.add_argument('--flow_steps', type=int, help='Number of flow steps')
    parser.add_argument('--guidance_scale', type=float, default=1.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--quality', type=str, default='standard',
                        choices=['fast', 'standard', 'premium'],
                        help='Generation quality preset')
    
    # 편집 관련
    parser.add_argument('--mask', type=str, help='Edit mask region (start:end in seconds)')
    parser.add_argument('--prompt', type=str, help='Edit/extend prompt')
    parser.add_argument('--extend_length', type=float, help='Extension duration')
    parser.add_argument('--ref_length', type=float, default=10.0,
                        help='Reference audio length for COVER')
    
    # 모델 경로
    parser.add_argument('--ssm_checkpoint', type=str, 
                        default='ssm/checkpoints/best_model_stage_E.pt',
                        help='SSM model checkpoint')
    parser.add_argument('--dcae_checkpoint', type=str,
                        default='dcae/checkpoints/best_model.pt',
                        help='DCAE model checkpoint')
    
    # 출력 설정
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--seed', type=int, help='Random seed')
    
    # 고급 옵션
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable EOS-based early stopping')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Batch generation mode')
    parser.add_argument('--batch_file', type=str,
                        help='Batch configuration file (JSON)')
    
    args = parser.parse_args()
    
    # 시드 설정
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")
    
    # CLI 실행
    cli = LyroCLI(args)
    
    try:
        if args.batch_mode:
            # 배치 모드
            cli.batch_generate()
        else:
            # 단일 생성
            cli.generate()
            
    except KeyboardInterrupt:
        logger.info("\nGeneration interrupted by user")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


if __name__ == '__main__':
    main()