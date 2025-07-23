# models/audio_encoder.py
"""
DCAE 기반 오디오 인코더 - 오디오를 latent로 변환
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Union

# DCAE vocoder import
dcae_path = Path(__file__).parent.parent / "dcae_vocoder"
sys.path.insert(0, str(dcae_path))

try:
    from music_dcae_pipeline import MusicDCAE
    DCAE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DCAE vocoder not available: {e}")
    DCAE_AVAILABLE = False


class LyroAudioEncoder(nn.Module):
    """
    DCAE 기반 오디오 인코더
    오디오 → latent 변환 담당
    """
    
    def __init__(
        self,
        dcae_checkpoint_path: Optional[str] = None,
        vocoder_checkpoint_path: Optional[str] = None,
        source_sample_rate: int = 44100,
        target_sample_rate: int = 44100,
        device: str = "auto"
    ):
        super().__init__()
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # DCAE 경로 설정
        if dcae_checkpoint_path is None:
            dcae_checkpoint_path = str(dcae_path / "dcae")
        if vocoder_checkpoint_path is None:
            vocoder_checkpoint_path = str(dcae_path / "vocoder")
        
        self.dcae_available = DCAE_AVAILABLE
        
        if self.dcae_available:
            try:
                # DCAE 파이프라인 초기화
                self.dcae = MusicDCAE(
                    source_sample_rate=source_sample_rate,
                    dcae_checkpoint_path=dcae_checkpoint_path,
                    vocoder_checkpoint_path=vocoder_checkpoint_path
                )
                self.dcae.to(device)
                print(f"DCAE Audio Encoder loaded from {dcae_checkpoint_path}")
                
            except Exception as e:
                print(f"Failed to load DCAE: {e}")
                self.dcae_available = False
                self._init_fallback_encoder()
        else:
            self._init_fallback_encoder()
    
    def _init_fallback_encoder(self):
        """DCAE 실패 시 fallback 인코더"""
        print("WARNING: Using fallback audio encoder (random latents)")
        self.latent_channels = 16
        self.latent_time_steps = 128
        
    def encode_audio(
        self, 
        audio: torch.Tensor, 
        sample_rate: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        오디오를 latent로 인코딩
        
        Args:
            audio: (B, C, T) 또는 (C, T) 형태의 오디오
            sample_rate: 오디오 샘플링 레이트
            
        Returns:
            latents: (B, 16, 128) 형태의 latent
            latent_lengths: latent 실제 길이
        """
        if not self.dcae_available:
            return self._fallback_encode(audio)
        
        try:
            # 차원 확인 및 배치 처리
            if audio.dim() == 2:
                audio = audio.unsqueeze(0)  # (C, T) → (B, C, T)
            
            batch_size = audio.shape[0]
            
            # 샘플링 레이트 기본값
            if sample_rate is None:
                sample_rate = 44100
            
            # DCAE 인코딩
            with torch.no_grad():
                latents, latent_lengths = self.dcae.encode(
                    audios=audio.to(self.device),
                    sr=sample_rate
                )
            
            return latents, latent_lengths
            
        except Exception as e:
            print(f"DCAE encoding failed: {e}, using fallback")
            return self._fallback_encode(audio)
    
    def _fallback_encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback 인코딩 (랜덤 latent)"""
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        batch_size = audio.shape[0]
        
        # 랜덤 latent 생성 (일관성을 위해 시드 사용)
        torch.manual_seed(42)
        latents = torch.randn(batch_size, 16, 128, device=audio.device)
        latent_lengths = torch.full((batch_size,), 128, device=audio.device)
        
        return latents, latent_lengths
    
    def decode_latents(
        self,
        latents: torch.Tensor,
        sample_rate: Optional[int] = None,
        use_chunking: bool = True
    ) -> Tuple[int, list]:
        """
        Latent를 오디오로 디코딩
        
        Args:
            latents: (B, 16, 128) 형태의 latent
            sample_rate: 출력 샘플링 레이트
            use_chunking: 청킹 사용 여부
            
        Returns:
            sample_rate: 출력 샘플링 레이트
            audio_list: 디코딩된 오디오 리스트
        """
        if not self.dcae_available:
            print("WARNING: DCAE not available, cannot decode latents")
            # 더미 오디오 반환
            dummy_audio = torch.zeros(latents.shape[0], 2, 44100)
            return 44100, [dummy_audio[i] for i in range(latents.shape[0])]
        
        try:
            with torch.no_grad():
                sample_rate_out, audio_list = self.dcae.decode(
                    latents=latents.to(self.device),
                    sr=sample_rate,
                    use_chunking=use_chunking
                )
            
            return sample_rate_out, audio_list
            
        except Exception as e:
            print(f"DCAE decoding failed: {e}")
            dummy_audio = torch.zeros(latents.shape[0], 2, 44100)
            return 44100, [dummy_audio[i] for i in range(latents.shape[0])]
    
    def forward(self, audio: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass (인코딩만)
        
        Args:
            audio: (B, C, T) 오디오
            sample_rate: 샘플링 레이트
            
        Returns:
            latents: (B, 16, 128) latent
        """
        latents, _ = self.encode_audio(audio, sample_rate)
        return latents
    
    @property
    def latent_shape(self) -> Tuple[int, int]:
        """Latent 형태 반환"""
        return (16, 128)  # (channels, time_steps)
    
    def get_info(self) -> dict:
        """인코더 정보 반환"""
        return {
            'type': 'DCAE Audio Encoder',
            'available': self.dcae_available,
            'latent_shape': self.latent_shape,
            'device': str(self.device),
            'sample_rates': {
                'input': '44.1kHz (default)',
                'processing': '44.1kHz',
                'output': 'configurable'
            }
        }


def create_audio_encoder(
    dcae_checkpoint_path: Optional[str] = None,
    device: str = "auto"
) -> LyroAudioEncoder:
    """편의 함수: 오디오 인코더 생성"""
    return LyroAudioEncoder(
        dcae_checkpoint_path=dcae_checkpoint_path,
        device=device
    )


# 테스트를 위한 함수
def test_audio_encoder():
    """오디오 인코더 테스트"""
    print("Testing Audio Encoder...")
    
    encoder = create_audio_encoder()
    print(f"Encoder info: {encoder.get_info()}")
    
    # 더미 오디오 테스트
    dummy_audio = torch.randn(2, 2, 44100)  # 2 batch, 2 channels, 1초
    print(f"Input audio shape: {dummy_audio.shape}")
    
    # 인코딩 테스트
    latents, lengths = encoder.encode_audio(dummy_audio)
    print(f"Encoded latents shape: {latents.shape}")
    print(f"Latent lengths: {lengths}")
    
    # 디코딩 테스트 (DCAE 사용 가능한 경우)
    if encoder.dcae_available:
        sr, audio_list = encoder.decode_latents(latents)
        print(f"Decoded audio: {len(audio_list)} files, sample_rate={sr}")
        if audio_list:
            print(f"First audio shape: {audio_list[0].shape}")
    
    print("Audio Encoder test completed!")


if __name__ == "__main__":
    test_audio_encoder()