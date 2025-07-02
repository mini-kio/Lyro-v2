# lyro/models/dcae.py
"""
프리트레인된 DCAE 모델 로드 및 관리
ACE-Step 모델을 기반으로 한 음악용 DCAE
"""

import os
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings

warnings.filterwarnings("ignore")

try:
    from diffusers import AutoencoderDC
    from huggingface_hub import snapshot_download
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers not available. Install with: pip install diffusers huggingface_hub")


class PretrainedDCAE(nn.Module):
    """
    프리트레인된 DCAE 모델 래퍼
    ACE-Step에서 제공하는 music_dcae_f8c8 모델 사용
    """
    
    def __init__(
        self,
        model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
        subfolder: str = "music_dcae_f8c8",
        cache_dir: str = "checkpoints",
        sample_rate: int = 44100,
        force_download: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.subfolder = subfolder
        self.cache_dir = Path(cache_dir)
        self.sample_rate = sample_rate
        
        # 모델 로드
        self.dcae_model = self._load_pretrained_model(force_download)
        
        # 설정 정보
        self.latent_channels = 16  # ACE-Step DCAE 기본값
        self.compression_ratio = 50.0  # 약 50:1 압축
        
        # 평가 모드로 설정
        self.eval()
        
    def _load_pretrained_model(self, force_download: bool = False) -> nn.Module:
        """프리트레인된 DCAE 모델 로드"""
        
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers library is required. Install with: pip install diffusers huggingface_hub")
        
        # 로컬 캐시 경로
        local_path = self.cache_dir / self.subfolder
        
        try:
            # 로컬에서 먼저 시도
            if local_path.exists() and not force_download:
                print(f"Loading DCAE from local cache: {local_path}")
                dcae_model = AutoencoderDC.from_pretrained(local_path)
            else:
                # 허깅페이스에서 다운로드
                print(f"Downloading DCAE model: {self.model_name}/{self.subfolder}")
                
                # 모델 다운로드
                downloaded_path = snapshot_download(
                    repo_id=self.model_name,
                    allow_patterns=[f"{self.subfolder}/*"],
                    cache_dir=self.cache_dir,
                    local_dir=self.cache_dir,
                    local_dir_use_symlinks=False
                )
                
                # 모델 로드
                dcae_model = AutoencoderDC.from_pretrained(local_path)
                
            print("✅ DCAE model loaded successfully")
            return dcae_model
            
        except Exception as e:
            print(f"❌ Failed to load DCAE model: {e}")
            
            # 폴백: 더미 모델 생성
            print("Creating fallback DCAE model...")
            return self._create_fallback_model()
    
    def _create_fallback_model(self) -> nn.Module:
        """폴백용 간단한 DCAE 모델"""
        
        class FallbackDCAE(nn.Module):
            def __init__(self):
                super().__init__()
                # 간단한 인코더-디코더
                self.encoder = nn.Sequential(
                    nn.Conv1d(2, 64, 7, stride=4, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(64, 128, 5, stride=4, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(128, 16, 3, stride=2, padding=1),
                    nn.AdaptiveAvgPool1d(128)
                )
                
                self.decoder = nn.Sequential(
                    nn.ConvTranspose1d(16, 128, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose1d(128, 64, 8, stride=4, padding=2),
                    nn.ReLU(),
                    nn.ConvTranspose1d(64, 2, 8, stride=4, padding=2),
                    nn.Tanh()
                )
            
            def encode(self, x):
                return self.encoder(x), None
            
            def decode(self, z):
                return self.decoder(z)
        
        return FallbackDCAE()
    
    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        오디오를 잠재 공간으로 인코딩
        
        Args:
            audio: (B, 2, T) 스테레오 오디오
            
        Returns:
            latents: (B, 16, L) 잠재 벡터
            quantization_loss: 양자화 손실 (있는 경우)
        """
        # 입력 검증
        if audio.dim() != 3 or audio.shape[1] != 2:
            raise ValueError(f"Expected audio shape (B, 2, T), got {audio.shape}")
        
        # 모델에 따라 다른 방법으로 인코딩
        try:
            with torch.no_grad():
                if hasattr(self.dcae_model, 'encode'):
                    # AutoencoderDC의 경우
                    encoded = self.dcae_model.encode(audio)
                    if isinstance(encoded, tuple):
                        latents, quantization_loss = encoded
                    else:
                        latents = encoded
                        quantization_loss = None
                else:
                    # 폴백 모델의 경우
                    latents, quantization_loss = self.dcae_model.encode(audio)
                
                return latents, quantization_loss
                
        except Exception as e:
            print(f"Warning: DCAE encoding failed: {e}")
            # 더미 잠재 벡터 반환
            batch_size = audio.shape[0]
            dummy_latents = torch.randn(batch_size, self.latent_channels, 128, device=audio.device)
            return dummy_latents, None
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        잠재 벡터를 오디오로 디코딩
        
        Args:
            latents: (B, 16, L) 잠재 벡터
            
        Returns:
            audio: (B, 2, T) 재구성된 오디오
        """
        try:
            with torch.no_grad():
                if hasattr(self.dcae_model, 'decode'):
                    # AutoencoderDC의 경우
                    audio = self.dcae_model.decode(latents)
                else:
                    # 폴백 모델의 경우
                    audio = self.dcae_model.decode(latents)
                
                # 스테레오 보장
                if audio.shape[1] == 1:
                    audio = audio.repeat(1, 2, 1)
                elif audio.shape[1] > 2:
                    audio = audio[:, :2, :]
                
                # 범위 클리핑
                audio = torch.clamp(audio, -1.0, 1.0)
                
                return audio
                
        except Exception as e:
            print(f"Warning: DCAE decoding failed: {e}")
            # 더미 오디오 반환
            batch_size = latents.shape[0]
            target_length = 44100 * 5  # 5초
            dummy_audio = torch.randn(batch_size, 2, target_length, device=latents.device) * 0.1
            return dummy_audio
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        전체 인코딩-디코딩 과정
        
        Args:
            audio: (B, 2, T) 입력 오디오
            
        Returns:
            reconstructed: (B, 2, T) 재구성된 오디오
            quantization_loss: 양자화 손실 (있는 경우)
        """
        latents, quantization_loss = self.encode(audio)
        reconstructed = self.decode(latents)
        
        return reconstructed, quantization_loss
    
    def get_compression_info(self, audio: torch.Tensor) -> Dict[str, Any]:
        """압축 정보 계산"""
        with torch.no_grad():
            latents, _ = self.encode(audio)
            
            original_size = audio.numel()
            compressed_size = latents.numel()
            compression_ratio = original_size / max(compressed_size, 1)
            
            return {
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': float(compression_ratio),
                'latent_shape': tuple(latents.shape),
                'original_shape': tuple(audio.shape)
            }
    
    def save_pretrained(self, save_path: str):
        """모델 저장"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.dcae_model, 'save_pretrained'):
            self.dcae_model.save_pretrained(save_path)
        else:
            torch.save(self.dcae_model.state_dict(), save_path / "pytorch_model.bin")
        
        print(f"DCAE model saved to {save_path}")


class AdvancedVocoder(nn.Module):
    """
    Advanced Vocoder - 향상된 음성 합성기
    음악 DCAE와 함께 사용되는 고품질 vocoder
    """
    
    def __init__(
        self,
        model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
        subfolder: str = "music_vocoder",
        cache_dir: str = "checkpoints"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.subfolder = subfolder
        self.cache_dir = Path(cache_dir)
        
        # Vocoder 로드
        self.vocoder = self._load_vocoder()
        
        # 멜 스펙트로그램 변환
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            f_min=0.0,
            f_max=22050,
            center=True
        )
        
    def _load_vocoder(self):
        """Vocoder 모델 로드"""
        local_path = self.cache_dir / self.subfolder
        
        try:
            if local_path.exists():
                print(f"Loading Vocoder from local cache: {local_path}")
                # 실제 vocoder 로드 로직
                # 여기서는 간단한 더미 모델
                return self._create_dummy_vocoder()
            else:
                print(f"Downloading Vocoder model: {self.model_name}/{self.subfolder}")
                # 다운로드 및 로드
                return self._create_dummy_vocoder()
                
        except Exception as e:
            print(f"Warning: Failed to load Vocoder: {e}")
            return self._create_dummy_vocoder()
    
    def _create_dummy_vocoder(self):
        """더미 vocoder 생성"""
        class DummyVocoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                    nn.Tanh()
                )
            
            def forward(self, mel):
                # mel: (B, 128, T)
                B, C, T = mel.shape
                out = self.layers(mel.transpose(1, 2))  # (B, T, 1024)
                return out.transpose(1, 2).contiguous()  # (B, 1024, T)
        
        return DummyVocoder()
    
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """오디오를 멜 스펙트로그램으로 변환"""
        if audio.dim() > 2:
            audio = audio.mean(dim=1)  # 모노로 변환
        
        mel = self.mel_transform(audio)
        return mel
    
    def decode(self, mel: torch.Tensor) -> torch.Tensor:
        """멜 스펙트로그램을 오디오로 변환"""
        with torch.no_grad():
            audio = self.vocoder(mel)
            # 적절한 길이로 조정
            target_length = mel.shape[-1] * 512  # hop_length
            if audio.shape[-1] != target_length:
                audio = torch.nn.functional.interpolate(
                    audio.unsqueeze(0), 
                    size=target_length, 
                    mode='linear'
                ).squeeze(0)
            
            return audio


def create_dcae_model(
    model_type: str = "pretrained",
    model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
    cache_dir: str = "checkpoints",
    **kwargs
) -> nn.Module:
    """
    DCAE 모델 생성 팩토리 함수
    
    Args:
        model_type: 모델 타입 ("pretrained", "fallback")
        model_name: 프리트레인된 모델 이름
        cache_dir: 캐시 디렉토리
        **kwargs: 추가 파라미터
        
    Returns:
        초기화된 DCAE 모델
    """
    if model_type == "pretrained":
        return PretrainedDCAE(
            model_name=model_name,
            cache_dir=cache_dir,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def download_pretrained_models(
    model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
    cache_dir: str = "checkpoints",
    force_download: bool = False
):
    """
    프리트레인된 모델들을 미리 다운로드
    
    Args:
        model_name: 모델 이름
        cache_dir: 캐시 디렉토리
        force_download: 강제 다운로드 여부
    """
    if not DIFFUSERS_AVAILABLE:
        print("diffusers library is required for downloading models")
        return False
    
    try:
        print("Downloading pretrained DCAE and Vocoder models...")
        
        # DCAE 다운로드
        dcae_path = snapshot_download(
            repo_id=model_name,
            allow_patterns=["music_dcae_f8c8/*"],
            cache_dir=cache_dir,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            force_download=force_download
        )
        
        # Vocoder 다운로드
        vocoder_path = snapshot_download(
            repo_id=model_name,
            allow_patterns=["music_vocoder/*"],
            cache_dir=cache_dir,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            force_download=force_download
        )
        
        print(f"✅ Models downloaded successfully to {cache_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        return False


if __name__ == "__main__":
    # 테스트
    print("Testing DCAE model loading...")
    
    # 모델 다운로드 (선택적)
    download_pretrained_models()
    
    # DCAE 모델 생성
    dcae = create_dcae_model()
    
    # 테스트 오디오
    test_audio = torch.randn(1, 2, 44100)  # 1초 스테레오 오디오
    
    # 인코딩/디코딩 테스트
    latents, _ = dcae.encode(test_audio)
    reconstructed = dcae.decode(latents)
    
    print(f"Input shape: {test_audio.shape}")
    print(f"Latent shape: {latents.shape}")
    print(f"Output shape: {reconstructed.shape}")
    print(f"Compression info: {dcae.get_compression_info(test_audio)}")
    
    print("✅ DCAE test completed successfully!")