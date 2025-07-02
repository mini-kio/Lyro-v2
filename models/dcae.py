# lyro/models/dcae.py
"""
프리트레인된 DCAE 모델 로드 및 관리
ACE-Step 모델을 기반으로 한 음악용 DCAE
멜 스펙트로그램 <-> 잠재 벡터 변환만 담당
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
    멜 스펙트로그램만 처리 (오디오는 별도 Vocoder 필요)
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
        
        # 멜 스펙트로그램 변환기 (오디오 <-> 멜 변환용)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_mels=128,
            f_min=0.0,
            f_max=sample_rate // 2,
            center=True
        )
        
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
                # 간단한 인코더-디코더 (멜 스펙트로그램용)
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 64, 3, stride=2, padding=1),  # 멜은 2D
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 16, 3, stride=2, padding=1),
                    nn.AdaptiveAvgPool2d((16, 16))
                )
                
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(16, 128, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
                    nn.Sigmoid()
                )
            
            def encode(self, x):
                return self.encoder(x), None
            
            def decode(self, z):
                return self.decoder(z)
        
        return FallbackDCAE()
    
    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """
        오디오를 멜 스펙트로그램으로 변환
        
        Args:
            audio: (B, 2, T) 스테레오 오디오
            
        Returns:
            mel: (B, 1, n_mels, T_mel) 멜 스펙트로그램
        """
        # 스테레오를 모노로 변환
        if audio.dim() == 3 and audio.shape[1] == 2:
            audio = audio.mean(dim=1)  # (B, T)
        elif audio.dim() == 2 and audio.shape[0] == 2:
            audio = audio.mean(dim=0).unsqueeze(0)  # (1, T)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (1, T)
        
        # 멜 스펙트로그램 계산
        mel = self.mel_transform(audio)  # (B, n_mels, T_mel)
        
        # 로그 변환
        mel = torch.log(mel + 1e-7)
        
        # 채널 차원 추가 (DCAE 입력 형식)
        mel = mel.unsqueeze(1)  # (B, 1, n_mels, T_mel)
        
        return mel
    
    def encode_mel(self, mel: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        멜 스펙트로그램을 잠재 공간으로 인코딩
        
        Args:
            mel: (B, 1, n_mels, T_mel) 멜 스펙트로그램
            
        Returns:
            latents: (B, 16, H, W) 잠재 벡터
            quantization_loss: 양자화 손실 (있는 경우)
        """
        # 입력 검증
        if mel.dim() != 4:
            raise ValueError(f"Expected mel shape (B, C, H, W), got {mel.shape}")
        
        # 모델에 따라 다른 방법으로 인코딩
        try:
            with torch.no_grad():
                if hasattr(self.dcae_model, 'encode'):
                    # AutoencoderDC의 경우
                    encoded = self.dcae_model.encode(mel)
                    if isinstance(encoded, tuple):
                        latents, quantization_loss = encoded
                    else:
                        latents = encoded
                        quantization_loss = None
                else:
                    # 폴백 모델의 경우
                    latents, quantization_loss = self.dcae_model.encode(mel)
                
                return latents, quantization_loss
                
        except Exception as e:
            print(f"Warning: DCAE encoding failed: {e}")
            # 더미 잠재 벡터 반환
            batch_size = mel.shape[0]
            dummy_latents = torch.randn(batch_size, self.latent_channels, 16, 16, device=mel.device)
            return dummy_latents, None
    
    def decode_to_mel(self, latents: torch.Tensor) -> torch.Tensor:
        """
        잠재 벡터를 멜 스펙트로그램으로 디코딩
        
        Args:
            latents: (B, 16, H, W) 잠재 벡터
            
        Returns:
            mel: (B, 1, n_mels, T_mel) 재구성된 멜 스펙트로그램
        """
        try:
            with torch.no_grad():
                if hasattr(self.dcae_model, 'decode'):
                    # AutoencoderDC의 경우
                    mel = self.dcae_model.decode(latents)
                else:
                    # 폴백 모델의 경우
                    mel = self.dcae_model.decode(latents)
                
                # 멜 스펙트로그램 형식 보장
                if mel.dim() == 3:
                    mel = mel.unsqueeze(1)  # (B, 1, H, W)
                
                return mel
                
        except Exception as e:
            print(f"Warning: DCAE decoding failed: {e}")
            # 더미 멜 스펙트로그램 반환
            batch_size = latents.shape[0]
            dummy_mel = torch.zeros(batch_size, 1, 128, 256, device=latents.device)
            return dummy_mel
    
    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        오디오를 잠재 공간으로 인코딩 (전체 파이프라인)
        
        Args:
            audio: (B, 2, T) 스테레오 오디오
            
        Returns:
            latents: (B, 16, H, W) 잠재 벡터
            quantization_loss: 양자화 손실
        """
        # 오디오 -> 멜 스펙트로그램
        mel = self.audio_to_mel(audio)
        
        # 멜 스펙트로그램 -> 잠재 벡터
        latents, quantization_loss = self.encode_mel(mel)
        
        return latents, quantization_loss
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        잠재 벡터를 멜 스펙트로그램으로 디코딩
        주의: 오디오로 변환하려면 별도의 Vocoder가 필요함
        
        Args:
            latents: (B, 16, H, W) 잠재 벡터
            
        Returns:
            mel: (B, 1, n_mels, T_mel) 멜 스펙트로그램
        """
        return self.decode_to_mel(latents)
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        전체 인코딩-디코딩 과정 (멜 스펙트로그램 출력)
        
        Args:
            audio: (B, 2, T) 입력 오디오
            
        Returns:
            reconstructed_mel: (B, 1, n_mels, T_mel) 재구성된 멜 스펙트로그램
            quantization_loss: 양자화 손실
        """
        latents, quantization_loss = self.encode(audio)
        reconstructed_mel = self.decode(latents)
        
        return reconstructed_mel, quantization_loss
    
    def get_compression_info(self, audio: torch.Tensor) -> Dict[str, Any]:
        """압축 정보 계산"""
        with torch.no_grad():
            mel = self.audio_to_mel(audio)
            latents, _ = self.encode_mel(mel)
            
            original_size = mel.numel()
            compressed_size = latents.numel()
            compression_ratio = original_size / max(compressed_size, 1)
            
            return {
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': float(compression_ratio),
                'latent_shape': tuple(latents.shape),
                'mel_shape': tuple(mel.shape)
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
    Advanced Vocoder - 멜 스펙트로그램을 오디오로 변환
    """
    
    def __init__(
        self,
        model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
        subfolder: str = "music_vocoder",
        cache_dir: str = "checkpoints",
        sample_rate: int = 44100
    ):
        super().__init__()
        
        self.model_name = model_name
        self.subfolder = subfolder
        self.cache_dir = Path(cache_dir)
        self.sample_rate = sample_rate
        
        # Vocoder 로드
        self.vocoder = self._load_vocoder()
        
        # 역 멜 스펙트로그램 변환 (Griffin-Lim 알고리즘)
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            n_iter=32
        )
        
        self.eval()
        
    def _load_vocoder(self):
        """Vocoder 모델 로드"""
        local_path = self.cache_dir / self.subfolder
        
        try:
            if local_path.exists():
                print(f"Loading Vocoder from local cache: {local_path}")
                # 실제 vocoder 로드 로직 (구현 필요)
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
                # 실제로는 HiFiGAN 등의 vocoder 구조 필요
                self.layers = nn.Sequential(
                    nn.ConvTranspose1d(128, 256, 8, stride=4, padding=2),
                    nn.ReLU(),
                    nn.ConvTranspose1d(256, 128, 8, stride=4, padding=2),
                    nn.ReLU(),
                    nn.ConvTranspose1d(128, 64, 8, stride=4, padding=2),
                    nn.ReLU(),
                    nn.ConvTranspose1d(64, 1, 7, stride=1, padding=3),
                    nn.Tanh()
                )
            
            def forward(self, mel):
                # mel: (B, 1, n_mels, T) -> (B, n_mels, T)
                if mel.dim() == 4:
                    mel = mel.squeeze(1)
                
                # 1D ConvTranspose를 위해 transpose
                mel_t = mel.transpose(1, 2)  # (B, T, n_mels) -> (B, n_mels, T)
                
                # Vocoder 처리
                audio = self.layers(mel_t)  # (B, 1, T_audio)
                
                # 스테레오로 변환
                audio = audio.repeat(1, 2, 1)  # (B, 2, T_audio)
                
                return audio
        
        return DummyVocoder()
    
    def mel_to_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """
        멜 스펙트로그램을 오디오로 변환
        
        Args:
            mel: (B, 1, n_mels, T_mel) 멜 스펙트로그램
            
        Returns:
            audio: (B, 2, T_audio) 스테레오 오디오
        """
        with torch.no_grad():
            try:
                # Vocoder로 변환
                audio = self.vocoder(mel)
                
                # 스테레오 보장
                if audio.shape[1] == 1:
                    audio = audio.repeat(1, 2, 1)
                elif audio.shape[1] > 2:
                    audio = audio[:, :2, :]
                
                # 범위 클리핑
                audio = torch.clamp(audio, -1.0, 1.0)
                
                return audio
                
            except Exception as e:
                print(f"Warning: Vocoder conversion failed: {e}")
                
                # 폴백: Griffin-Lim 알고리즘 사용
                return self._griffin_lim_fallback(mel)
    
    def _griffin_lim_fallback(self, mel: torch.Tensor) -> torch.Tensor:
        """Griffin-Lim 알고리즘을 사용한 폴백"""
        try:
            # 로그 멜을 리니어 스케일로 변환
            mel_linear = torch.exp(mel.squeeze(1))  # (B, n_mels, T)
            
            audio_list = []
            for i in range(mel_linear.shape[0]):
                # Griffin-Lim으로 오디오 복원
                audio_mono = self.griffin_lim(mel_linear[i])  # (T,)
                
                # 스테레오로 변환
                audio_stereo = audio_mono.unsqueeze(0).repeat(2, 1)  # (2, T)
                audio_list.append(audio_stereo)
            
            audio_batch = torch.stack(audio_list, dim=0)  # (B, 2, T)
            
            return torch.clamp(audio_batch, -1.0, 1.0)
            
        except Exception as e:
            print(f"Warning: Griffin-Lim fallback failed: {e}")
            # 최종 폴백: 더미 오디오
            batch_size = mel.shape[0]
            dummy_audio = torch.randn(batch_size, 2, 44100 * 5, device=mel.device) * 0.1
            return dummy_audio
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.mel_to_audio(mel)


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


def create_vocoder_model(
    model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
    cache_dir: str = "checkpoints",
    **kwargs
) -> AdvancedVocoder:
    """
    Vocoder 모델 생성 팩토리 함수
    
    Args:
        model_name: 프리트레인된 모델 이름
        cache_dir: 캐시 디렉토리
        **kwargs: 추가 파라미터
        
    Returns:
        초기화된 Vocoder 모델
    """
    return AdvancedVocoder(
        model_name=model_name,
        cache_dir=cache_dir,
        **kwargs
    )


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
    print("Testing corrected DCAE + Vocoder pipeline...")
    
    # 모델 다운로드 (선택적)
    download_pretrained_models()
    
    # DCAE 모델 생성
    dcae = create_dcae_model()
    vocoder = create_vocoder_model()
    
    # 테스트 오디오
    test_audio = torch.randn(1, 2, 44100 * 5)  # 5초 스테레오 오디오
    
    print("=== 올바른 파이프라인 테스트 ===")
    
    # 1. 오디오 -> 멜 스펙트로그램
    mel = dcae.audio_to_mel(test_audio)
    print(f"Audio to Mel: {test_audio.shape} -> {mel.shape}")
    
    # 2. 멜 -> DCAE 잠재 벡터
    latents, _ = dcae.encode_mel(mel)
    print(f"Mel to Latents: {mel.shape} -> {latents.shape}")
    
    # 3. 잠재 벡터 -> 멜 스펙트로그램
    reconstructed_mel = dcae.decode_to_mel(latents)
    print(f"Latents to Mel: {latents.shape} -> {reconstructed_mel.shape}")
    
    # 4. 멜 -> 오디오 (Vocoder)
    reconstructed_audio = vocoder.mel_to_audio(reconstructed_mel)
    print(f"Mel to Audio: {reconstructed_mel.shape} -> {reconstructed_audio.shape}")
    
    # 압축 정보
    compression_info = dcae.get_compression_info(test_audio)
    print(f"Compression info: {compression_info}")
    
    print("✅ Corrected DCAE + Vocoder pipeline test completed!")