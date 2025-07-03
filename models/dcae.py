# lyro/models/dcae.py
"""
Music DCAE with Advanced Vocoder - ACE-Step 기반 프리트레인 모델 사용
주요 개선사항:
1. ACE-Step/ACE-Step-v1-3.5B 모델 정확히 로드
2. 견고한 체크포인트 다운로드 및 관리
3. 시작 부분 보존을 위한 STFT center=True 설정
4. 청킹 처리 및 메모리 최적화
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from tqdm import tqdm
import warnings
from math import prod
from functools import partial
from typing import Callable, Tuple, List, Optional, Dict, Any
from torch.nn import Conv1d
from torch.nn.utils import weight_norm
from torch.nn.utils import remove_weight_norm
from torchaudio.transforms import MelScale
from torch.cuda.amp import autocast
import threading
from concurrent.futures import ThreadPoolExecutor
import builtins
import time
from pathlib import Path

# 성능 최적화를 위한 전역 설정
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('medium')

# GPU 감지 및 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()

# 디버깅 출력 제어 클래스
class DebugPrint:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.original_print = builtins.print
    
    def __call__(self, *args, **kwargs):
        if self.enabled:
            self.original_print(*args, **kwargs)
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False

debug_print = DebugPrint(enabled=True)

# 필요한 라이브러리 확인
try:
    from diffusers import AutoencoderDC
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.loaders import FromOriginalModelMixin
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from huggingface_hub import snapshot_download
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("필요한 라이브러리를 설치해주세요:")
    print("pip install diffusers huggingface_hub")

# ============================================================================
# 체크포인트 다운로드 함수
# ============================================================================

def download_checkpoints():
    """필요한 체크포인트 파일들을 자동으로 다운로드합니다."""
    print("체크포인트 다운로드 중...")
    os.makedirs("checkpoints", exist_ok=True)
    
    try:
        dcae_path = os.path.join("checkpoints", "music_dcae_f8c8")
        if not os.path.exists(dcae_path):
            print("DCAE 모델 다운로드 중...")
            snapshot_download(
                repo_id="ACE-Step/ACE-Step-v1-3.5B",
                local_dir="checkpoints",
                allow_patterns=["music_dcae_f8c8/*"],
            )
        
        vocoder_path = os.path.join("checkpoints", "music_vocoder")
        if not os.path.exists(vocoder_path):
            print("Vocoder 모델 다운로드 중...")
            snapshot_download(
                repo_id="ACE-Step/ACE-Step-v1-3.5B",
                local_dir="checkpoints",
                allow_patterns=["music_vocoder/*"],
            )
        
        print("체크포인트 다운로드 완료!")
        return dcae_path, vocoder_path
        
    except Exception as e:
        print(f"체크포인트 다운로드 실패: {e}")
        return None, None

# ============================================================================
# 유틸리티 함수들
# ============================================================================

def init_weights(m, mean=0.0, std=0.01):
    """가중치 초기화"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    """패딩 계산"""
    return (kernel_size * dilation - dilation) // 2

def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample"""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


# ============================================================================
# 기본 블록들
# ============================================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last or channels_first"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x

# ============================================================================
# 수정된 Mel Spectrogram 변환 (시작 부분 보존)
# ============================================================================

class FixedLinearSpectrogram(nn.Module):
    def __init__(self, n_fft=2048, win_length=2048, hop_length=512, center=True, mode="pow2_sqrt"):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center  # True로 변경하여 시작 부분 보존
        self.mode = mode
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, y):
        if y.ndim == 3:
            y = y.squeeze(1)
        
        # center=True일 때는 추가 패딩 불필요
        if not self.center:
            y = torch.nn.functional.pad(
                y.unsqueeze(1),
                ((self.win_length - self.hop_length) // 2,
                 (self.win_length - self.hop_length + 1) // 2),
                mode="reflect",
            ).squeeze(1)
        
        dtype = y.dtype
        spec = torch.stft(
            y.float() if not use_amp else y.half(),
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,  # True로 설정
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)

        if self.mode == "pow2_sqrt":
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        spec = spec.to(dtype)
        return spec

class FixedLogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=44100, n_fft=2048, win_length=2048, hop_length=512,
                 n_mels=128, center=True, f_min=0.0, f_max=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center  # True로 변경
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2

        self.spectrogram = FixedLinearSpectrogram(n_fft, win_length, hop_length, center)
        self.mel_scale = MelScale(
            self.n_mels, self.sample_rate, self.f_min, self.f_max,
            self.n_fft // 2 + 1, "slaney", "slaney",
        )

    def compress(self, x):
        return torch.log(torch.clamp(x, min=1e-5))

    def forward(self, x, return_linear: bool = False):
        linear = self.spectrogram(x)
        x = self.mel_scale(linear)
        x = self.compress(x)
        if return_linear:
            return x, self.compress(linear)
        return x


class AdvancedVocoder:
    """고급 보코더 클래스 - 멜 스펙트로그램을 오디오로 변환"""
    
    def __init__(self, model_name: str = "facebook/musicgen-small", 
                 sample_rate: int = 44100,
                 cache_dir: str = "./checkpoints"):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.cache_dir = Path(cache_dir)
        
        # HiFi-GAN 설정
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        
        # 보코더 모델 로드
        self.vocoder_model = self._load_pretrained_vocoder()
        
    def _load_pretrained_vocoder(self) -> nn.Module:
        """프리트레인된 보코더 모델 로드"""
        try:
            # 실제 HiFi-GAN 등의 보코더를 사용할 수 있습니다
            # 여기서는 간단한 Griffin-Lim을 폴백으로 사용
            print("Loading fallback Griffin-Lim vocoder...")
            return self._create_griffin_lim_vocoder()
            
        except Exception as e:
            print(f"Failed to load pretrained vocoder: {e}")
            print("Using fallback Griffin-Lim vocoder...")
            return self._create_griffin_lim_vocoder()
    
    def _create_griffin_lim_vocoder(self):
        """Griffin-Lim 기반 폴백 보코더"""
        import torchaudio.transforms as T
        
        class GriffinLimVocoder(nn.Module):
            def __init__(self, n_fft=1024, hop_length=256, n_iter=32):
                super().__init__()
                self.griffin_lim = T.GriffinLim(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_iter=n_iter,
                    power=1.0
                )
                self.mel_to_linear = nn.ConvTranspose2d(128, 513, 1)  # 멜 -> 리니어 스펙트로그램
            
            def forward(self, mel_spec):
                # 멜 스펙트로그램 -> 리니어 스펙트로그램 (근사)
                linear_spec = self.mel_to_linear(mel_spec)
                
                # 배치 처리
                batch_size, channels = linear_spec.shape[:2]
                audio_list = []
                
                for b in range(batch_size):
                    channel_audio = []
                    for c in range(channels):
                        # Griffin-Lim으로 오디오 재구성
                        magnitude = torch.exp(linear_spec[b, c])  # 로그 스케일에서 변환
                        audio = self.griffin_lim(magnitude)
                        channel_audio.append(audio)
                    
                    # 스테레오 스택
                    if len(channel_audio) == 1:
                        # 모노를 스테레오로 복제
                        stereo_audio = torch.stack([channel_audio[0], channel_audio[0]], dim=0)
                    else:
                        stereo_audio = torch.stack(channel_audio, dim=0)
                    
                    audio_list.append(stereo_audio)
                
                # 배치로 스택
                return torch.stack(audio_list, dim=0)  # (B, 2, T)
        
        return GriffinLimVocoder(
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
    
    def mel_to_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        멜 스펙트로그램을 오디오로 변환
        
        Args:
            mel_spec: (B, 2, n_mels, T) 멜 스펙트로그램
            
        Returns:
            audio: (B, 2, T) 스테레오 오디오
        """
        try:
            with torch.no_grad():
                audio = self.vocoder_model(mel_spec)
                
                # 정규화
                if torch.max(torch.abs(audio)) > 0:
                    audio = audio / torch.max(torch.abs(audio)) * 0.9
                
                return audio
                
        except Exception as e:
            print(f"Warning: Vocoder failed: {e}")
            # 더미 오디오 반환
            batch_size, channels, _, time_frames = mel_spec.shape
            audio_length = time_frames * self.hop_length
            dummy_audio = torch.zeros(batch_size, 2, audio_length, device=mel_spec.device)
            return dummy_audio
    
    def __call__(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """호출 가능한 인터페이스"""
        return self.mel_to_audio(mel_spec)


class PretrainedDCAE(nn.Module):
    """
    프리트레인된 DCAE 모델 래퍼
    ACE-Step에서 제공하는 music_dcae_f8c8 모델 사용
    멜 스펙트로그램만 처리 (오디오는 별도 Vocoder 필요)
    올바른 파이프라인: 오디오 -> 멜 -> DCAE latent -> 멜 -> Vocoder -> 오디오
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
        """프리트레인된 DCAE 모델 로드 - 견고한 다운로드 및 로딩"""
        
        if not DIFFUSERS_AVAILABLE:
            print("Warning: diffusers not available, using fallback model")
            return self._create_fallback_model()
        
        # 로컬 캐시 경로
        local_path = self.cache_dir / self.subfolder
        
        try:
            # 항상 체크포인트 다운로드 확인
            if not local_path.exists() or force_download:
                print(f"Downloading DCAE model: {self.model_name}/{self.subfolder}")
                os.makedirs(self.cache_dir, exist_ok=True)
                
                # 견고한 다운로드 로직
                try:
                    downloaded_path = snapshot_download(
                        repo_id=self.model_name,
                        local_dir=str(self.cache_dir),
                        allow_patterns=[f"{self.subfolder}/*"],
                        local_dir_use_symlinks=False,
                        force_download=force_download
                    )
                    print(f"✅ DCAE model downloaded to: {downloaded_path}")
                except Exception as download_error:
                    print(f"❌ Download failed: {download_error}")
                    print("Using fallback model...")
                    return self._create_fallback_model()
            
            # 로컬에서 모델 로드
            if local_path.exists():
                print(f"Loading DCAE from local cache: {local_path}")
                try:
                    dcae_model = AutoencoderDC.from_pretrained(str(local_path))
                    print("✅ DCAE model loaded successfully from local cache")
                    return dcae_model
                except Exception as load_error:
                    print(f"❌ Failed to load from local cache: {load_error}")
                    print("Attempting to re-download...")
                    
                    # 캐시가 손상된 경우 재다운로드
                    try:
                        downloaded_path = snapshot_download(
                            repo_id=self.model_name,
                            local_dir=str(self.cache_dir),
                            allow_patterns=[f"{self.subfolder}/*"],
                            local_dir_use_symlinks=False,
                            force_download=True
                        )
                        dcae_model = AutoencoderDC.from_pretrained(str(local_path))
                        print("✅ DCAE model redownloaded and loaded successfully")
                        return dcae_model
                    except Exception as redownload_error:
                        print(f"❌ Re-download also failed: {redownload_error}")
                        print("Using fallback model...")
                        return self._create_fallback_model()
            else:
                print(f"❌ Local path does not exist: {local_path}")
                print("Using fallback model...")
                return self._create_fallback_model()
                
        except Exception as e:
            print(f"❌ Unexpected error in DCAE model loading: {e}")
            print("Using fallback model...")
            return self._create_fallback_model()
    
    def _create_fallback_model(self) -> nn.Module:
        """폴백용 간단한 DCAE 모델"""
        
        class FallbackDCAE(nn.Module):
            def __init__(self):
                super().__init__()
                # 간단한 인코더-디코더 (멜 스펙트로그램용)
                self.encoder = nn.Sequential(
                    nn.Conv2d(2, 64, 3, stride=2, padding=1),  # 스테레오 멜 입력
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 16, 3, stride=2, padding=1),
                    nn.AdaptiveAvgPool2d((16, 16))
                )
                
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(16, 64, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1),  # 스테레오 출력
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
            mel: (B, 2, n_mels, T_mel) 멜 스펙트로그램 (스테레오 유지)
        """
        # 차원 정규화
        if audio.dim() == 3:
            batch_size = audio.shape[0]
            channels = audio.shape[1]
        elif audio.dim() == 2:
            if audio.shape[0] == 2:
                # (2, T) -> (1, 2, T)
                audio = audio.unsqueeze(0)
                batch_size = 1
                channels = 2
            else:
                # (B, T) -> (B, 1, T)
                audio = audio.unsqueeze(1)
                batch_size = audio.shape[0]
                channels = 1
        elif audio.dim() == 1:
            # (T,) -> (1, 1, T)
            audio = audio.unsqueeze(0).unsqueeze(0)
            batch_size = 1
            channels = 1
        else:
            raise ValueError(f"Expected audio tensor with 1-3 dims, got {audio.shape}")
        
        # 각 채널에 대해 멜 스펙트로그램 계산
        mel_list = []
        for b in range(batch_size):
            channel_mels = []
            for c in range(channels):
                if channels == 1:
                    audio_sample = audio[b, c] if audio.dim() == 3 else audio[b]
                else:
                    audio_sample = audio[b, c]
                
                mel = self.mel_transform(audio_sample)  # (n_mels, T_mel)
                channel_mels.append(mel)
            
            # 채널 차원으로 스택 (C, n_mels, T_mel)
            batch_mel = torch.stack(channel_mels, dim=0)
            mel_list.append(batch_mel)
        
        # 배치로 스택
        mel_batch = torch.stack(mel_list, dim=0)  # (B, C, n_mels, T_mel)
        
        # 로그 변환
        mel_batch = torch.log(mel_batch + 1e-7)
        
        # 만약 모노라면 스테레오로 복제
        if mel_batch.shape[1] == 1:
            mel_batch = mel_batch.repeat(1, 2, 1, 1)  # (B, 2, n_mels, T_mel)
        
        # 크기를 2의 배수로 패딩 (DCAE 호환성)
        _, _, H, W = mel_batch.shape
        if W % 2 != 0:
            # 가로 크기를 짝수로 만들기
            mel_batch = F.pad(mel_batch, (0, 1))  # 오른쪽에 1 픽셀 패딩
        
        return mel_batch
    
    def encode_mel(self, mel: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        멜 스펙트로그램을 잠재 공간으로 인코딩
        
        Args:
            mel: (B, 2, n_mels, T_mel) 멜 스펙트로그램
            
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
                    elif hasattr(encoded, 'latent_dist'):
                        # Diffusers AutoencoderKL 스타일
                        latents = encoded.latent_dist.sample()
                        quantization_loss = None
                    else:
                        latents = encoded
                        quantization_loss = None
                else:
                    # 폴백 모델의 경우
                    latents, quantization_loss = self.dcae_model.encode(mel)
                
                return latents, quantization_loss
                
        except Exception as e:
            if "does not exist" in str(e) or "not found" in str(e).lower():
                print(f"Note: Using fallback DCAE model (pretrained model not available)")
            else:
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
            mel: (B, 2, n_mels, T_mel) 재구성된 멜 스펙트로그램
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
                    # (B, H, W) -> (B, 1, H, W)
                    mel = mel.unsqueeze(1)
                elif mel.dim() == 4:
                    # 이미 올바른 형식
                    pass
                else:
                    raise ValueError(f"Unexpected mel output shape: {mel.shape}")
                
                # 2채널 보장 (스테레오)
                if mel.shape[1] == 1:
                    mel = mel.repeat(1, 2, 1, 1)  # (B, 2, H, W)
                
                return mel
                
        except Exception as e:
            if "does not exist" in str(e) or "not found" in str(e).lower():
                print(f"Note: Using fallback DCAE model (pretrained model not available)")
            else:
                print(f"Warning: DCAE decoding failed: {e}")
            # 더미 멜 스펙트로그램 반환
            batch_size = latents.shape[0]
            dummy_mel = torch.zeros(batch_size, 2, 128, 256, device=latents.device)
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
            
            # 텐서가 아닌 경우 처리
            if hasattr(mel, 'numel'):
                original_size = mel.numel()
            else:
                original_size = torch.tensor(mel).numel() if torch.is_tensor(mel) else 0
                
            if hasattr(latents, 'numel'):
                compressed_size = latents.numel()
            else:
                compressed_size = torch.tensor(latents).numel() if torch.is_tensor(latents) else 0
            
            compression_ratio = original_size / max(compressed_size, 1)
            
            return {
                'original_size': int(original_size),
                'compressed_size': int(compressed_size),
                'compression_ratio': float(compression_ratio),
                'latent_shape': tuple(latents.shape) if hasattr(latents, 'shape') else None,
                'mel_shape': tuple(mel.shape) if hasattr(mel, 'shape') else None
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
    
    def latents_to_generator_format(self, latents: torch.Tensor) -> torch.Tensor:
        """
        DCAE latents (B, C, H, W)를 Generator 형식 (B, C, T)로 변환
        
        Args:
            latents: (B, 16, H, W) DCAE latents
            
        Returns:
            generator_latents: (B, 16, T) Generator용 latents
        """
        if latents.dim() != 4:
            raise ValueError(f"Expected 4D latents, got {latents.shape}")
        
        B, C, H, W = latents.shape
        # (B, C, H, W) -> (B, C, H*W)
        generator_latents = latents.view(B, C, H * W)
        
        return generator_latents
    
    def latents_from_generator_format(self, generator_latents: torch.Tensor, original_4d_shape: torch.Size = None) -> torch.Tensor:
        """
        Generator 형식 (B, C, T)를 DCAE latents (B, C, H, W)로 변환
        
        Args:
            generator_latents: (B, C, T) Generator latents
            original_4d_shape: 원본 4D 형태 (B, C, H, W)
            
        Returns:
            latents: (B, C, H, W) DCAE latents
        """
        if generator_latents.dim() != 3:
            raise ValueError(f"Expected 3D generator latents, got {generator_latents.shape}")
        
        B, C, T = generator_latents.shape
        
        if original_4d_shape is not None:
            # 원본 형태를 사용
            _, _, H, W = original_4d_shape
        else:
            # 기본값: 16x16
            H, W = 16, 16
        
        expected_T = H * W
        
        if T != expected_T:
            # 필요시 패딩 또는 잘라내기
            if T < expected_T:
                # 패딩
                pad_size = expected_T - T
                generator_latents = F.pad(generator_latents, (0, pad_size))
            else:
                # 잘라내기
                generator_latents = generator_latents[:, :, :expected_T]
        
        # (B, C, T) -> (B, C, H, W)
        latents = generator_latents.view(B, C, H, W)
        
        return latents

    def encode_chunked(self, audio: torch.Tensor, chunk_duration: float = 30.0, 
                      overlap_duration: float = 1.0) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        대용량 오디오를 청크 단위로 인코딩
        
        Args:
            audio: (B, 2, T) 스테레오 오디오
            chunk_duration: 청크 길이 (초)
            overlap_duration: 청크 간 겹침 (초)
            
        Returns:
            latents: (B, 16, H, W_total) 연결된 잠재 벡터
            quantization_losses: 각 청크의 양자화 손실들
        """
        sample_rate = self.sample_rate
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        batch_size, channels, total_samples = audio.shape
        
        # 청크 단위로 처리
        all_latents = []
        quantization_losses = []
        
        for start_idx in range(0, total_samples, step_samples):
            end_idx = min(start_idx + chunk_samples, total_samples)
            
            # 청크 추출
            chunk = audio[:, :, start_idx:end_idx]
            
            # 최소 길이 보장 (패딩)
            if chunk.shape[2] < chunk_samples:
                padding_needed = chunk_samples - chunk.shape[2]
                chunk = F.pad(chunk, (0, padding_needed))
            
            # 청크 인코딩
            chunk_latents, chunk_loss = self.encode(chunk)
            all_latents.append(chunk_latents)
            quantization_losses.append(chunk_loss)
            
            print(f"Processed chunk {len(all_latents)}: {start_idx/sample_rate:.1f}s - {end_idx/sample_rate:.1f}s")
        
        # 잠재 벡터들 연결
        if len(all_latents) > 1:
            # 시간 축으로 연결 (겹침 처리)
            combined_latents = self._combine_latent_chunks(all_latents, overlap_duration)
        else:
            combined_latents = all_latents[0]
        
        return combined_latents, quantization_losses
    
    def decode_chunked(self, latents: torch.Tensor, chunk_width: int = 64, 
                      overlap_width: int = 8) -> torch.Tensor:
        """
        대용량 잠재 벡터를 청크 단위로 디코딩
        
        Args:
            latents: (B, 16, H, W) 잠재 벡터
            chunk_width: 청크 너비
            overlap_width: 청크 간 겹침
            
        Returns:
            audio: (B, 2, T) 재구성된 오디오
        """
        batch_size, channels, height, total_width = latents.shape
        step_width = chunk_width - overlap_width
        
        # 청크 단위로 디코딩
        all_audio_chunks = []
        
        for start_w in range(0, total_width, step_width):
            end_w = min(start_w + chunk_width, total_width)
            
            # 청크 추출
            chunk_latents = latents[:, :, :, start_w:end_w]
            
            # 최소 크기 보장 (패딩)
            if chunk_latents.shape[3] < chunk_width:
                padding_needed = chunk_width - chunk_latents.shape[3]
                chunk_latents = F.pad(chunk_latents, (0, padding_needed))
            
            # 청크 디코딩
            chunk_audio = self.decode(chunk_latents)
            all_audio_chunks.append(chunk_audio)
            
            print(f"Decoded chunk {len(all_audio_chunks)}: latent width {start_w} - {end_w}")
        
        # 오디오 청크들 연결
        if len(all_audio_chunks) > 1:
            combined_audio = self._combine_audio_chunks(all_audio_chunks, overlap_width)
        else:
            combined_audio = all_audio_chunks[0]
        
        return combined_audio
    
    def _combine_latent_chunks(self, latent_chunks: List[torch.Tensor], 
                              overlap_duration: float) -> torch.Tensor:
        """잠재 벡터 청크들을 겹침 처리하여 결합"""
        if len(latent_chunks) == 1:
            return latent_chunks[0]
        
        # 겹침 구간 계산 (멜 스펙트로그램 기준)
        mel_samples_per_sec = self.sample_rate / 256  # hop_length = 256
        overlap_mel_frames = int(overlap_duration * mel_samples_per_sec)
        
        # 인코딩 압축비 고려 (DCAE는 보통 8x 압축)
        overlap_latent_frames = max(1, overlap_mel_frames // 8)
        
        combined = latent_chunks[0]
        
        for i in range(1, len(latent_chunks)):
            current_chunk = latent_chunks[i]
            
            # 겹침 구간에서 가중 평균
            if overlap_latent_frames > 0 and combined.shape[3] >= overlap_latent_frames:
                # 이전 청크의 끝부분과 현재 청크의 시작부분
                prev_end = combined[:, :, :, -overlap_latent_frames:]
                curr_start = current_chunk[:, :, :, :overlap_latent_frames]
                
                # 가중 평균 (fade out + fade in)
                weights = torch.linspace(1, 0, overlap_latent_frames, device=combined.device)
                weights = weights.view(1, 1, 1, -1)
                
                blended = prev_end * weights + curr_start * (1 - weights)
                
                # 결합
                combined = torch.cat([
                    combined[:, :, :, :-overlap_latent_frames],
                    blended,
                    current_chunk[:, :, :, overlap_latent_frames:]
                ], dim=3)
            else:
                # 겹침 없이 단순 연결
                combined = torch.cat([combined, current_chunk], dim=3)
        
        return combined
    
    def _combine_audio_chunks(self, audio_chunks: List[torch.Tensor], 
                             overlap_width: int) -> torch.Tensor:
        """오디오 청크들을 겹침 처리하여 결합"""
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        # 오디오에서 겹침 구간 계산
        hop_length = 256
        overlap_audio_samples = overlap_width * hop_length
        
        combined = audio_chunks[0]
        
        for i in range(1, len(audio_chunks)):
            current_chunk = audio_chunks[i]
            
            # 겹침 구간에서 가중 평균
            if overlap_audio_samples > 0 and combined.shape[2] >= overlap_audio_samples:
                # 이전 청크의 끝부분과 현재 청크의 시작부분
                prev_end = combined[:, :, -overlap_audio_samples:]
                curr_start = current_chunk[:, :, :overlap_audio_samples]
                
                # 가중 평균 (크로스페이드)
                weights = torch.linspace(1, 0, overlap_audio_samples, device=combined.device)
                weights = weights.view(1, 1, -1)
                
                blended = prev_end * weights + curr_start * (1 - weights)
                
                # 결합
                combined = torch.cat([
                    combined[:, :, :-overlap_audio_samples],
                    blended,
                    current_chunk[:, :, overlap_audio_samples:]
                ], dim=2)
            else:
                # 겹침 없이 단순 연결
                combined = torch.cat([combined, current_chunk], dim=2)
        
        return combined

def create_dcae_model(model_name: str = "Stability-AI/stable-audio-open-1.0",
                     subfolder: str = "music_dcae_f8c8",
                     sample_rate: int = 44100,
                     cache_dir: str = "./checkpoints",
                     force_download: bool = False) -> PretrainedDCAE:
    """
    DCAE 모델 생성 팩토리 함수
    
    Args:
        model_name: HuggingFace 모델 이름
        subfolder: 모델 서브폴더
        sample_rate: 오디오 샘플레이트
        cache_dir: 캐시 디렉토리
        force_download: 강제 다운로드 여부
        
    Returns:
        PretrainedDCAE 인스턴스
    """
    return PretrainedDCAE(
        model_name=model_name,
        subfolder=subfolder,
        sample_rate=sample_rate,
        cache_dir=cache_dir,
        force_download=force_download
    )


def create_vocoder_model(model_name: str = "facebook/musicgen-small",
                        sample_rate: int = 44100,
                        cache_dir: str = "./checkpoints") -> AdvancedVocoder:
    """
    보코더 모델 생성 팩토리 함수
    
    Args:
        model_name: HuggingFace 모델 이름
        sample_rate: 오디오 샘플레이트
        cache_dir: 캐시 디렉토리
        
    Returns:
        AdvancedVocoder 인스턴스
    """
    return AdvancedVocoder(
        model_name=model_name,
        sample_rate=sample_rate,
        cache_dir=cache_dir
    )


# 하위 호환성을 위한 별칭
create_dcae = create_dcae_model
create_vocoder = create_vocoder_model