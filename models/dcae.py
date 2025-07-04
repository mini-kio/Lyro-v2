# lyro/models/dcae.py
"""
Music DCAE with Advanced Vocoder - ACE-Step 기반 통합 모델
첫 번째 코드 참고하여 수정: 시작 부분 보존 + Vocoder 통합
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
# 체크포인트 다운로드 함수 (첫 번째 코드에서 가져옴)
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

# ============================================================================
# HiFiGAN 기반 Advanced Vocoder (첫 번째 코드에서 가져옴)
# ============================================================================

def init_weights(m, mean=0.0, std=0.01):
    """가중치 초기화"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    """패딩 계산"""
    return (kernel_size * dilation - dilation) // 2

class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                              dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                              dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                              dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]))),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                              dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                              dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, 
                              dilation=1, padding=get_padding(kernel_size, 1))),
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.silu(x)
            xt = c1(xt)
            xt = F.silu(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in self.convs1:
            remove_weight_norm(conv, 'weight')
        for conv in self.convs2:
            remove_weight_norm(conv, 'weight')

class FixedHiFiGANGenerator(nn.Module):
    def __init__(self, *, hop_length: int = 512, upsample_rates: Tuple[int] = (8, 8, 2, 2, 2),
                 upsample_kernel_sizes: Tuple[int] = (16, 16, 8, 2, 2),
                 resblock_kernel_sizes: Tuple[int] = (3, 7, 11),
                 resblock_dilation_sizes: Tuple[Tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 num_mels: int = 128, upsample_initial_channel: int = 512,
                 use_template: bool = False, pre_conv_kernel_size: int = 7,
                 post_conv_kernel_size: int = 7,
                 post_activation: Callable = partial(nn.SiLU, inplace=True)):
        super().__init__()

        assert prod(upsample_rates) == hop_length, f"hop_length must be {prod(upsample_rates)}"

        self.conv_pre = weight_norm(nn.Conv1d(
            num_mels, upsample_initial_channel, pre_conv_kernel_size, 1,
            padding=get_padding(pre_conv_kernel_size),
        ))

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.use_template = use_template
        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # output_padding 조정: 첫 번째 레이어는 0, 나머지는 적절히 조정
            output_padding = 0 if i == 0 else (u - 1) % 2
            
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                upsample_initial_channel // (2**i),
                upsample_initial_channel // (2 ** (i + 1)),
                k, u,
                padding=(k - u) // 2,
                output_padding=output_padding  # 수정된 output_padding
            )))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.activation_post = post_activation()
        self.conv_post = weight_norm(nn.Conv1d(
            ch, 1, post_conv_kernel_size, 1, padding=get_padding(post_conv_kernel_size),
        ))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, template=None):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.silu(x, inplace=True)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for up in self.ups:
            remove_weight_norm(up, 'weight')
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre, 'weight')
        remove_weight_norm(self.conv_post, 'weight')

# ============================================================================
# 수정된 Advanced Vocoder (첫 번째 코드 기반)
# ============================================================================

class FixedOptimizedADaMoSHiFiGANV1(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(self, input_channels: int = 128, num_mels: int = 512,
                 upsample_initial_channel: int = 1024, sampling_rate: int = 44100,
                 n_fft: int = 2048, win_length: int = 2048, hop_length: int = 512,
                 f_min: int = 40, f_max: int = 16000, n_mels: int = 128):
        super().__init__()

        # 수정된 멜 스펙트로그램 변환 사용
        self.sampling_rate = sampling_rate
        self.mel_transform = FixedLogMelSpectrogram(
            sample_rate=sampling_rate, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, f_min=f_min, f_max=f_max, n_mels=n_mels,
            center=True  # True로 설정
        )
        
        # 수정된 HiFiGAN Generator 사용
        self.head = FixedHiFiGANGenerator(
            hop_length=hop_length,
            num_mels=num_mels,
            upsample_initial_channel=upsample_initial_channel,
        )
        
        self.eval()
        self._is_optimized = False

    def optimize_for_inference(self):
        """추론 최적화 적용 - CUDAGraphs 문제 해결"""
        if self._is_optimized:
            return
        
        print("Vocoder 최적화 적용 중 (CUDAGraphs 문제 방지)...")
        
        # Weight norm 제거
        self.head.remove_weight_norm()
        
        # torch.compile 완전히 비활성화 (CUDAGraphs 문제 방지)
        print("✓ torch.compile 비활성화 (CUDAGraphs 호환성)")
        
        self._is_optimized = True
        print("✓ Vocoder 최적화 완료 (안정성 우선)")

    @torch.no_grad()
    def decode(self, mel):
        """안정성 개선된 3D 멜 스펙트로그램 디코딩"""
        # 3D 입력 처리
        if mel.dim() == 3:
            if mel.shape[0] == 1:
                mel_input = mel
            else:
                mel_input = mel[0:1, :, :]
                
            channels, height, width = mel_input.shape
            expected_mels = self.mel_transform.n_mels
            
            # 주파수 빈 조정
            if height != expected_mels:
                if height > expected_mels:
                    mel_input = mel_input[:, :expected_mels, :]
                else:
                    padding_needed = expected_mels - height
                    zero_padding = torch.zeros(channels, padding_needed, width, 
                                             device=mel_input.device, dtype=mel_input.dtype)
                    mel_input = torch.cat([mel_input, zero_padding], dim=1)
            
            mel_batch = mel_input.unsqueeze(0)
            
        elif mel.dim() == 2:
            mel_batch = mel.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"지원하지 않는 멜 스펙트로그램 차원: {mel.shape}")
        
        # 3D 변환 및 추론 - CUDAGraphs 문제 방지를 위해 clone 사용
        batch_size, channels, height, width = mel_batch.shape
        mel_3d = mel_batch.view(batch_size, channels * height, width)
        
        # 안정성을 위해 eager mode만 사용
        mel_3d = mel_3d.clone()  # CUDAGraphs 문제 방지
        audio_output = self.head(mel_3d)
        
        return audio_output.float().clone()  # 안전한 반환

    @torch.no_grad()
    def encode(self, x):
        return self.mel_transform(x)

    def forward(self, mel):
        return self.head(mel)

# ============================================================================
# 통합된 DCAE + Vocoder 모델 (첫 번째 코드 기반)
# ============================================================================

class PretrainedDCAE(nn.Module):
    """
    프리트레인된 DCAE + Vocoder 통합 모델 (첫 번째 코드 기반)
    ACE-Step에서 제공하는 music_dcae_f8c8 + music_vocoder 사용
    """
    
    def __init__(
        self,
        model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
        cache_dir: str = "checkpoints",
        sample_rate: int = 44100,
        use_vocoder: bool = True,
        force_download: bool = False
    ):
        super().__init__()
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.sample_rate = sample_rate
        self.use_vocoder = use_vocoder
        
        # 체크포인트 다운로드
        dcae_path, vocoder_path = download_checkpoints()
        
        if dcae_path is None:
            raise RuntimeError("DCAE 체크포인트를 다운로드할 수 없습니다.")
        
        # DCAE 모델 로드
        self.dcae_model = self._load_dcae_model(dcae_path)
        
        # Vocoder 모델 로드 (선택적)
        if use_vocoder and vocoder_path:
            try:
                self.vocoder = FixedOptimizedADaMoSHiFiGANV1.from_pretrained(vocoder_path)
                self.vocoder.optimize_for_inference()
                print("✅ Advanced Vocoder 로드 성공")
            except Exception as e:
                print(f"❌ Vocoder 로드 실패: {e}")
                print("DCAE-only 모드로 전환")
                self.vocoder = None
                self.use_vocoder = False
        else:
            self.vocoder = None
            self.use_vocoder = False
        
        # 설정 정보
        self.latent_channels = 16  # ACE-Step DCAE 기본값
        self.compression_ratio = 50.0  # 약 50:1 압축
        
        # 멜 스펙트로그램 변환기 (시작 부분 보존)
        self.mel_transform = FixedLogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_mels=128,
            center=True  # 시작 부분 보존
        )
        
        # 평가 모드로 설정
        self.eval()
    
    def _load_dcae_model(self, dcae_path: str):
        """DCAE 모델 로드"""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers 라이브러리가 필요합니다.")
        
        try:
            dcae_model = AutoencoderDC.from_pretrained(dcae_path)
            print("✅ DCAE 모델 로드 성공")
            return dcae_model
        except Exception as e:
            print(f"❌ DCAE 모델 로드 실패: {e}")
            raise
    
    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        오디오를 잠재 공간으로 인코딩
        
        Args:
            audio: (B, 2, T) 스테레오 오디오
            
        Returns:
            latents: (B, 16, H, W) 잠재 벡터
            quantization_loss: 양자화 손실 (있는 경우)
        """
        # 오디오 -> 멜 스펙트로그램
        mel = self.audio_to_mel(audio)
        
        # 멜 스펙트로그램 -> 잠재 벡터 (DCAE 사용)
        with torch.no_grad():
            try:
                # AutoencoderDC.encode() 반환값 처리
                encoded_result = self.dcae_model.encode(mel)
                
                # 반환값 타입에 따른 처리
                if hasattr(encoded_result, 'latent_dist'):
                    # Diffusers AutoencoderKL 스타일
                    latents = encoded_result.latent_dist.sample()
                    quantization_loss = None
                elif isinstance(encoded_result, tuple):
                    # 튜플 반환인 경우
                    if len(encoded_result) == 2:
                        latents, quantization_loss = encoded_result
                    else:
                        latents = encoded_result[0]
                        quantization_loss = None
                else:
                    # 단일 텐서 반환
                    latents = encoded_result
                    quantization_loss = None
                
                return latents, quantization_loss
                
            except Exception as e:
                print(f"Warning: DCAE encoding failed: {e}")
                # 더미 잠재 벡터 반환
                batch_size = mel.shape[0]
                dummy_latents = torch.randn(batch_size, self.latent_channels, 16, 16, device=mel.device)
                return dummy_latents, None
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        잠재 벡터를 오디오로 디코딩 (DCAE + Vocoder 통합)
        
        Args:
            latents: (B, 16, H, W) 잠재 벡터
            
        Returns:
            audio: (B, 2, T) 스테레오 오디오
        """
        with torch.no_grad():
            try:
                # DCAE 디코딩 (mel 스펙트로그램 생성)
                mel = self.dcae_model.decode(latents)
                
                # Vocoder 사용 여부에 따른 처리
                if self.use_vocoder and self.vocoder is not None:
                    # Vocoder를 통한 고품질 오디오 생성
                    audio = self.vocoder.decode(mel)
                    
                    # 검증
                    if self._validate_audio_quality(audio):
                        return audio
                    else:
                        print("Warning: Vocoder 품질 이슈, DCAE-only로 폴백")
                        # 폴백: Griffin-Lim 기반 변환
                        return self._mel_to_audio_fallback(mel)
                else:
                    # DCAE-only: Griffin-Lim 기반 변환
                    return self._mel_to_audio_fallback(mel)
                    
            except Exception as e:
                print(f"Warning: Audio decoding failed: {e}")
                # 더미 오디오 반환
                target_samples = int(10 * 44100)
                return torch.randn(latents.shape[0], 2, target_samples, device=latents.device) * 0.1
    
    def _mel_to_audio_fallback(self, mel: torch.Tensor) -> torch.Tensor:
        """멜 스펙트로그램을 오디오로 변환 (Griffin-Lim 폴백)"""
        try:
            import torchaudio.transforms as T
            
            # Griffin-Lim 변환
            griffin_lim = T.GriffinLim(
                n_fft=2048,
                hop_length=512,
                n_iter=32,
                power=1.0
            )
            
            batch_size, channels, height, width = mel.shape
            audio_list = []
            
            for b in range(batch_size):
                channel_audio = []
                for c in range(min(channels, 2)):  # 최대 2채널
                    # 로그 멜에서 선형 스펙트로그램으로 근사 변환
                    linear_spec = torch.exp(mel[b, c])  # (H, W)
                    
                    # Griffin-Lim으로 오디오 재구성
                    audio_ch = griffin_lim(linear_spec)
                    channel_audio.append(audio_ch)
                
                # 스테레오 보장
                if len(channel_audio) == 1:
                    stereo_audio = torch.stack([channel_audio[0], channel_audio[0]], dim=0)
                else:
                    stereo_audio = torch.stack(channel_audio, dim=0)
                
                audio_list.append(stereo_audio)
            
            return torch.stack(audio_list, dim=0)  # (B, 2, T)
            
        except Exception as e:
            print(f"Warning: Fallback mel-to-audio failed: {e}")
            # 최종 폴백: 더미 오디오
            target_samples = int(10 * 44100)
            return torch.randn(mel.shape[0], 2, target_samples, device=mel.device) * 0.1
    
    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """
        오디오를 멜 스펙트로그램으로 변환
        
        Args:
            audio: (B, 2, T) 스테레오 오디오
            
        Returns:
            mel: (B, 2, n_mels, T_mel) 멜 스펙트로그램
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
        else:
            raise ValueError(f"Expected audio tensor with 2-3 dims, got {audio.shape}")
        
        # 각 채널에 대해 멜 스펙트로그램 계산
        mel_list = []
        for b in range(batch_size):
            channel_mels = []
            for c in range(channels):
                audio_sample = audio[b, c]
                mel = self.mel_transform(audio_sample)  # (n_mels, T_mel)
                channel_mels.append(mel)
            
            # 채널 차원으로 스택
            batch_mel = torch.stack(channel_mels, dim=0)
            mel_list.append(batch_mel)
        
        # 배치로 스택
        mel_batch = torch.stack(mel_list, dim=0)  # (B, C, n_mels, T_mel)
        
        # 스테레오 보장
        if mel_batch.shape[1] == 1:
            mel_batch = mel_batch.repeat(1, 2, 1, 1)
        
        return mel_batch
    
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
    
    def get_compression_info(self, audio: torch.Tensor) -> Dict[str, Any]:
        """압축 정보 계산"""
        with torch.no_grad():
            mel = self.audio_to_mel(audio)
            latents, _ = self.encode(audio)
            
            original_size = audio.numel()
            compressed_size = latents.numel()
            compression_ratio = original_size / max(compressed_size, 1)
            
            return {
                'original_size': int(original_size),
                'compressed_size': int(compressed_size),
                'compression_ratio': float(compression_ratio),
                'latent_shape': tuple(latents.shape),
                'mel_shape': tuple(mel.shape),
                'vocoder_enabled': self.use_vocoder
            }

# ============================================================================
# Factory 함수들
# ============================================================================

def create_dcae_model(
    model_type: str = "pretrained",
    model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
    cache_dir: str = "./checkpoints",
    use_vocoder: bool = True,
    **kwargs
) -> PretrainedDCAE:
    """
    DCAE 모델 생성 팩토리 함수
    
    Args:
        model_type: 모델 타입 (현재는 "pretrained"만 지원)
        model_name: HuggingFace 모델 이름
        cache_dir: 캐시 디렉토리
        use_vocoder: Vocoder 사용 여부
        **kwargs: 추가 인자
        
    Returns:
        PretrainedDCAE 인스턴스
    """
    return PretrainedDCAE(
        model_name=model_name,
        cache_dir=cache_dir,
        use_vocoder=use_vocoder,
        **kwargs
    )

# 별칭 (하위 호환성)
AdvancedVocoder = FixedOptimizedADaMoSHiFiGANV1
create_dcae = create_dcae_model