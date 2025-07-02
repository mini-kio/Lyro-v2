# lyro/utils/audio.py
"""
LYRO 오디오 처리 유틸리티
오디오 로딩, 전처리, 변환, 검증 등의 기능 제공
"""

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Union, List, Dict, Any
import warnings
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """오디오 처리 설정"""
    sample_rate: int = 44100
    channels: int = 2
    bit_depth: int = 16
    normalize: bool = True
    max_length: Optional[int] = None


class AudioProcessor:
    """오디오 처리 메인 클래스"""
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        
        # 변환 객체들 초기화
        self._init_transforms()
    
    def _init_transforms(self):
        """변환 객체들 초기화"""
        sr = self.config.sample_rate
        
        # 스펙트로그램 변환
        self.stft_transform = T.Spectrogram(
            n_fft=1024,
            hop_length=256,
            power=None,  # Complex spectrogram
        )
        
        # 멜 스펙트로그램
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            f_min=0.0,
            f_max=sr // 2,
        )
        
        # MFCC
        self.mfcc_transform = T.MFCC(
            sample_rate=sr,
            n_mfcc=13,
            melkwargs={
                "n_fft": 1024,
                "hop_length": 256,
                "n_mels": 80,
            }
        )
    
    def load_audio(
        self, 
        path: Union[str, Path], 
        target_sr: Optional[int] = None,
        mono: bool = False,
        normalize: bool = None
    ) -> Tuple[torch.Tensor, int]:
        """
        오디오 파일 로딩
        
        Args:
            path: 오디오 파일 경로
            target_sr: 타겟 샘플레이트 (None이면 config 사용)
            mono: 모노로 변환할지 여부
            normalize: 정규화 여부
            
        Returns:
            (audio_tensor, sample_rate)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        target_sr = target_sr or self.config.sample_rate
        normalize = normalize if normalize is not None else self.config.normalize
        
        try:
            # torchaudio로 로딩
            audio, sr = torchaudio.load(path)
            
            # 샘플레이트 변환
            if sr != target_sr:
                resampler = T.Resample(sr, target_sr)
                audio = resampler(audio)
                sr = target_sr
            
            # 채널 처리 - 차원 안전성 확보
            if audio.dim() == 1:
                # (T,) -> (1, T) 단일 채널
                audio = audio.unsqueeze(0)
            elif audio.dim() == 2:
                # (C, T) 형태 확인
                if mono and audio.shape[0] > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                elif not mono and audio.shape[0] == 1 and self.config.channels == 2:
                    audio = audio.repeat(2, 1)
            else:
                raise ValueError(f"Unexpected audio tensor dimensions: {audio.shape}")
            
            # 정규화
            if normalize:
                audio = self.normalize_audio(audio)
            
            return audio, sr
            
        except Exception as e:
            # fallback to librosa
            try:
                audio_np, sr = librosa.load(
                    str(path), 
                    sr=target_sr, 
                    mono=mono
                )
                audio = torch.from_numpy(audio_np).float()
                
                if not mono and audio.dim() == 1 and self.config.channels == 2:
                    audio = audio.unsqueeze(0).repeat(2, 1)
                elif audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                
                if normalize:
                    audio = self.normalize_audio(audio)
                
                return audio, sr
                
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio {path}: {e}, {e2}")
    
    def save_audio(
        self,
        audio: torch.Tensor,
        path: Union[str, Path],
        sample_rate: int = None,
        bit_depth: int = None
    ):
        """
        오디오 저장
        
        Args:
            audio: 오디오 텐서 (C, T)
            path: 저장 경로
            sample_rate: 샘플레이트
            bit_depth: 비트 깊이
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        sample_rate = sample_rate or self.config.sample_rate
        bit_depth = bit_depth or self.config.bit_depth
        
        # 범위 클리핑
        audio = torch.clamp(audio, -1.0, 1.0)
        
        # 비트 깊이에 따른 변환
        if bit_depth == 16:
            audio = (audio * 32767).short()
        elif bit_depth == 24:
            audio = (audio * 8388607).int()
        elif bit_depth == 32:
            audio = audio.float()
        
        try:
            torchaudio.save(str(path), audio, sample_rate)
        except Exception:
            # fallback to soundfile
            audio_np = audio.cpu().numpy()
            if audio_np.ndim == 2:
                audio_np = audio_np.T  # (T, C)
            sf.write(str(path), audio_np, sample_rate)
    
    def normalize_audio(self, audio: torch.Tensor, method: str = "peak") -> torch.Tensor:
        """
        오디오 정규화
        
        Args:
            audio: 오디오 텐서
            method: 정규화 방법 (peak, rms, lufs)
            
        Returns:
            정규화된 오디오
        """
        if method == "peak":
            # Peak 정규화
            peak = torch.max(torch.abs(audio))
            if peak > 0:
                audio = audio / peak
                
        elif method == "rms":
            # RMS 정규화
            rms = torch.sqrt(torch.mean(audio ** 2))
            target_rms = 0.1  # -20dB
            if rms > 0:
                audio = audio * (target_rms / rms)
                
        elif method == "lufs":
            # 간단한 LUFS 근사
            audio = self.normalize_loudness(audio, target_lufs=-23.0)
        
        return torch.clamp(audio, -1.0, 1.0)
    
    def normalize_loudness(self, audio: torch.Tensor, target_lufs: float = -23.0) -> torch.Tensor:
        """
        라우드니스 정규화 (간단한 구현)
        
        Args:
            audio: 오디오 텐서
            target_lufs: 타겟 LUFS 값
            
        Returns:
            정규화된 오디오
        """
        # 간단한 RMS 기반 라우드니스 근사
        rms = torch.sqrt(torch.mean(audio ** 2))
        
        if rms > 0:
            # LUFS를 RMS로 근사 변환
            target_rms = 10 ** (target_lufs / 20)
            audio = audio * (target_rms / rms)
        
        return torch.clamp(audio, -1.0, 1.0)
    
    def trim_silence(
        self, 
        audio: torch.Tensor, 
        threshold_db: float = -40.0,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> torch.Tensor:
        """
        무음 구간 제거
        
        Args:
            audio: 오디오 텐서
            threshold_db: 무음 판단 임계값 (dB)
            frame_length: 프레임 길이
            hop_length: 홉 길이
            
        Returns:
            무음이 제거된 오디오
        """
        # 에너지 계산
        frame_length = min(frame_length, audio.shape[-1])
        
        # 모노로 변환하여 에너지 계산
        mono_audio = audio.mean(dim=0) if audio.dim() > 1 else audio
        
        # 프레임별 에너지
        frames = mono_audio.unfold(-1, frame_length, hop_length)
        energy = torch.mean(frames ** 2, dim=-1)
        
        # dB 변환
        energy_db = 10 * torch.log10(energy + 1e-10)
        
        # 무음이 아닌 프레임 찾기
        non_silent = energy_db > threshold_db
        
        if non_silent.any():
            start_frame = torch.where(non_silent)[0][0].item()
            end_frame = torch.where(non_silent)[0][-1].item()
            
            start_sample = start_frame * hop_length
            end_sample = min((end_frame + 1) * hop_length + frame_length, audio.shape[-1])
            
            return audio[..., start_sample:end_sample]
        else:
            return audio
    
    def adjust_length(
        self, 
        audio: torch.Tensor, 
        target_length: int,
        method: str = "pad"
    ) -> torch.Tensor:
        """
        오디오 길이 조정
        
        Args:
            audio: 오디오 텐서
            target_length: 타겟 길이
            method: 조정 방법 (pad, crop, stretch)
            
        Returns:
            길이가 조정된 오디오
        """
        current_length = audio.shape[-1]
        
        if current_length == target_length:
            return audio
        
        if method == "pad":
            if current_length < target_length:
                # 패딩
                pad_length = target_length - current_length
                audio = F.pad(audio, (0, pad_length))
            else:
                # 크롭
                start = (current_length - target_length) // 2
                audio = audio[..., start:start + target_length]
                
        elif method == "crop":
            if current_length > target_length:
                start = (current_length - target_length) // 2
                audio = audio[..., start:start + target_length]
            else:
                # 반복 패딩
                repeat_factor = (target_length // current_length) + 1
                audio = audio.repeat(1, repeat_factor)
                audio = audio[..., :target_length]
                
        elif method == "stretch":
            # 시간 스트레칭 (간단한 리샘플링)
            stretch_factor = target_length / current_length
            new_sr = int(self.config.sample_rate / stretch_factor)
            
            resampler = T.Resample(self.config.sample_rate, new_sr)
            audio = resampler(audio)
            
            # 타겟 길이에 맞춤
            if audio.shape[-1] != target_length:
                audio = F.interpolate(
                    audio.unsqueeze(0), 
                    size=target_length, 
                    mode='linear', 
                    align_corners=False
                ).squeeze(0)
        
        return audio


class AudioAugmentation:
    """오디오 증강 클래스"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def add_noise(
        self, 
        audio: torch.Tensor, 
        noise_level: float = 0.01,
        noise_type: str = "white"
    ) -> torch.Tensor:
        """
        노이즈 추가
        
        Args:
            audio: 오디오 텐서
            noise_level: 노이즈 레벨
            noise_type: 노이즈 타입 (white, pink)
            
        Returns:
            노이즈가 추가된 오디오
        """
        if noise_type == "white":
            noise = torch.randn_like(audio) * noise_level
        elif noise_type == "pink":
            # 간단한 핑크 노이즈 근사
            white_noise = torch.randn_like(audio)
            # 고주파 감쇠
            noise = self._apply_pink_filter(white_noise) * noise_level
        else:
            noise = torch.zeros_like(audio)
        
        return audio + noise
    
    def time_stretch(self, audio: torch.Tensor, stretch_factor: float) -> torch.Tensor:
        """
        시간 스트레칭
        
        Args:
            audio: 오디오 텐서
            stretch_factor: 스트레치 팩터 (1.0보다 크면 느려짐)
            
        Returns:
            스트레치된 오디오
        """
        if stretch_factor == 1.0:
            return audio
        
        # 새로운 길이 계산
        new_length = int(audio.shape[-1] * stretch_factor)
        
        # 보간
        audio_stretched = F.interpolate(
            audio.unsqueeze(0),
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze(0)
        
        return audio_stretched
    
    def pitch_shift(self, audio: torch.Tensor, semitones: float) -> torch.Tensor:
        """
        피치 시프트 (간단한 구현)
        
        Args:
            audio: 오디오 텐서
            semitones: 세미톤 단위 피치 변화
            
        Returns:
            피치가 변경된 오디오
        """
        if semitones == 0:
            return audio
        
        # 피치 시프트를 리샘플링으로 근사
        pitch_factor = 2 ** (semitones / 12.0)
        
        # 시간 스트레치 (피치 변화)
        stretched = self.time_stretch(audio, 1.0 / pitch_factor)
        
        # 원래 길이로 복원
        original_length = audio.shape[-1]
        if stretched.shape[-1] != original_length:
            stretched = F.interpolate(
                stretched.unsqueeze(0),
                size=original_length,
                mode='linear',
                align_corners=False
            ).squeeze(0)
        
        return stretched
    
    def apply_gain(self, audio: torch.Tensor, gain_db: float) -> torch.Tensor:
        """
        게인 적용
        
        Args:
            audio: 오디오 텐서
            gain_db: dB 단위 게인
            
        Returns:
            게인이 적용된 오디오
        """
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    def apply_eq(
        self, 
        audio: torch.Tensor, 
        low_gain: float = 0.0,
        mid_gain: float = 0.0,
        high_gain: float = 0.0
    ) -> torch.Tensor:
        """
        간단한 3밴드 EQ
        
        Args:
            audio: 오디오 텐서
            low_gain: 저주파 게인 (dB)
            mid_gain: 중간주파 게인 (dB)
            high_gain: 고주파 게인 (dB)
            
        Returns:
            EQ가 적용된 오디오
        """
        # STFT
        stft = torch.stft(
            audio.view(-1), 
            n_fft=1024, 
            hop_length=256, 
            return_complex=True
        )
        
        freq_bins = stft.shape[0]
        
        # 주파수 밴드 정의
        low_cutoff = freq_bins // 4
        high_cutoff = freq_bins * 3 // 4
        
        # 게인 적용
        stft[:low_cutoff] *= 10 ** (low_gain / 20)
        stft[low_cutoff:high_cutoff] *= 10 ** (mid_gain / 20)
        stft[high_cutoff:] *= 10 ** (high_gain / 20)
        
        # ISTFT
        audio_eq = torch.istft(stft, n_fft=1024, hop_length=256)
        
        # 원래 shape으로 복원
        if audio.dim() > 1:
            audio_eq = audio_eq.view(audio.shape)
        
        return audio_eq
    
    def _apply_pink_filter(self, white_noise: torch.Tensor) -> torch.Tensor:
        """핑크 노이즈 필터 적용"""
        # STFT
        stft = torch.stft(
            white_noise.view(-1),
            n_fft=1024,
            hop_length=256,
            return_complex=True
        )
        
        # 주파수별 감쇠 (1/f 특성)
        freq_bins = stft.shape[0]
        freqs = torch.arange(1, freq_bins + 1, dtype=torch.float32)
        
        # 1/f 감쇠
        attenuation = 1.0 / torch.sqrt(freqs)
        attenuation = attenuation.view(-1, 1)
        
        stft *= attenuation
        
        # ISTFT
        pink_noise = torch.istft(stft, n_fft=1024, hop_length=256)
        
        # 원래 shape으로 복원
        if white_noise.dim() > 1:
            pink_noise = pink_noise.view(white_noise.shape)
        
        return pink_noise


class AudioAnalyzer:
    """오디오 분석 클래스"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def get_features(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        오디오 특성 추출
        
        Args:
            audio: 오디오 텐서
            
        Returns:
            특성 딕셔너리
        """
        features = {}
        
        # 모노 변환
        mono_audio = audio.mean(dim=0) if audio.dim() > 1 else audio
        
        # RMS 에너지
        features['rms'] = torch.sqrt(torch.mean(mono_audio ** 2))
        
        # 제로 크로싱 비율
        features['zcr'] = self._zero_crossing_rate(mono_audio)
        
        # 스펙트럴 중심
        features['spectral_centroid'] = self._spectral_centroid(mono_audio)
        
        # 스펙트럴 대역폭
        features['spectral_bandwidth'] = self._spectral_bandwidth(mono_audio)
        
        # 스펙트럴 롤오프
        features['spectral_rolloff'] = self._spectral_rolloff(mono_audio)
        
        return features
    
    def _zero_crossing_rate(self, audio: torch.Tensor) -> torch.Tensor:
        """제로 크로싱 비율 계산"""
        signs = torch.sign(audio)
        sign_changes = torch.abs(torch.diff(signs))
        zcr = torch.mean(sign_changes) / 2.0
        return zcr
    
    def _spectral_centroid(self, audio: torch.Tensor) -> torch.Tensor:
        """스펙트럴 중심 계산"""
        stft = torch.stft(audio, n_fft=1024, hop_length=256, return_complex=True)
        magnitude = torch.abs(stft)
        
        freqs = torch.linspace(0, self.sample_rate / 2, magnitude.shape[0])
        
        # 가중 평균
        weighted_sum = torch.sum(magnitude * freqs.view(-1, 1), dim=0)
        total_magnitude = torch.sum(magnitude, dim=0)
        
        centroid = weighted_sum / (total_magnitude + 1e-10)
        return torch.mean(centroid)
    
    def _spectral_bandwidth(self, audio: torch.Tensor) -> torch.Tensor:
        """스펙트럴 대역폭 계산"""
        stft = torch.stft(audio, n_fft=1024, hop_length=256, return_complex=True)
        magnitude = torch.abs(stft)
        
        freqs = torch.linspace(0, self.sample_rate / 2, magnitude.shape[0])
        
        # 중심 주파수
        centroid = self._spectral_centroid(audio)
        
        # 대역폭 계산
        diff_squared = (freqs.view(-1, 1) - centroid) ** 2
        weighted_variance = torch.sum(magnitude * diff_squared, dim=0)
        total_magnitude = torch.sum(magnitude, dim=0)
        
        bandwidth = torch.sqrt(weighted_variance / (total_magnitude + 1e-10))
        return torch.mean(bandwidth)
    
    def _spectral_rolloff(self, audio: torch.Tensor, percentile: float = 0.85) -> torch.Tensor:
        """스펙트럴 롤오프 계산"""
        stft = torch.stft(audio, n_fft=1024, hop_length=256, return_complex=True)
        magnitude = torch.abs(stft)
        
        # 누적 에너지
        cumulative_energy = torch.cumsum(magnitude, dim=0)
        total_energy = torch.sum(magnitude, dim=0)
        
        # 롤오프 지점 찾기
        rolloff_threshold = total_energy * percentile
        rolloff_indices = torch.argmax((cumulative_energy >= rolloff_threshold).float(), dim=0)
        
        # 주파수로 변환
        freqs = torch.linspace(0, self.sample_rate / 2, magnitude.shape[0])
        rolloff_freqs = freqs[rolloff_indices]
        
        return torch.mean(rolloff_freqs)


def ensure_audio_format(
    audio: torch.Tensor,
    target_channels: int = 2,
    target_length: Optional[int] = None,
    sample_rate: int = 44100
) -> torch.Tensor:
    """
    오디오를 지정된 포맷으로 변환
    
    Args:
        audio: 입력 오디오
        target_channels: 타겟 채널 수
        target_length: 타겟 길이
        sample_rate: 샘플레이트
        
    Returns:
        포맷이 맞춰진 오디오
    """
    # 채널 처리
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    current_channels = audio.shape[0]
    
    if current_channels < target_channels:
        # 채널 복제
        audio = audio.repeat(target_channels, 1)
    elif current_channels > target_channels:
        # 채널 다운믹스
        if target_channels == 1:
            audio = audio.mean(dim=0, keepdim=True)
        else:
            audio = audio[:target_channels]
    
    # 길이 처리
    if target_length is not None:
        current_length = audio.shape[-1]
        
        if current_length < target_length:
            # 패딩
            pad_length = target_length - current_length
            audio = F.pad(audio, (0, pad_length))
        elif current_length > target_length:
            # 크롭
            audio = audio[..., :target_length]
    
    return audio


def detect_audio_quality(audio: torch.Tensor, sample_rate: int = 44100) -> Dict[str, float]:
    """
    오디오 품질 검사
    
    Args:
        audio: 오디오 텐서
        sample_rate: 샘플레이트
        
    Returns:
        품질 메트릭들
    """
    quality_metrics = {}
    
    # 클리핑 검사
    clipping_ratio = torch.mean((torch.abs(audio) >= 0.99).float()).item()
    quality_metrics['clipping_ratio'] = clipping_ratio
    
    # 동적 범위
    peak = torch.max(torch.abs(audio)).item()
    rms = torch.sqrt(torch.mean(audio ** 2)).item()
    dynamic_range = peak / (rms + 1e-10)
    quality_metrics['dynamic_range'] = dynamic_range
    
    # SNR 추정 (고주파를 노이즈로 가정)
    # 간단한 고주파 필터링
    mono_audio = audio.mean(dim=0) if audio.dim() > 1 else audio
    
    # 고주파 추출 (간단한 차분)
    high_freq = mono_audio[1:] - mono_audio[:-1]
    noise_power = torch.mean(high_freq ** 2).item()
    signal_power = torch.mean(mono_audio ** 2).item()
    
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    quality_metrics['estimated_snr'] = snr
    
    # 무음 비율
    silence_threshold = 0.01
    silence_ratio = torch.mean((torch.abs(audio) < silence_threshold).float()).item()
    quality_metrics['silence_ratio'] = silence_ratio
    
    return quality_metrics
