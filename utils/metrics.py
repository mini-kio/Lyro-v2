# lyro/utils/metrics.py
"""
LYRO 평가 메트릭
오디오 품질, 생성 품질, 텍스트-오디오 정렬 등 평가
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math
from scipy import signal
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")


@dataclass
class MetricResult:
    """메트릭 결과 구조체"""
    value: float
    name: str
    higher_is_better: bool = True
    unit: str = ""
    
    def __str__(self):
        direction = "↑" if self.higher_is_better else "↓"
        return f"{self.name}: {self.value:.4f}{self.unit} {direction}"


class AudioQualityMetrics:
    """오디오 품질 메트릭"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def compute_snr(self, clean: torch.Tensor, noisy: torch.Tensor) -> MetricResult:
        """
        Signal-to-Noise Ratio 계산
        
        Args:
            clean: 깨끗한 오디오
            noisy: 노이즈가 있는 오디오
            
        Returns:
            SNR 메트릭
        """
        # 같은 길이로 맞춤
        min_len = min(clean.shape[-1], noisy.shape[-1])
        clean = clean[..., :min_len]
        noisy = noisy[..., :min_len]
        
        # 신호 파워
        signal_power = torch.mean(clean ** 2)
        
        # 노이즈 파워
        noise = noisy - clean
        noise_power = torch.mean(noise ** 2)
        
        # SNR 계산 (dB)
        snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        
        return MetricResult(
            value=float(snr_db.item()),
            name="SNR",
            higher_is_better=True,
            unit="dB"
        )
    
    def compute_si_sdr(self, target: torch.Tensor, estimate: torch.Tensor) -> MetricResult:
        """
        Scale-Invariant Signal-to-Distortion Ratio 계산
        
        Args:
            target: 타겟 오디오
            estimate: 추정 오디오
            
        Returns:
            SI-SDR 메트릭
        """
        # 벡터로 변환
        target = target.flatten()
        estimate = estimate.flatten()
        
        # 같은 길이로 맞춤
        min_len = min(len(target), len(estimate))
        target = target[:min_len]
        estimate = estimate[:min_len]
        
        # Zero-mean
        target = target - torch.mean(target)
        estimate = estimate - torch.mean(estimate)
        
        # 스케일 팩터 계산
        alpha = torch.sum(estimate * target) / (torch.sum(target ** 2) + 1e-10)
        
        # 스케일된 타겟과 에러
        scaled_target = alpha * target
        error = estimate - scaled_target
        
        # SI-SDR 계산
        target_power = torch.sum(scaled_target ** 2)
        error_power = torch.sum(error ** 2)
        
        si_sdr = 10 * torch.log10(target_power / (error_power + 1e-10))
        
        return MetricResult(
            value=float(si_sdr.item()),
            name="SI-SDR",
            higher_is_better=True,
            unit="dB"
        )
    
    def compute_pesq(self, reference: torch.Tensor, degraded: torch.Tensor) -> MetricResult:
        """
        PESQ (간단한 근사 구현)
        실제 PESQ는 복잡하므로 스펙트럴 유사도로 근사
        
        Args:
            reference: 참조 오디오
            degraded: 열화된 오디오
            
        Returns:
            PESQ 근사 메트릭
        """
        # 모노로 변환
        if reference.dim() > 1:
            reference = reference.mean(dim=0)
        if degraded.dim() > 1:
            degraded = degraded.mean(dim=0)
        
        # 같은 길이로 맞춤
        min_len = min(reference.shape[-1], degraded.shape[-1])
        reference = reference[:min_len]
        degraded = degraded[:min_len]
        
        # 스펙트로그램 계산
        window = torch.hann_window(512, device=reference.device)
        
        ref_stft = torch.stft(reference, n_fft=512, hop_length=128, 
                             window=window, return_complex=True)
        deg_stft = torch.stft(degraded, n_fft=512, hop_length=128,
                             window=window, return_complex=True)
        
        ref_mag = torch.abs(ref_stft)
        deg_mag = torch.abs(deg_stft)
        
        # 스펙트럴 유사도
        similarity = F.cosine_similarity(
            ref_mag.flatten(), 
            deg_mag.flatten(), 
            dim=0
        )
        
        # PESQ 스케일로 변환 (1-5)
        pesq_score = 1 + 4 * similarity
        
        return MetricResult(
            value=float(pesq_score.item()),
            name="PESQ (approx)",
            higher_is_better=True,
            unit=""
        )
    
    def compute_stoi(self, clean: torch.Tensor, enhanced: torch.Tensor) -> MetricResult:
        """
        STOI (Short-Time Objective Intelligibility) 간단한 근사
        
        Args:
            clean: 깨끗한 오디오
            enhanced: 향상된 오디오
            
        Returns:
            STOI 근사 메트릭
        """
        # 모노로 변환
        if clean.dim() > 1:
            clean = clean.mean(dim=0)
        if enhanced.dim() > 1:
            enhanced = enhanced.mean(dim=0)
        
        # 같은 길이로 맞춤
        min_len = min(clean.shape[-1], enhanced.shape[-1])
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]
        
        # 프레임 단위로 상관계수 계산
        frame_length = int(0.03 * self.sample_rate)  # 30ms 프레임
        hop_length = frame_length // 2
        
        correlations = []
        
        for i in range(0, min_len - frame_length, hop_length):
            clean_frame = clean[i:i + frame_length]
            enhanced_frame = enhanced[i:i + frame_length]
            
            # 상관계수 계산
            corr = F.cosine_similarity(
                clean_frame.unsqueeze(0),
                enhanced_frame.unsqueeze(0),
                dim=1
            )
            correlations.append(corr.item())
        
        # 평균 상관계수
        stoi_score = np.mean(correlations) if correlations else 0.0
        
        return MetricResult(
            value=float(stoi_score),
            name="STOI (approx)",
            higher_is_better=True,
            unit=""
        )


class PerceptualMetrics:
    """지각적 메트릭"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
        # 멜 스펙트로그램 변환
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            f_min=0.0,
            f_max=sample_rate // 2
        )
    
    def compute_mel_distance(self, audio1: torch.Tensor, audio2: torch.Tensor) -> MetricResult:
        """
        멜 스펙트로그램 거리 계산
        
        Args:
            audio1: 첫 번째 오디오
            audio2: 두 번째 오디오
            
        Returns:
            멜 거리 메트릭
        """
        # 모노로 변환
        if audio1.dim() > 1:
            audio1 = audio1.mean(dim=0)
        if audio2.dim() > 1:
            audio2 = audio2.mean(dim=0)
        
        # 멜 스펙트로그램 계산
        mel1 = self.mel_transform(audio1)
        mel2 = self.mel_transform(audio2)
        
        # 같은 크기로 맞춤
        min_time = min(mel1.shape[-1], mel2.shape[-1])
        mel1 = mel1[..., :min_time]
        mel2 = mel2[..., :min_time]
        
        # L1 거리
        mel_distance = F.l1_loss(mel1, mel2)
        
        return MetricResult(
            value=float(mel_distance.item()),
            name="Mel Distance",
            higher_is_better=False,
            unit=""
        )
    
    def compute_spectral_convergence(self, target: torch.Tensor, generated: torch.Tensor) -> MetricResult:
        """
        스펙트럴 컨버전스 계산
        
        Args:
            target: 타겟 오디오
            generated: 생성된 오디오
            
        Returns:
            스펙트럴 컨버전스 메트릭
        """
        # STFT 계산
        window = torch.hann_window(1024, device=target.device)
        
        target_stft = torch.stft(target.flatten(), n_fft=1024, hop_length=256,
                                window=window, return_complex=True)
        generated_stft = torch.stft(generated.flatten(), n_fft=1024, hop_length=256,
                                   window=window, return_complex=True)
        
        target_mag = torch.abs(target_stft)
        generated_mag = torch.abs(generated_stft)
        
        # 같은 크기로 맞춤
        min_shape = [min(target_mag.shape[i], generated_mag.shape[i]) for i in range(2)]
        target_mag = target_mag[:min_shape[0], :min_shape[1]]
        generated_mag = generated_mag[:min_shape[0], :min_shape[1]]
        
        # 스펙트럴 컨버전스
        numerator = torch.norm(target_mag - generated_mag, p='fro')
        denominator = torch.norm(target_mag, p='fro')
        
        sc = numerator / (denominator + 1e-10)
        
        return MetricResult(
            value=float(sc.item()),
            name="Spectral Convergence",
            higher_is_better=False,
            unit=""
        )
    
    def compute_log_stft_magnitude_distance(self, target: torch.Tensor, generated: torch.Tensor) -> MetricResult:
        """
        로그 STFT 크기 거리 계산
        
        Args:
            target: 타겟 오디오
            generated: 생성된 오디오
            
        Returns:
            로그 STFT 거리 메트릭
        """
        # STFT 계산
        window = torch.hann_window(1024, device=target.device)
        
        target_stft = torch.stft(target.flatten(), n_fft=1024, hop_length=256,
                                window=window, return_complex=True)
        generated_stft = torch.stft(generated.flatten(), n_fft=1024, hop_length=256,
                                   window=window, return_complex=True)
        
        target_mag = torch.abs(target_stft)
        generated_mag = torch.abs(generated_stft)
        
        # 같은 크기로 맞춤
        min_shape = [min(target_mag.shape[i], generated_mag.shape[i]) for i in range(2)]
        target_mag = target_mag[:min_shape[0], :min_shape[1]]
        generated_mag = generated_mag[:min_shape[0], :min_shape[1]]
        
        # 로그 변환
        target_log = torch.log(target_mag + 1e-7)
        generated_log = torch.log(generated_mag + 1e-7)
        
        # L1 거리
        log_distance = F.l1_loss(target_log, generated_log)
        
        return MetricResult(
            value=float(log_distance.item()),
            name="Log STFT Distance",
            higher_is_better=False,
            unit=""
        )


class MusicMetrics:
    """음악 특화 메트릭"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def compute_harmonic_structure_similarity(self, audio1: torch.Tensor, audio2: torch.Tensor) -> MetricResult:
        """
        하모닉 구조 유사도 계산
        
        Args:
            audio1: 첫 번째 오디오
            audio2: 두 번째 오디오
            
        Returns:
            하모닉 유사도 메트릭
        """
        # 모노로 변환
        if audio1.dim() > 1:
            audio1 = audio1.mean(dim=0)
        if audio2.dim() > 1:
            audio2 = audio2.mean(dim=0)
        
        # 하모닉 성분 추출 (librosa 사용)
        try:
            audio1_np = audio1.cpu().numpy()
            audio2_np = audio2.cpu().numpy()
            
            # 하모닉/퍼커시브 분리
            harmonic1 = librosa.effects.harmonic(audio1_np, margin=3.0)
            harmonic2 = librosa.effects.harmonic(audio2_np, margin=3.0)
            
            # 크로마 특성 계산
            chroma1 = librosa.feature.chroma_stft(y=harmonic1, sr=self.sample_rate)
            chroma2 = librosa.feature.chroma_stft(y=harmonic2, sr=self.sample_rate)
            
            # 코사인 유사도
            similarity = np.mean([
                np.corrcoef(chroma1[:, i], chroma2[:, i])[0, 1] 
                for i in range(min(chroma1.shape[1], chroma2.shape[1]))
                if not np.isnan(np.corrcoef(chroma1[:, i], chroma2[:, i])[0, 1])
            ])
            
            if np.isnan(similarity):
                similarity = 0.0
                
        except Exception:
            # fallback: 스펙트럴 유사도
            window = torch.hann_window(1024, device=audio1.device)
            stft1 = torch.stft(audio1, n_fft=1024, hop_length=256, window=window, return_complex=True)
            stft2 = torch.stft(audio2, n_fft=1024, hop_length=256, window=window, return_complex=True)
            
            mag1 = torch.abs(stft1)
            mag2 = torch.abs(stft2)
            
            similarity = F.cosine_similarity(mag1.flatten(), mag2.flatten(), dim=0).item()
        
        return MetricResult(
            value=float(similarity),
            name="Harmonic Similarity",
            higher_is_better=True,
            unit=""
        )
    
    def compute_rhythm_similarity(self, audio1: torch.Tensor, audio2: torch.Tensor) -> MetricResult:
        """
        리듬 유사도 계산
        
        Args:
            audio1: 첫 번째 오디오
            audio2: 두 번째 오디오
            
        Returns:
            리듬 유사도 메트릭
        """
        # 모노로 변환
        if audio1.dim() > 1:
            audio1 = audio1.mean(dim=0)
        if audio2.dim() > 1:
            audio2 = audio2.mean(dim=0)
        
        try:
            audio1_np = audio1.cpu().numpy()
            audio2_np = audio2.cpu().numpy()
            
            # 템포 추출
            tempo1, beats1 = librosa.beat.beat_track(y=audio1_np, sr=self.sample_rate)
            tempo2, beats2 = librosa.beat.beat_track(y=audio2_np, sr=self.sample_rate)
            
            # 템포 유사도
            tempo_similarity = 1.0 - abs(tempo1 - tempo2) / max(tempo1, tempo2, 1.0)
            
            # 비트 패턴 유사도 (간단한 구현)
            if len(beats1) > 0 and len(beats2) > 0:
                # 비트 간격 패턴
                intervals1 = np.diff(beats1)
                intervals2 = np.diff(beats2)
                
                # 정규화
                if len(intervals1) > 0 and len(intervals2) > 0:
                    intervals1 = intervals1 / np.mean(intervals1)
                    intervals2 = intervals2 / np.mean(intervals2)
                    
                    # 유사도 계산
                    min_len = min(len(intervals1), len(intervals2))
                    if min_len > 0:
                        corr = np.corrcoef(intervals1[:min_len], intervals2[:min_len])[0, 1]
                        if np.isnan(corr):
                            corr = 0.0
                        pattern_similarity = max(0.0, corr)
                    else:
                        pattern_similarity = 0.0
                else:
                    pattern_similarity = 0.0
            else:
                pattern_similarity = 0.0
            
            # 전체 리듬 유사도
            rhythm_similarity = 0.5 * tempo_similarity + 0.5 * pattern_similarity
            
        except Exception:
            # fallback: 에너지 기반 유사도
            # 프레임별 에너지 계산
            frame_length = int(0.1 * self.sample_rate)  # 100ms 프레임
            hop_length = frame_length // 2
            
            energy1 = []
            energy2 = []
            
            for i in range(0, min(len(audio1), len(audio2)) - frame_length, hop_length):
                frame1 = audio1[i:i + frame_length]
                frame2 = audio2[i:i + frame_length]
                
                energy1.append(torch.mean(frame1 ** 2).item())
                energy2.append(torch.mean(frame2 ** 2).item())
            
            if len(energy1) > 1 and len(energy2) > 1:
                rhythm_similarity = float(np.corrcoef(energy1, energy2)[0, 1])
                if np.isnan(rhythm_similarity):
                    rhythm_similarity = 0.0
            else:
                rhythm_similarity = 0.0
        
        return MetricResult(
            value=float(rhythm_similarity),
            name="Rhythm Similarity",
            higher_is_better=True,
            unit=""
        )
    
    def compute_musical_key_similarity(self, audio1: torch.Tensor, audio2: torch.Tensor) -> MetricResult:
        """
        음악적 키 유사도 계산
        
        Args:
            audio1: 첫 번째 오디오
            audio2: 두 번째 오디오
            
        Returns:
            키 유사도 메트릭
        """
        # 모노로 변환
        if audio1.dim() > 1:
            audio1 = audio1.mean(dim=0)
        if audio2.dim() > 1:
            audio2 = audio2.mean(dim=0)
        
        try:
            audio1_np = audio1.cpu().numpy()
            audio2_np = audio2.cpu().numpy()
            
            # 크로마 특성 계산
            chroma1 = librosa.feature.chroma_stft(y=audio1_np, sr=self.sample_rate)
            chroma2 = librosa.feature.chroma_stft(y=audio2_np, sr=self.sample_rate)
            
            # 평균 크로마 벡터
            chroma1_mean = np.mean(chroma1, axis=1)
            chroma2_mean = np.mean(chroma2, axis=1)
            
            # 코사인 유사도
            similarity = np.dot(chroma1_mean, chroma2_mean) / (
                np.linalg.norm(chroma1_mean) * np.linalg.norm(chroma2_mean) + 1e-10
            )
            
            if np.isnan(similarity):
                similarity = 0.0
                
        except Exception:
            similarity = 0.0
        
        return MetricResult(
            value=float(similarity),
            name="Key Similarity",
            higher_is_better=True,
            unit=""
        )


class TextAudioAlignmentMetrics:
    """텍스트-오디오 정렬 메트릭"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def compute_semantic_alignment(self, audio: torch.Tensor, text: str, model=None) -> MetricResult:
        """
        의미적 정렬 점수 계산 (간단한 구현)
        실제로는 복잡한 멀티모달 모델이 필요
        
        Args:
            audio: 오디오 텐서
            text: 텍스트
            model: 정렬 평가 모델 (선택적)
            
        Returns:
            정렬 점수 메트릭
        """
        # 간단한 휴리스틱 구현
        # 실제로는 CLIP 스타일의 멀티모달 모델이 필요
        
        # 오디오 특성 추출
        if audio.dim() > 1:
            audio = audio.mean(dim=0)
        
        # 기본적인 오디오 특성
        rms_energy = torch.sqrt(torch.mean(audio ** 2)).item()
        zero_crossing_rate = torch.mean(torch.abs(torch.diff(torch.sign(audio)))).item() / 2
        
        # 텍스트 특성 (간단한 분석)
        text_features = self._extract_simple_text_features(text)
        
        # 간단한 정렬 점수 (실제로는 학습된 모델 필요)
        alignment_score = 0.5  # 기본값
        
        # 에너지와 텍스트 강도의 상관관계
        if text_features['intensity'] > 0.5 and rms_energy > 0.1:
            alignment_score += 0.2
        elif text_features['intensity'] < 0.3 and rms_energy < 0.05:
            alignment_score += 0.2
        
        # 리듬감과 텍스트 리듬의 상관관계
        if text_features['rhythm'] > 0.5 and zero_crossing_rate > 0.1:
            alignment_score += 0.2
        
        alignment_score = min(1.0, alignment_score)
        
        return MetricResult(
            value=float(alignment_score),
            name="Semantic Alignment",
            higher_is_better=True,
            unit=""
        )
    
    def _extract_simple_text_features(self, text: str) -> Dict[str, float]:
        """간단한 텍스트 특성 추출"""
        features = {
            'intensity': 0.5,  # 강도
            'rhythm': 0.5,     # 리듬감
            'emotion': 0.5,    # 감정
        }
        
        text_lower = text.lower()
        
        # 강도 키워드
        intensity_words = ['loud', 'strong', 'powerful', 'intense', 'hard', 'heavy']
        soft_words = ['soft', 'gentle', 'quiet', 'calm', 'peaceful']
        
        intensity_score = 0.5
        for word in intensity_words:
            if word in text_lower:
                intensity_score += 0.1
        for word in soft_words:
            if word in text_lower:
                intensity_score -= 0.1
        
        features['intensity'] = max(0.0, min(1.0, intensity_score))
        
        # 리듬감 키워드
        rhythm_words = ['fast', 'quick', 'rapid', 'dancing', 'beat', 'rhythm']
        slow_words = ['slow', 'relaxed', 'ballad', 'gentle']
        
        rhythm_score = 0.5
        for word in rhythm_words:
            if word in text_lower:
                rhythm_score += 0.1
        for word in slow_words:
            if word in text_lower:
                rhythm_score -= 0.1
        
        features['rhythm'] = max(0.0, min(1.0, rhythm_score))
        
        return features


class CompressionMetrics:
    """압축 관련 메트릭"""
    
    def compute_compression_ratio(self, original_size: int, compressed_size: int) -> MetricResult:
        """압축률 계산"""
        ratio = original_size / max(compressed_size, 1)
        
        return MetricResult(
            value=float(ratio),
            name="Compression Ratio",
            higher_is_better=True,
            unit=":1"
        )
    
    def compute_bitrate(self, audio: torch.Tensor, sample_rate: int, compressed_size: int) -> MetricResult:
        """비트레이트 계산"""
        duration = audio.shape[-1] / sample_rate
        bitrate_kbps = (compressed_size * 8) / (duration * 1000)
        
        return MetricResult(
            value=float(bitrate_kbps),
            name="Bitrate",
            higher_is_better=False,
            unit="kbps"
        )


class MetricCalculator:
    """통합 메트릭 계산기"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
        # 각 메트릭 클래스 초기화
        self.audio_quality = AudioQualityMetrics(sample_rate)
        self.perceptual = PerceptualMetrics(sample_rate)
        self.music = MusicMetrics(sample_rate)
        self.alignment = TextAudioAlignmentMetrics(sample_rate)
        self.compression = CompressionMetrics()
    
    def compute_all_metrics(
        self,
        target_audio: torch.Tensor,
        generated_audio: torch.Tensor,
        text: Optional[str] = None,
        compression_info: Optional[Dict] = None
    ) -> Dict[str, MetricResult]:
        """
        모든 메트릭 계산
        
        Args:
            target_audio: 타겟 오디오
            generated_audio: 생성된 오디오
            text: 관련 텍스트 (선택적)
            compression_info: 압축 정보 (선택적)
            
        Returns:
            메트릭 결과 딕셔너리
        """
        metrics = {}
        
        try:
            # 오디오 품질 메트릭
            metrics['snr'] = self.audio_quality.compute_snr(target_audio, generated_audio)
            metrics['si_sdr'] = self.audio_quality.compute_si_sdr(target_audio, generated_audio)
            metrics['pesq'] = self.audio_quality.compute_pesq(target_audio, generated_audio)
            metrics['stoi'] = self.audio_quality.compute_stoi(target_audio, generated_audio)
            
            # 지각적 메트릭
            metrics['mel_distance'] = self.perceptual.compute_mel_distance(target_audio, generated_audio)
            metrics['spectral_convergence'] = self.perceptual.compute_spectral_convergence(target_audio, generated_audio)
            metrics['log_stft_distance'] = self.perceptual.compute_log_stft_magnitude_distance(target_audio, generated_audio)
            
            # 음악 메트릭
            metrics['harmonic_similarity'] = self.music.compute_harmonic_structure_similarity(target_audio, generated_audio)
            metrics['rhythm_similarity'] = self.music.compute_rhythm_similarity(target_audio, generated_audio)
            metrics['key_similarity'] = self.music.compute_musical_key_similarity(target_audio, generated_audio)
            
            # 텍스트-오디오 정렬 (텍스트가 있는 경우)
            if text:
                metrics['semantic_alignment'] = self.alignment.compute_semantic_alignment(generated_audio, text)
            
            # 압축 메트릭 (압축 정보가 있는 경우)
            if compression_info:
                if 'original_size' in compression_info and 'compressed_size' in compression_info:
                    metrics['compression_ratio'] = self.compression.compute_compression_ratio(
                        compression_info['original_size'],
                        compression_info['compressed_size']
                    )
                
                if 'compressed_size' in compression_info:
                    metrics['bitrate'] = self.compression.compute_bitrate(
                        target_audio,
                        self.sample_rate,
                        compression_info['compressed_size']
                    )
        
        except Exception as e:
            print(f"Error computing metrics: {e}")
        
        return metrics
    
    def get_summary(self, metrics: Dict[str, MetricResult]) -> Dict[str, float]:
        """메트릭 요약 생성"""
        summary = {}
        
        for name, metric in metrics.items():
            summary[name] = metric.value
        
        # 전체 품질 점수 계산 (가중 평균)
        quality_score = 0.0
        weights = {
            'snr': 0.2,
            'si_sdr': 0.2,
            'mel_distance': -0.15,  # 낮을수록 좋음
            'harmonic_similarity': 0.15,
            'rhythm_similarity': 0.15,
            'semantic_alignment': 0.15,
        }
        
        total_weight = 0.0
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                value = metrics[metric_name].value
                if weight < 0:  # 낮을수록 좋은 메트릭
                    value = 1.0 / (1.0 + value)  # 역변환
                    weight = abs(weight)
                
                quality_score += weight * value
                total_weight += weight
        
        if total_weight > 0:
            quality_score /= total_weight
            summary['overall_quality'] = quality_score
        
        return summary
