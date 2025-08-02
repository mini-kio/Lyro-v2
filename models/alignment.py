#!/usr/bin/env python3
"""
Audio-Lyrics Alignment System
MERT-v1-330M + mHuBERT-147 기반 오디오-가사 정렬
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModel, AutoProcessor
import warnings

warnings.filterwarnings("ignore")


class AudioLyricsAligner(nn.Module):
    """
    오디오-가사 정렬 시스템
    
    구성요소:
    1. MERT-v1-330M: 음악 오디오 인코딩 (75Hz, 24kHz)
    2. mHuBERT-147: 다국어 음성 인코딩 
    3. Alignment Network: 시간축 정렬
    4. TTS Integration: 기존 TTS 보조 학습 유지
    """
    
    def __init__(
        self,
        mert_model: str = "m-a-p/MERT-v1-330M",
        mhubert_model: str = "utter-project/mHuBERT-147", 
        feature_rate: int = 75,  # Hz
        sample_rate: int = 24000,  # Hz
        alignment_dim: int = 512,
        max_audio_length: float = 30.0,  # seconds
        device: str = "auto"
    ):
        super().__init__()
        
        self.feature_rate = feature_rate
        self.sample_rate = sample_rate
        self.alignment_dim = alignment_dim
        self.max_audio_length = max_audio_length
        self.max_audio_frames = int(max_audio_length * feature_rate)  # 30s * 75Hz = 2250 frames
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 1. MERT-v1-330M 로드 (음악 특화)
        print("Loading MERT-v1-330M for music understanding...")
        try:
            self.mert_processor = AutoProcessor.from_pretrained(mert_model)
            self.mert_model = AutoModel.from_pretrained(mert_model)
            self.mert_model.eval()
            
            # MERT 파라미터 frozen
            for param in self.mert_model.parameters():
                param.requires_grad = False
            
            # MERT 출력 차원 확인
            self.mert_dim = self.mert_model.config.hidden_size  # 1024
            self.mert_available = True
            
        except Exception as e:
            print(f"Warning: Failed to load MERT: {e}")
            print("Using fallback audio encoder")
            self.mert_available = False
            self.mert_dim = 768
            self._create_fallback_audio_encoder()
        
        # 2. mHuBERT-147 로드 (다국어 음성)
        print("Loading mHuBERT-147 for multilingual speech...")
        try:
            self.mhubert_processor = AutoProcessor.from_pretrained(mhubert_model)
            self.mhubert_model = AutoModel.from_pretrained(mhubert_model)
            self.mhubert_model.eval()
            
            # mHuBERT 파라미터 frozen
            for param in self.mhubert_model.parameters():
                param.requires_grad = False
            
            # mHuBERT 출력 차원 확인  
            self.mhubert_dim = self.mhubert_model.config.hidden_size  # 768
            self.mhubert_available = True
            
        except Exception as e:
            print(f"Warning: Failed to load mHuBERT: {e}")
            print("Using fallback speech encoder")
            self.mhubert_available = False
            self.mhubert_dim = 768
            self._create_fallback_speech_encoder()
        
        # 3. 정렬 네트워크 구성
        self._build_alignment_network()
        
        # 4. TTS 통합 레이어
        self._build_tts_integration()
        
        print(f"AudioLyricsAligner initialized:")
        print(f"  - MERT available: {self.mert_available}")
        print(f"  - mHuBERT available: {self.mhubert_available}")
        print(f"  - Feature rate: {self.feature_rate}Hz")
        print(f"  - Sample rate: {self.sample_rate}Hz")
        print(f"  - Max audio length: {self.max_audio_length}s")
    
    def _create_fallback_audio_encoder(self):
        """MERT 로드 실패시 fallback 오디오 인코더"""
        self.fallback_audio_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=25, stride=5),  # 24kHz -> ~5kHz
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=25, stride=5),  # ~1kHz
            nn.ReLU(), 
            nn.Conv1d(128, 256, kernel_size=25, stride=5),  # ~200Hz
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=25, stride=3),  # ~75Hz
            nn.ReLU(),
            nn.Conv1d(512, self.mert_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(self.max_audio_frames)
        )
    
    def _create_fallback_speech_encoder(self):
        """mHuBERT 로드 실패시 fallback 음성 인코더"""
        self.fallback_speech_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=25, stride=5),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=25, stride=5), 
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=25, stride=5),
            nn.ReLU(),
            nn.Conv1d(256, self.mhubert_dim, kernel_size=25, stride=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(self.max_audio_frames)
        )
    
    def _build_alignment_network(self):
        """정렬 네트워크 구축"""
        
        # 오디오 특징 프로젝션
        self.audio_proj = nn.Sequential(
            nn.Linear(self.mert_dim, self.alignment_dim),
            nn.LayerNorm(self.alignment_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 음성 특징 프로젝션  
        self.speech_proj = nn.Sequential(
            nn.Linear(self.mhubert_dim, self.alignment_dim),
            nn.LayerNorm(self.alignment_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Simple concatenation instead of cross-attention
        
        # 시간축 정렬 네트워크
        self.alignment_lstm = nn.LSTM(
            input_size=self.alignment_dim * 2,  # audio + speech
            hidden_size=self.alignment_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        # Removed alignment scorer and segment classifier
        # These will be handled by REPA Loss during training
    
    def _build_tts_integration(self):
        """Simplified TTS integration - basic feature projection only"""
        
        # Simple feature projection for TTS compatibility
        self.tts_feature_proj = nn.Linear(self.alignment_dim * 2, 256)
    
    def encode_audio_mert(self, audio: torch.Tensor) -> torch.Tensor:
        """MERT로 오디오 인코딩"""
        if not self.mert_available:
            # Fallback encoder
            if audio.dim() == 2:
                audio = audio.mean(dim=0, keepdim=True)  # 모노로 변환
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            audio = audio.unsqueeze(0) if audio.dim() == 2 else audio  # (B, 1, T)
            features = self.fallback_audio_encoder(audio)  # (B, mert_dim, T)
            return features.transpose(-1, -2)  # (B, T, mert_dim)
        
        # MERT 처리
        batch_size = audio.shape[0]
        audio_length = audio.shape[-1]
        
        # 24kHz로 리샘플링 (필요시)
        current_sample_rate = audio_length / (audio_length / self.sample_rate) if audio_length > 0 else self.sample_rate
        if abs(current_sample_rate - self.sample_rate) > 1:  # 1Hz 허용 오차
            original_freq = int(current_sample_rate)
            if original_freq > 0:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=original_freq,
                    new_freq=self.sample_rate
                )
                audio = resampler(audio)
        
        with torch.no_grad():
            # MERT는 배치 처리가 어려울 수 있으므로 개별 처리
            features_list = []
            for i in range(batch_size):
                audio_i = audio[i].cpu().numpy()
                
                # MERT 전처리
                inputs = self.mert_processor(
                    audio_i,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt"
                ).to(self.device)
                
                # MERT 인코딩
                outputs = self.mert_model(**inputs)
                features = outputs.last_hidden_state  # (1, T, mert_dim)
                
                # 75Hz로 맞춤 (interpolation)
                target_frames = min(int(len(audio_i) / self.sample_rate * self.feature_rate), 
                                   self.max_audio_frames)
                
                if features.shape[1] != target_frames:
                    features = F.interpolate(
                        features.transpose(1, 2),  # (1, mert_dim, T)
                        size=target_frames,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)  # (1, T, mert_dim)
                
                features_list.append(features)
            
            return torch.cat(features_list, dim=0)  # (B, T, mert_dim)
    
    def encode_speech_mhubert(self, audio: torch.Tensor) -> torch.Tensor:
        """mHuBERT로 음성 인코딩"""
        if not self.mhubert_available:
            # Fallback encoder
            if audio.dim() == 2:
                audio = audio.mean(dim=0, keepdim=True)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
                
            audio = audio.unsqueeze(0) if audio.dim() == 2 else audio
            features = self.fallback_speech_encoder(audio)
            return features.transpose(-1, -2)
        
        # mHuBERT 처리
        batch_size = audio.shape[0]
        
        with torch.no_grad():
            features_list = []
            for i in range(batch_size):
                audio_i = audio[i].cpu().numpy()
                
                # mHuBERT 전처리 (16kHz 기본)
                inputs = self.mhubert_processor(
                    audio_i,
                    sampling_rate=16000,  # mHuBERT는 보통 16kHz
                    return_tensors="pt"
                ).to(self.device)
                
                # mHuBERT 인코딩
                outputs = self.mhubert_model(**inputs)
                features = outputs.last_hidden_state
                
                # 75Hz로 맞춤
                target_frames = min(int(len(audio_i) / self.sample_rate * self.feature_rate),
                                   self.max_audio_frames)
                
                if features.shape[1] != target_frames:
                    features = F.interpolate(
                        features.transpose(1, 2),
                        size=target_frames,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)
                
                features_list.append(features)
            
            return torch.cat(features_list, dim=0)
    
    def forward(
        self, 
        audio: torch.Tensor,
        lyrics_timestamps: Optional[List[Tuple[float, float, str]]] = None,
        return_alignment: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        오디오-가사 정렬 수행
        
        Args:
            audio: (B, T) 오디오 웨이브폼
            lyrics_timestamps: [(start, end, text), ...] 가사 타임스탬프 (선택적)
            return_alignment: 정렬 결과 반환 여부
            
        Returns:
            Dict containing alignment results and features
        """
        batch_size = audio.shape[0]
        
        # 1. 멀티모달 인코딩
        audio_features = self.encode_audio_mert(audio)     # (B, T, mert_dim)
        speech_features = self.encode_speech_mhubert(audio) # (B, T, mhubert_dim)
        
        # 2. 특징 프로젝션
        audio_proj = self.audio_proj(audio_features)     # (B, T, alignment_dim)
        speech_proj = self.speech_proj(speech_features)   # (B, T, alignment_dim)
        
        # 3. Simple concatenation fusion
        fused_features = torch.cat([audio_proj, speech_proj], dim=-1)  # (B, T, 2*alignment_dim)
        
        # LSTM으로 시퀀스 모델링
        lstm_out, _ = self.alignment_lstm(fused_features)  # (B, T, 2*alignment_dim)
        
        # 5. Simple TTS feature projection
        tts_features = self.tts_feature_proj(lstm_out)  # (B, T, 256)
        
        results = {
            'audio_features': audio_features,
            'speech_features': speech_features,
            'fused_features': lstm_out,  # For REPA Loss
            'tts_features': tts_features
        }
        
        # 7. 정렬 수행 (요청시) - Simplified
        if return_alignment and lyrics_timestamps is not None:
            # Basic timestamp mapping based on feature frames
            results['timestamps'] = lyrics_timestamps
        
        return results
    
    def get_feature_frames_for_timestamp(self, timestamp: float) -> int:
        """Convert timestamp to feature frame index"""
        return min(int(timestamp * self.feature_rate), self.max_audio_frames - 1)
    
    def get_tts_features(
        self,
        alignment_results: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get TTS-compatible features from alignment results"""
        
        tts_features = alignment_results['tts_features']  # (B, T, 256)
        
        # Simple temporal pooling
        return tts_features.mean(dim=1)  # (B, 256)




if __name__ == "__main__":
    # 테스트 코드
    print("Testing AudioLyricsAligner...")
    
    # 모델 초기화
    aligner = AudioLyricsAligner()
    
    # 더미 데이터
    batch_size = 2
    audio_length = 24000 * 10  # 10초
    audio = torch.randn(batch_size, audio_length)
    
    lyrics_timestamps = [
        [(0.0, 2.0, "첫 번째 가사"),
         (2.5, 5.0, "두 번째 가사"),
         (6.0, 8.0, "세 번째 가사")]
    ] * batch_size
    
    # 정렬 수행
    with torch.no_grad():
        results = aligner(audio, lyrics_timestamps)
    
    print("Results:")
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    print("AudioLyricsAligner test completed!")