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
        
        # Cross-modal attention (오디오 ↔ 음성)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.alignment_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 시간축 정렬 네트워크
        self.alignment_lstm = nn.LSTM(
            input_size=self.alignment_dim * 2,  # audio + speech
            hidden_size=self.alignment_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        
        # 정렬 점수 계산
        self.alignment_scorer = nn.Sequential(
            nn.Linear(self.alignment_dim * 2, self.alignment_dim),
            nn.ReLU(),
            nn.Linear(self.alignment_dim, 1),
            nn.Sigmoid()
        )
        
        # 세그먼트 분류기 (가사 구간 vs 악기 구간)
        self.segment_classifier = nn.Sequential(
            nn.Linear(self.alignment_dim * 2, self.alignment_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.alignment_dim, 3),  # [lyrics, instrumental, silence]
            nn.Softmax(dim=-1)
        )
    
    def _build_tts_integration(self):
        """TTS 통합 레이어 (기존 TTS 보조 학습 유지)"""
        
        # TTS 조건 생성 (정렬된 가사를 TTS 입력으로)
        self.tts_condition_generator = nn.Sequential(
            nn.Linear(self.alignment_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),  # TTS 조건 차원
            nn.LayerNorm(256)
        )
        
        # TTS 손실 가중치 (정렬 품질에 따라 동적 조정)
        self.tts_weight_predictor = nn.Sequential(
            nn.Linear(self.alignment_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
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
        if audio_length / audio.shape[-2] != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=int(audio_length / audio.shape[-2]),
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
        
        # 3. Cross-modal attention
        # 오디오가 음성을 참고
        audio_enhanced, audio_attn = self.cross_attention(
            query=audio_proj,
            key=speech_proj, 
            value=speech_proj
        )
        
        # 4. 융합 및 정렬
        fused_features = torch.cat([audio_enhanced, speech_proj], dim=-1)  # (B, T, 2*alignment_dim)
        
        # LSTM으로 시퀀스 모델링
        lstm_out, _ = self.alignment_lstm(fused_features)  # (B, T, 2*alignment_dim)
        
        # 5. 정렬 점수 및 세그먼트 분류
        alignment_scores = self.alignment_scorer(lstm_out).squeeze(-1)  # (B, T)
        segment_probs = self.segment_classifier(lstm_out)  # (B, T, 3)
        
        # 6. TTS 통합
        tts_conditions = self.tts_condition_generator(lstm_out)  # (B, T, 256)
        tts_weights = self.tts_weight_predictor(audio_enhanced).squeeze(-1)  # (B, T)
        
        results = {
            'audio_features': audio_features,
            'speech_features': speech_features,
            'alignment_scores': alignment_scores,
            'segment_probs': segment_probs,  # [lyrics, instrumental, silence]
            'tts_conditions': tts_conditions,
            'tts_weights': tts_weights,
            'cross_attention': audio_attn
        }
        
        # 7. 정렬 수행 (요청시)
        if return_alignment and lyrics_timestamps is not None:
            alignments = self.align_lyrics_to_audio(
                alignment_scores, segment_probs, lyrics_timestamps
            )
            results['alignments'] = alignments
        
        return results
    
    def align_lyrics_to_audio(
        self,
        alignment_scores: torch.Tensor,
        segment_probs: torch.Tensor, 
        lyrics_timestamps: List[Tuple[float, float, str]]
    ) -> List[Dict]:
        """가사를 오디오에 정렬"""
        
        batch_size = alignment_scores.shape[0]
        results = []
        
        for b in range(batch_size):
            scores = alignment_scores[b].cpu().numpy()
            segments = segment_probs[b].cpu().numpy()
            
            # 가사 구간 탐지 (lyrics probability > 0.5)
            lyrics_mask = segments[:, 0] > 0.5  # lyrics class
            
            # 정렬 점수가 높은 구간 찾기
            high_align_mask = scores > np.percentile(scores, 75)
            
            # 두 조건을 모두 만족하는 구간
            candidate_mask = lyrics_mask & high_align_mask
            
            # 연속 구간 찾기
            aligned_segments = []
            in_segment = False
            start_frame = 0
            
            for frame in range(len(candidate_mask)):
                if candidate_mask[frame] and not in_segment:
                    # 새 세그먼트 시작
                    start_frame = frame
                    in_segment = True
                elif not candidate_mask[frame] and in_segment:
                    # 세그먼트 종료
                    end_frame = frame
                    
                    start_time = start_frame / self.feature_rate
                    end_time = end_frame / self.feature_rate
                    confidence = scores[start_frame:end_frame].mean()
                    
                    aligned_segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'confidence': confidence
                    })
                    
                    in_segment = False
            
            # 마지막 세그먼트 처리
            if in_segment:
                end_frame = len(candidate_mask)
                start_time = start_frame / self.feature_rate
                end_time = end_frame / self.feature_rate
                confidence = scores[start_frame:end_frame].mean()
                
                aligned_segments.append({
                    'start_time': start_time,
                    'end_time': end_time, 
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'confidence': confidence
                })
            
            results.append({
                'segments': aligned_segments,
                'lyrics_mask': lyrics_mask,
                'alignment_scores': scores
            })
        
        return results
    
    def get_tts_training_data(
        self,
        alignment_results: Dict[str, torch.Tensor],
        lyrics_text: List[str]
    ) -> Dict[str, torch.Tensor]:
        """TTS 훈련을 위한 데이터 준비"""
        
        tts_conditions = alignment_results['tts_conditions']  # (B, T, 256)
        tts_weights = alignment_results['tts_weights']        # (B, T)
        segment_probs = alignment_results['segment_probs']    # (B, T, 3)
        
        # 가사 구간만 추출
        lyrics_mask = segment_probs[:, :, 0] > 0.5  # lyrics class
        
        # 가중 평균으로 TTS 조건 생성
        weighted_conditions = []
        for b in range(tts_conditions.shape[0]):
            mask = lyrics_mask[b]
            if mask.any():
                weights = tts_weights[b][mask]
                conditions = tts_conditions[b][mask]
                
                # 가중 평균
                weighted_condition = (conditions * weights.unsqueeze(-1)).sum(dim=0) / weights.sum()
            else:
                # 가사 구간이 없으면 전체 평균
                weighted_condition = tts_conditions[b].mean(dim=0)
            
            weighted_conditions.append(weighted_condition)
        
        return {
            'tts_conditions': torch.stack(weighted_conditions),  # (B, 256)
            'tts_weights': tts_weights,
            'lyrics_masks': lyrics_mask,
            'lyrics_text': lyrics_text
        }




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
    
    print("✅ AudioLyricsAligner test completed!")