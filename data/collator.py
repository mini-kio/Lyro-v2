# lyro/data/collator.py
"""
LYRO 데이터 콜레이터 - 배치 처리 및 패딩
통합된 배치 처리로 오디오, 텍스트, 메타데이터 관리
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class LyroCollator:
    """
    LYRO 통합 콜레이터
    오디오, 가사, 캡션, 메타데이터를 일관된 배치로 처리
    """
    
    max_audio_length: int = 441000  # 10초 @ 44.1kHz
    max_text_length: int = 512
    sample_rate: int = 44100
    pad_to_multiple: int = 256
    ensure_stereo: bool = True
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """배치 처리 메인 함수"""
        
        # 배치 크기
        batch_size = len(batch)
        
        # 출력 딕셔너리 초기화
        collated = {
            'audio': None,
            'audio_lengths': torch.zeros(batch_size, dtype=torch.long),
            
            # 텍스트 관련
            'lyrics': None,
            'lyrics_mask': None,
            'lyrics_lengths': torch.zeros(batch_size, dtype=torch.long),
            
            'captions': None,
            'captions_mask': None,
            'captions_lengths': torch.zeros(batch_size, dtype=torch.long),
            
            # 메타데이터
            'task_types': [],
            'genres': [],
            'ids': [],
            
            # 참조 오디오 (COVER 태스크용)
            'reference_audio': None,
            'reference_lengths': torch.zeros(batch_size, dtype=torch.long),
        }
        
        # 오디오 처리
        collated.update(self._collate_audio(batch))
        
        # 텍스트 처리
        collated.update(self._collate_text(batch))
        
        # 메타데이터 처리
        collated.update(self._collate_metadata(batch))
        
        # 참조 오디오 처리
        collated.update(self._collate_reference_audio(batch))
        
        return collated
    
    def _collate_audio(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """오디오 배치 처리"""
        audio_list = []
        lengths = []
        
        for item in batch:
            audio = item.get('audio')
            if audio is None:
                # 더미 오디오 생성
                audio = torch.zeros(2, self.max_audio_length)
                length = 0
            else:
                # 텐서 변환
                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio).float()
                
                # 스테레오 보장
                if self.ensure_stereo:
                    audio = self._ensure_stereo(audio)
                
                # 길이 조정
                length = audio.shape[-1]
                audio = self._adjust_audio_length(audio)
            
            audio_list.append(audio)
            lengths.append(length)
        
        # 스택
        audio_tensor = torch.stack(audio_list, dim=0)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        
        return {
            'audio': audio_tensor,
            'audio_lengths': lengths_tensor
        }
    
    def _collate_text(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """텍스트 배치 처리 (가사 및 캡션)"""
        # 가사 처리
        lyrics_tokens = []
        lyrics_lengths = []
        
        # 캡션 처리
        captions_tokens = []
        captions_lengths = []
        
        for item in batch:
            # 가사
            lyrics = item.get('lyrics_tokens', [])
            if isinstance(lyrics, torch.Tensor):
                lyrics = lyrics.tolist()
            
            lyrics = self._pad_tokens(lyrics, self.max_text_length)
            lyrics_tokens.append(lyrics)
            lyrics_lengths.append(len([t for t in lyrics if t != 0]))  # 패딩 제외
            
            # 캡션
            captions = item.get('caption_tokens', [])
            if isinstance(captions, torch.Tensor):
                captions = captions.tolist()
            
            captions = self._pad_tokens(captions, self.max_text_length)
            captions_tokens.append(captions)
            captions_lengths.append(len([t for t in captions if t != 0]))
        
        # 텐서 변환
        lyrics_tensor = torch.tensor(lyrics_tokens, dtype=torch.long)
        captions_tensor = torch.tensor(captions_tokens, dtype=torch.long)
        
        # 마스크 생성
        lyrics_mask = self._create_attention_mask(lyrics_tensor)
        captions_mask = self._create_attention_mask(captions_tensor)
        
        return {
            'lyrics': lyrics_tensor,
            'lyrics_mask': lyrics_mask,
            'lyrics_lengths': torch.tensor(lyrics_lengths, dtype=torch.long),
            
            'captions': captions_tensor,
            'captions_mask': captions_mask,
            'captions_lengths': torch.tensor(captions_lengths, dtype=torch.long),
        }
    
    def _collate_metadata(self, batch: List[Dict]) -> Dict[str, Any]:
        """메타데이터 배치 처리"""
        task_types = []
        genres = []
        ids = []
        
        for item in batch:
            task_types.append(item.get('task', 'SONG'))
            genres.append(item.get('genre', ['unknown']))
            ids.append(item.get('id', 'unknown'))
        
        return {
            'task_types': task_types,
            'genres': genres,
            'ids': ids,
        }
    
    def _collate_reference_audio(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """참조 오디오 배치 처리 (COVER 태스크용)"""
        ref_audio_list = []
        ref_lengths = []
        
        for item in batch:
            ref_audio = item.get('reference_audio')
            if ref_audio is None:
                # 더미 참조 오디오
                ref_audio = torch.zeros(2, self.max_audio_length)
                length = 0
            else:
                if isinstance(ref_audio, np.ndarray):
                    ref_audio = torch.from_numpy(ref_audio).float()
                
                if self.ensure_stereo:
                    ref_audio = self._ensure_stereo(ref_audio)
                
                length = ref_audio.shape[-1]
                ref_audio = self._adjust_audio_length(ref_audio)
            
            ref_audio_list.append(ref_audio)
            ref_lengths.append(length)
        
        ref_audio_tensor = torch.stack(ref_audio_list, dim=0)
        ref_lengths_tensor = torch.tensor(ref_lengths, dtype=torch.long)
        
        return {
            'reference_audio': ref_audio_tensor,
            'reference_lengths': ref_lengths_tensor,
        }
    
    def _ensure_stereo(self, audio: torch.Tensor) -> torch.Tensor:
        """스테레오 포맷 보장"""
        if audio.dim() == 1:
            # 모노 -> 스테레오
            audio = audio.unsqueeze(0).repeat(2, 1)
        elif audio.dim() == 2:
            if audio.shape[0] == 1:
                # 단일 채널 -> 스테레오
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                # 다중 채널 -> 스테레오
                audio = audio[:2, :]
        else:
            raise ValueError(f"Unexpected audio dimensions: {audio.shape}")
        
        return audio
    
    def _adjust_audio_length(self, audio: torch.Tensor) -> torch.Tensor:
        """오디오 길이 조정 (패딩 또는 크롭)"""
        current_length = audio.shape[-1]
        
        if current_length > self.max_audio_length:
            # 랜덤 크롭
            start_idx = torch.randint(0, current_length - self.max_audio_length + 1, (1,)).item()
            audio = audio[..., start_idx:start_idx + self.max_audio_length]
        elif current_length < self.max_audio_length:
            # 제로 패딩
            pad_length = self.max_audio_length - current_length
            audio = F.pad(audio, (0, pad_length))
        
        # pad_to_multiple 적용
        if self.pad_to_multiple > 1:
            current_length = audio.shape[-1]
            pad_to = ((current_length + self.pad_to_multiple - 1) // self.pad_to_multiple) * self.pad_to_multiple
            if pad_to > current_length:
                audio = F.pad(audio, (0, pad_to - current_length))
        
        return audio
    
    def _pad_tokens(self, tokens: List[int], max_length: int, pad_token: int = 0) -> List[int]:
        """토큰 패딩"""
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        elif len(tokens) < max_length:
            tokens = tokens + [pad_token] * (max_length - len(tokens))
        
        return tokens
    
    def _create_attention_mask(self, tokens: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
        """어텐션 마스크 생성"""
        return (tokens != pad_token).bool()


class LyroDCAECollator:
    """
    DCAE 학습용 간소화된 콜레이터
    오디오만 처리
    """
    
    def __init__(
        self,
        max_length: int = 44100,
        min_length: int = 4410,
        target_length: Optional[int] = None,
        pad_to_multiple: int = 256,
        ensure_stereo: bool = True
    ):
        self.max_length = max_length
        self.min_length = min_length
        self.target_length = target_length or max_length
        self.pad_to_multiple = pad_to_multiple
        self.ensure_stereo = ensure_stereo
    
    def __call__(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """DCAE용 오디오 배치 처리"""
        processed_audio = []
        
        for audio in batch:
            # 텐서 변환
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            
            # 스테레오 보장
            if self.ensure_stereo:
                audio = self._ensure_stereo(audio)
            
            # 길이 검증
            if audio.shape[-1] < self.min_length:
                # 너무 짧으면 반복
                repeat_factor = (self.min_length // audio.shape[-1]) + 1
                audio = audio.repeat(1, repeat_factor)
            
            # 길이 조정
            audio = self._adjust_length(audio)
            
            processed_audio.append(audio)
        
        # 배치 스택
        return torch.stack(processed_audio, dim=0)
    
    def _ensure_stereo(self, audio: torch.Tensor) -> torch.Tensor:
        """스테레오 포맷 보장"""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).repeat(2, 1)
        elif audio.dim() == 2:
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                audio = audio[:2, :]
        
        return audio
    
    def _adjust_length(self, audio: torch.Tensor) -> torch.Tensor:
        """길이 조정"""
        current_length = audio.shape[-1]
        
        # 타겟 길이에 맞춤
        if current_length > self.target_length:
            # 중앙 크롭
            start_idx = (current_length - self.target_length) // 2
            audio = audio[..., start_idx:start_idx + self.target_length]
        elif current_length < self.target_length:
            # 제로 패딩
            pad_length = self.target_length - current_length
            audio = F.pad(audio, (0, pad_length))
        
        # pad_to_multiple 적용
        if self.pad_to_multiple > 1:
            current_length = audio.shape[-1]
            pad_to = ((current_length + self.pad_to_multiple - 1) // self.pad_to_multiple) * self.pad_to_multiple
            if pad_to > current_length:
                audio = F.pad(audio, (0, pad_to - current_length))
        
        return audio


class StreamingCollator:
    """
    실시간 생성용 스트리밍 콜레이터
    """
    
    def __init__(self, chunk_size: int = 44100):
        self.chunk_size = chunk_size
        self.buffer = []
    
    def add_chunk(self, audio_chunk: torch.Tensor):
        """오디오 청크 추가"""
        self.buffer.append(audio_chunk)
    
    def get_batch(self, batch_size: int = 1) -> Optional[torch.Tensor]:
        """배치 생성"""
        if len(self.buffer) < batch_size:
            return None
        
        batch = self.buffer[:batch_size]
        self.buffer = self.buffer[batch_size:]
        
        return torch.stack(batch, dim=0)
    
    def clear(self):
        """버퍼 클리어"""
        self.buffer.clear()
