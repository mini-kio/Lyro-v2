# lyro/data/processor.py
"""
LYRO 데이터 처리 - Tokenizer + Collator 통합
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class ProcessorConfig:
    """데이터 처리 설정"""
    vocab_size: int = 32000
    max_lyrics_length: int = 512
    max_caption_length: int = 256
    max_audio_length: int = 441000  # 10초 @ 44.1kHz
    sample_rate: int = 44100
    pad_to_multiple: int = 256
    ensure_stereo: bool = True


class LyroTokenizer:
    """LYRO 토크나이저"""
    
    # 특수 토큰
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3,
        '<TASK=SONG>': 4,
        '<TASK=INST>': 5,
        '<TASK=COVER>': 6,
        '<verse>': 7,
        '<chorus>': 8,
        '<bridge>': 9,
        '<outro>': 10,
        '<genre>': 11,
        '<tempo>': 12,
        '<mood>': 13,
        '<instrument>': 14,
    }
    
    def __init__(self, vocab_size: int = 32000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.id_to_token = {v: k for k, v in self.SPECIAL_TOKENS.items()}
    
    def encode_lyrics(self, lyrics: str) -> List[int]:
        """가사 인코딩"""
        if not lyrics or lyrics.strip() == '':
            return [self.SPECIAL_TOKENS['<PAD>']]
        
        # 가사 구조 처리
        processed = self._process_lyric_structure(lyrics)
        
        # 문자 단위 토크나이저 (간단한 구현)
        tokens = [min(ord(c) % 1000, 999) + 100 for c in processed[:self.max_length-2]]
        
        # BOS/EOS 추가
        return [self.SPECIAL_TOKENS['<BOS>']] + tokens + [self.SPECIAL_TOKENS['<EOS>']]
    
    def encode_caption(self, caption: str) -> List[int]:
        """캡션 인코딩"""
        if not caption or caption.strip() == '':
            return [self.SPECIAL_TOKENS['<PAD>']]
        
        # 캡션 전처리
        processed = self._process_caption(caption)
        
        # 문자 단위 토크나이저
        tokens = [min(ord(c) % 1000, 999) + 100 for c in processed[:self.max_length-2]]
        
        return [self.SPECIAL_TOKENS['<BOS>']] + tokens + [self.SPECIAL_TOKENS['<EOS>']]
    
    def decode(self, token_ids: List[int]) -> str:
        """토큰 디코딩"""
        if not token_ids:
            return ""
        
        # 특수 토큰 필터링
        filtered = [t for t in token_ids if t not in self.id_to_token]
        
        # 문자 변환
        try:
            decoded = ''.join(chr((token - 100) % 256) for token in filtered if token >= 100)
            return decoded
        except:
            return ""
    
    def _process_lyric_structure(self, lyrics: str) -> str:
        """가사 구조 태그 처리"""
        import re
        
        patterns = [
            (r'\(Verse[^)]*\)', '<verse>'),
            (r'\(Chorus[^)]*\)', '<chorus>'),
            (r'\(Bridge[^)]*\)', '<bridge>'),
            (r'\(Outro[^)]*\)', '<outro>'),
            (r'\[Verse[^\]]*\]', '<verse>'),
            (r'\[Chorus[^\]]*\]', '<chorus>'),
            (r'\[Bridge[^\]]*\]', '<bridge>'),
            (r'\[Outro[^\]]*\]', '<outro>'),
        ]
        
        processed = lyrics
        for pattern, replacement in patterns:
            processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
        
        return processed
    
    def _process_caption(self, caption: str) -> str:
        """캡션 처리"""
        # 장르, 템포 등 키워드 강조
        genre_keywords = ['rock', 'pop', 'jazz', 'classical', 'electronic', 'hip hop']
        tempo_keywords = ['fast', 'slow', 'moderate', 'upbeat', 'ballad']
        mood_keywords = ['happy', 'sad', 'energetic', 'calm', 'dramatic']
        
        processed = caption.lower()
        
        for genre in genre_keywords:
            if genre in processed:
                processed = processed.replace(genre, f'<genre>{genre}')
        
        for tempo in tempo_keywords:
            if tempo in processed:
                processed = processed.replace(tempo, f'<tempo>{tempo}')
        
        for mood in mood_keywords:
            if mood in processed:
                processed = processed.replace(mood, f'<mood>{mood}')
        
        return processed
    
    def batch_encode(self, texts: List[str], text_type: str = 'lyrics') -> torch.Tensor:
        """배치 인코딩"""
        if text_type == 'lyrics':
            encoded = [self.encode_lyrics(text) for text in texts]
        elif text_type == 'caption':
            encoded = [self.encode_caption(text) for text in texts]
        else:
            raise ValueError(f"Unknown text_type: {text_type}")
        
        # 패딩
        max_len = min(max(len(tokens) for tokens in encoded), self.max_length)
        padded = []
        
        for tokens in encoded:
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [self.SPECIAL_TOKENS['<PAD>']] * (max_len - len(tokens))
            padded.append(tokens)
        
        return torch.tensor(padded, dtype=torch.long)
    
    def create_attention_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """어텐션 마스크 생성"""
        return (token_ids != self.SPECIAL_TOKENS['<PAD>']).bool()


class LyroCollator:
    """LYRO 데이터 콜레이터 (수정됨)"""
    
    def __init__(self, tokenizer: LyroTokenizer = None, config: ProcessorConfig = None, **kwargs):
        # 기존 매개변수 지원
        self.tokenizer = tokenizer or LyroTokenizer()
        self.config = config or ProcessorConfig()
        
        # kwargs에서 직접 설정 가져오기 (하위호환성)
        self.max_audio_length = kwargs.get('max_audio_length', self.config.max_audio_length)
        self.max_text_length = kwargs.get('max_text_length', self.config.max_lyrics_length)
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """배치 처리 (수정됨)"""
        batch_size = len(batch)
        
        collated = {
            'audio': None,
            'audio_lengths': torch.zeros(batch_size, dtype=torch.long),
            
            'lyrics': None,
            'lyrics_mask': None,
            'lyrics_lengths': torch.zeros(batch_size, dtype=torch.long),
            
            'captions': [],
            'caption_tokens': None,
            'caption_mask': None,
            'caption_lengths': torch.zeros(batch_size, dtype=torch.long),
            
            'task_types': [],
            'genres': [],
            'ids': [],
            
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
                audio = torch.zeros(2, self.max_audio_length)
                length = 0
            else:
                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio).float()
                
                # 스테레오 보장
                if self.config.ensure_stereo:
                    audio = self._ensure_stereo(audio)
                
                # 길이 조정
                length = audio.shape[-1]
                audio = self._adjust_audio_length(audio)
            
            audio_list.append(audio)
            lengths.append(length)
        
        return {
            'audio': torch.stack(audio_list, dim=0),
            'audio_lengths': torch.tensor(lengths, dtype=torch.long)
        }
    
    def _collate_text(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """텍스트 배치 처리 (수정됨)"""
        lyrics_texts = []
        caption_texts = []
        
        for item in batch:
            # 다양한 키 이름 지원
            lyrics = item.get('lyrics_text', item.get('lyrics', ''))
            caption = item.get('caption_text', item.get('caption', ''))
            
            lyrics_texts.append(lyrics if lyrics else '')
            caption_texts.append(caption if caption else '')
        
        # 가사 토크나이징
        lyrics_tokens = self.tokenizer.batch_encode(lyrics_texts, 'lyrics')
        lyrics_mask = self.tokenizer.create_attention_mask(lyrics_tokens)
        lyrics_lengths = torch.sum(lyrics_mask, dim=1)
        
        # 캡션 토크나이징
        caption_tokens = self.tokenizer.batch_encode(caption_texts, 'caption')
        caption_mask = self.tokenizer.create_attention_mask(caption_tokens)
        caption_lengths = torch.sum(caption_mask, dim=1)
        
        return {
            'lyrics': lyrics_tokens,
            'lyrics_mask': lyrics_mask,
            'lyrics_lengths': lyrics_lengths,
            
            'captions': caption_texts,  # 원본 텍스트도 보존
            'caption_tokens': caption_tokens,
            'caption_mask': caption_mask,
            'caption_lengths': caption_lengths,
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
        """참조 오디오 배치 처리"""
        ref_audio_list = []
        ref_lengths = []
        
        for item in batch:
            ref_audio = item.get('reference_audio')
            if ref_audio is None:
                ref_audio = torch.zeros(2, self.max_audio_length)
                length = 0
            else:
                if isinstance(ref_audio, np.ndarray):
                    ref_audio = torch.from_numpy(ref_audio).float()
                
                if self.config.ensure_stereo:
                    ref_audio = self._ensure_stereo(ref_audio)
                
                length = ref_audio.shape[-1]
                ref_audio = self._adjust_audio_length(ref_audio)
            
            ref_audio_list.append(ref_audio)
            ref_lengths.append(length)
        
        return {
            'reference_audio': torch.stack(ref_audio_list, dim=0),
            'reference_lengths': torch.tensor(ref_lengths, dtype=torch.long),
        }
    
    def _ensure_stereo(self, audio: torch.Tensor) -> torch.Tensor:
        """스테레오 포맷 보장"""
        # 차원 정리
        if audio.dim() == 1:
            # (T,) -> (2, T)
            audio = audio.unsqueeze(0).repeat(2, 1)
        elif audio.dim() == 2:
            if audio.shape[0] == 1:
                # (1, T) -> (2, T)
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                # (C, T) where C > 2 -> (2, T)
                audio = audio[:2, :]
            # audio.shape[0] == 2인 경우는 그대로 유지
        elif audio.dim() == 3:
            # (B, C, T) 형태인 경우 batch dimension 제거
            if audio.shape[0] == 1:
                audio = audio.squeeze(0)
                return self._ensure_stereo(audio)  # 재귀 호출
            else:
                raise ValueError(f"Unexpected batch size in audio tensor: {audio.shape}")
        else:
            raise ValueError(f"Unexpected audio tensor dimensions: {audio.shape}")
        
        return audio
    
    def _adjust_audio_length(self, audio: torch.Tensor) -> torch.Tensor:
        """오디오 길이 조정"""
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
        if self.config.pad_to_multiple > 1:
            current_length = audio.shape[-1]
            pad_to = ((current_length + self.config.pad_to_multiple - 1) 
                     // self.config.pad_to_multiple) * self.config.pad_to_multiple
            if pad_to > current_length:
                audio = F.pad(audio, (0, pad_to - current_length))
        
        return audio


class DataProcessor:
    """통합 데이터 처리기"""
    
    def __init__(self, config: ProcessorConfig = None):
        self.config = config or ProcessorConfig()
        self.tokenizer = LyroTokenizer(
            vocab_size=self.config.vocab_size,
            max_length=self.config.max_lyrics_length
        )
        self.collator = LyroCollator(self.tokenizer, self.config)
    
    def process_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """단일 샘플 처리"""
        # 배치 형태로 만들어서 콜레이터 적용
        batch = [sample]
        processed = self.collator(batch)
        
        # 배치 차원 제거
        for key, value in processed.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                processed[key] = value[0]
            elif isinstance(value, list) and len(value) > 0:
                processed[key] = value[0]
        
        return processed
    
    def process_audio_only(self, audio: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """오디오만 처리"""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        
        # 스테레오 보장
        if self.config.ensure_stereo:
            audio = self.collator._ensure_stereo(audio)
        
        # 길이 조정
        audio = self.collator._adjust_audio_length(audio)
        
        return audio
    
    def process_text_only(self, text: str, text_type: str = 'lyrics') -> Dict[str, torch.Tensor]:
        """텍스트만 처리"""
        if text_type == 'lyrics':
            tokens = self.tokenizer.encode_lyrics(text)
        elif text_type == 'caption':
            tokens = self.tokenizer.encode_caption(text)
        else:
            raise ValueError(f"Unknown text_type: {text_type}")
        
        tokens_tensor = torch.tensor([tokens], dtype=torch.long)
        mask = self.tokenizer.create_attention_mask(tokens_tensor)
        
        return {
            'tokens': tokens_tensor[0],
            'mask': mask[0],
            'length': torch.sum(mask[0])
        }
    
    def create_generation_input(
        self,
        task: str = "SONG",
        lyrics: Optional[str] = None,
        caption: Optional[str] = None,
        reference_audio: Optional[torch.Tensor] = None,
        genre: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """생성용 입력 생성"""
        
        generation_input = {
            'task_type': task,
            'lyrics': None,
            'lyrics_mask': None,
            'captions': None,
            'reference_audio': None
        }
        
        # 가사 처리
        if lyrics:
            lyrics_data = self.process_text_only(lyrics, 'lyrics')
            generation_input['lyrics'] = lyrics_data['tokens'].unsqueeze(0)
            generation_input['lyrics_mask'] = lyrics_data['mask'].unsqueeze(0)
        
        # 캡션 처리
        if caption:
            generation_input['captions'] = [caption]
        elif genre:
            # 장르에서 캡션 생성
            genre_str = ", ".join(genre)
            caption = f"This is a {genre_str} music piece."
            generation_input['captions'] = [caption]
        
        # 참조 오디오 처리
        if reference_audio is not None:
            processed_ref = self.process_audio_only(reference_audio)
            generation_input['reference_audio'] = processed_ref.unsqueeze(0)
        
        return generation_input


def create_data_processor(config: ProcessorConfig = None) -> DataProcessor:
    """데이터 처리기 생성"""
    return DataProcessor(config)


def create_tokenizer(vocab_size: int = 32000, max_length: int = 512) -> LyroTokenizer:
    """토크나이저 생성"""
    return LyroTokenizer(vocab_size, max_length)


def create_collator(tokenizer: LyroTokenizer = None, config: ProcessorConfig = None, **kwargs) -> LyroCollator:
    """콜레이터 생성 (하위호환성 지원)"""
    return LyroCollator(tokenizer, config, **kwargs)