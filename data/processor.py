# lyro/data/processor.py
"""
LYRO 데이터 처리 - Latent Vector 기반 (수정됨)
Tokenizer + Collator for latent vectors
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class ProcessorConfig:
    """데이터 처리 설정 (수정됨 - Latent 기반)"""
    vocab_size: int = 32000
    max_lyrics_length: int = 512
    max_caption_length: int = 256
    latent_channels: int = 16
    latent_time_steps: int = 128
    latent_duration: float = 10.0
    pad_to_multiple: int = 8
    ensure_channels: bool = True


class LyroTokenizer:
    """LYRO 토크나이저 (기존과 동일)"""
    
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
    """LYRO 데이터 콜레이터 (수정됨 - Latent Vector 기반)"""
    
    def __init__(self, tokenizer: LyroTokenizer = None, config: ProcessorConfig = None, **kwargs):
        # 기존 매개변수 지원
        self.tokenizer = tokenizer or LyroTokenizer()
        self.config = config or ProcessorConfig()
        
        # kwargs에서 직접 설정 가져오기 (하위호환성)
        self.latent_channels = kwargs.get('latent_channels', self.config.latent_channels)
        self.latent_time_steps = kwargs.get('latent_time_steps', self.config.latent_time_steps)
        self.max_text_length = kwargs.get('max_text_length', self.config.max_lyrics_length)
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """배치 처리 (수정됨 - Latent Vector 기반)"""
        batch_size = len(batch)
        
        collated = {
            'latents': None,
            'latent_lengths': torch.zeros(batch_size, dtype=torch.long),
            
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
            
            'reference_latents': None,
            'reference_lengths': torch.zeros(batch_size, dtype=torch.long),
        }
        
        # Latent vector 처리
        collated.update(self._collate_latents(batch))
        
        # 텍스트 처리
        collated.update(self._collate_text(batch))
        
        # 메타데이터 처리
        collated.update(self._collate_metadata(batch))
        
        # 참조 latent 처리
        collated.update(self._collate_reference_latents(batch))
        
        return collated
    
    def _collate_latents(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Latent vector 배치 처리"""
        latent_list = []
        lengths = []
        
        for item in batch:
            latents = item.get('latents')
            if latents is None:
                latents = torch.zeros(self.latent_channels, self.latent_time_steps)
                length = 0
            else:
                if isinstance(latents, np.ndarray):
                    latents = torch.from_numpy(latents).float()
                
                # 채널 수 보장
                if self.config.ensure_channels:
                    latents = self._ensure_channels(latents)
                
                # 길이 조정
                length = latents.shape[-1]
                latents = self._adjust_latent_length(latents)
            
            latent_list.append(latents)
            lengths.append(length)
        
        return {
            'latents': torch.stack(latent_list, dim=0),
            'latent_lengths': torch.tensor(lengths, dtype=torch.long)
        }
    
    def _collate_text(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """텍스트 배치 처리"""
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
    
    def _collate_reference_latents(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """참조 latent 배치 처리"""
        ref_latent_list = []
        ref_lengths = []
        
        for item in batch:
            ref_latents = item.get('reference_latents')
            if ref_latents is None:
                ref_latents = torch.zeros(self.latent_channels, self.latent_time_steps)
                length = 0
            else:
                if isinstance(ref_latents, np.ndarray):
                    ref_latents = torch.from_numpy(ref_latents).float()
                
                if self.config.ensure_channels:
                    ref_latents = self._ensure_channels(ref_latents)
                
                length = ref_latents.shape[-1]
                ref_latents = self._adjust_latent_length(ref_latents)
            
            ref_latent_list.append(ref_latents)
            ref_lengths.append(length)
        
        return {
            'reference_latents': torch.stack(ref_latent_list, dim=0),
            'reference_lengths': torch.tensor(ref_lengths, dtype=torch.long),
        }
    
    def _ensure_channels(self, latents: torch.Tensor) -> torch.Tensor:
        """채널 수 보장"""
        # 차원 정리
        if latents.dim() == 1:
            # (T,) -> (1, T)
            latents = latents.unsqueeze(0)
        elif latents.dim() == 3 and latents.shape[0] == 1:
            # (1, C, T) -> (C, T)
            latents = latents.squeeze(0)
        elif latents.dim() != 2:
            raise ValueError(f"Unexpected latent tensor dimensions: {latents.shape}")
        
        current_channels = latents.shape[0]
        
        if current_channels < self.latent_channels:
            # 채널 패딩
            pad_channels = self.latent_channels - current_channels
            latents = torch.nn.functional.pad(latents, (0, 0, 0, pad_channels))
        elif current_channels > self.latent_channels:
            # 채널 크롭
            latents = latents[:self.latent_channels, :]
        
        return latents
    
    def _adjust_latent_length(self, latents: torch.Tensor) -> torch.Tensor:
        """Latent 길이 조정"""
        current_length = latents.shape[-1]
        
        if current_length > self.latent_time_steps:
            # 랜덤 크롭
            start_idx = torch.randint(0, current_length - self.latent_time_steps + 1, (1,)).item()
            latents = latents[..., start_idx:start_idx + self.latent_time_steps]
        elif current_length < self.latent_time_steps:
            # 제로 패딩
            pad_length = self.latent_time_steps - current_length
            latents = F.pad(latents, (0, pad_length))
        
        # pad_to_multiple 적용
        if self.config.pad_to_multiple > 1:
            current_length = latents.shape[-1]
            pad_to = ((current_length + self.config.pad_to_multiple - 1) 
                     // self.config.pad_to_multiple) * self.config.pad_to_multiple
            if pad_to > current_length:
                latents = F.pad(latents, (0, pad_to - current_length))
        
        return latents


class DataProcessor:
    """통합 데이터 처리기 (수정됨 - Latent 기반)"""
    
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
    
    def process_latents_only(self, latents: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Latent만 처리"""
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).float()
        
        # 채널 수 보장
        if self.config.ensure_channels:
            latents = self.collator._ensure_channels(latents)
        
        # 길이 조정
        latents = self.collator._adjust_latent_length(latents)
        
        return latents
    
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
        reference_latents: Optional[torch.Tensor] = None,
        genre: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """생성용 입력 생성 (수정됨 - Latent 기반)"""
        
        generation_input = {
            'task_type': task,
            'lyrics': None,
            'lyrics_mask': None,
            'captions': None,
            'reference_latents': None
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
        
        # 참조 latent 처리
        if reference_latents is not None:
            processed_ref = self.process_latents_only(reference_latents)
            generation_input['reference_latents'] = processed_ref.unsqueeze(0)
        
        return generation_input
    
    def augment_latents(self, latents: torch.Tensor, augmentation_prob: float = 0.5) -> torch.Tensor:
        """Latent vector 증강"""
        import random
        
        if random.random() > augmentation_prob:
            return latents
        
        augmented = latents.clone()
        
        # Random scaling
        if random.random() < 0.3:
            scale_factor = random.uniform(0.9, 1.1)
            augmented = augmented * scale_factor
        
        # Random noise
        if random.random() < 0.2:
            noise_level = random.uniform(0.005, 0.02)
            noise = torch.randn_like(augmented) * noise_level
            augmented = augmented + noise
        
        # Temporal shift
        if random.random() < 0.1:
            shift_amount = random.randint(-5, 5)
            if shift_amount != 0:
                augmented = torch.roll(augmented, shift_amount, dims=-1)
        
        return augmented
    
    def validate_latent_format(self, latents: torch.Tensor) -> bool:
        """Latent 형식 검증"""
        # 차원 확인
        if latents.dim() not in [2, 3]:
            return False
        
        # 채널 수 확인
        if latents.dim() == 2:
            channels = latents.shape[0]
        else:
            channels = latents.shape[1]
        
        if channels != self.config.latent_channels:
            return False
        
        # 시간 스텝 확인
        time_steps = latents.shape[-1]
        if time_steps != self.config.latent_time_steps:
            return False
        
        # 값 범위 확인 (일반적으로 latent는 -5 ~ 5 범위)
        if torch.abs(latents).max() > 10:
            return False
        
        return True


def create_data_processor(config: ProcessorConfig = None) -> DataProcessor:
    """데이터 처리기 생성"""
    return DataProcessor(config)


def create_tokenizer(vocab_size: int = 32000, max_length: int = 512) -> LyroTokenizer:
    """토크나이저 생성"""
    return LyroTokenizer(vocab_size, max_length)


def create_collator(tokenizer: LyroTokenizer = None, config: ProcessorConfig = None, **kwargs) -> LyroCollator:
    """콜레이터 생성 (하위호환성 지원)"""
    return LyroCollator(tokenizer, config, **kwargs)