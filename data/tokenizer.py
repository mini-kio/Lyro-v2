# lyro/data/tokenizer.py
"""
LYRO Tokenizer - 간소화된 버전
가사와 캡션 처리에 집중
"""

import torch
import numpy as np
from typing import List, Dict, Union, Optional
import sentencepiece as spm
from pathlib import Path
import json


class LyroTokenizer:
    """
    LYRO 토크나이저 - 간소화된 버전
    가사와 MusicCaps 스타일 캡션 처리
    """
    
    # 특수 토큰 정의
    SPECIAL_TOKENS = {
        # 기본 토큰
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3,
        
        # 구조 토큰
        '<TASK=SONG>': 4,
        '<TASK=INST>': 5,
        '<TASK=COVER>': 6,
        
        # 가사 구조
        '<verse>': 7,
        '<chorus>': 8,
        '<bridge>': 9,
        '<outro>': 10,
        
        # 캡션 토큰
        '<genre>': 11,
        '<tempo>': 12,
        '<mood>': 13,
        '<instrument>': 14,
    }
    
    VOCAB_SIZE = 32000
    
    def __init__(
        self,
        text_tokenizer_path: Optional[str] = None,
        vocab_size: int = 32000,
        max_length: int = 512
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # SentencePiece 토크나이저 로드 (선택적)
        if text_tokenizer_path and Path(text_tokenizer_path).exists():
            self.text_tokenizer = spm.SentencePieceProcessor()
            self.text_tokenizer.load(text_tokenizer_path)
        else:
            self.text_tokenizer = None
            print("Warning: Using character-level tokenization fallback")
        
        # 역방향 매핑
        self.id_to_token = {v: k for k, v in self.SPECIAL_TOKENS.items()}
    
    def encode_lyrics(self, lyrics: str) -> List[int]:
        """가사 인코딩"""
        if not lyrics or lyrics.strip() == '':
            return [self.SPECIAL_TOKENS['<PAD>']]
        
        # 가사 구조 처리
        processed_lyrics = self._process_lyric_structure(lyrics)
        
        # 토큰화
        if self.text_tokenizer:
            tokens = self.text_tokenizer.encode(processed_lyrics)
        else:
            # 문자 단위 토크나이저 (fallback)
            tokens = [min(ord(c) % 1000, 999) + 100 for c in processed_lyrics]
        
        # 길이 제한
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
        
        # BOS/EOS 추가
        return [self.SPECIAL_TOKENS['<BOS>']] + tokens + [self.SPECIAL_TOKENS['<EOS>']]
    
    def encode_caption(self, caption: str) -> List[int]:
        """MusicCaps 스타일 캡션 인코딩"""
        if not caption or caption.strip() == '':
            return [self.SPECIAL_TOKENS['<PAD>']]
        
        # 캡션 전처리
        processed_caption = self._process_caption(caption)
        
        # 토큰화
        if self.text_tokenizer:
            tokens = self.text_tokenizer.encode(processed_caption)
        else:
            tokens = [min(ord(c) % 1000, 999) + 100 for c in processed_caption]
        
        # 길이 제한
        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
        
        return [self.SPECIAL_TOKENS['<BOS>']] + tokens + [self.SPECIAL_TOKENS['<EOS>']]
    
    def decode(self, token_ids: List[int]) -> str:
        """토큰 디코딩"""
        if not token_ids:
            return ""
        
        # 특수 토큰 필터링
        filtered_tokens = []
        for token_id in token_ids:
            if token_id not in self.id_to_token:
                filtered_tokens.append(token_id)
        
        if not filtered_tokens:
            return ""
        
        # 디코딩
        if self.text_tokenizer:
            try:
                return self.text_tokenizer.decode(filtered_tokens)
            except:
                pass
        
        # 문자 단위 디코딩 (fallback)
        try:
            decoded = ''.join(chr((token - 100) % 256) for token in filtered_tokens)
            return decoded
        except:
            return ""
    
    def _process_lyric_structure(self, lyrics: str) -> str:
        """가사 구조 태그 처리"""
        import re
        
        # 구조 태그 매핑
        structure_patterns = [
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
        for pattern, replacement in structure_patterns:
            processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
        
        return processed
    
    def _process_caption(self, caption: str) -> str:
        """MusicCaps 스타일 캡션 처리"""
        import re
        
        # 장르, 템포, 악기 등 키워드 강조
        genre_keywords = ['rock', 'pop', 'jazz', 'classical', 'electronic', 'hip hop', 'country', 'blues']
        tempo_keywords = ['fast', 'slow', 'moderate', 'upbeat', 'ballad']
        mood_keywords = ['happy', 'sad', 'energetic', 'calm', 'dramatic', 'romantic']
        instrument_keywords = ['guitar', 'piano', 'drums', 'violin', 'saxophone', 'synthesizer']
        
        processed = caption.lower()
        
        # 키워드 태깅
        for genre in genre_keywords:
            if genre in processed:
                processed = processed.replace(genre, f'<genre>{genre}')
        
        for tempo in tempo_keywords:
            if tempo in processed:
                processed = processed.replace(tempo, f'<tempo>{tempo}')
        
        for mood in mood_keywords:
            if mood in processed:
                processed = processed.replace(mood, f'<mood>{mood}')
        
        for instrument in instrument_keywords:
            if instrument in processed:
                processed = processed.replace(instrument, f'<instrument>{instrument}')
        
        return processed
    
    def batch_encode(
        self,
        texts: List[str],
        text_type: str = 'lyrics',
        padding: bool = True,
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """배치 인코딩"""
        max_length = max_length or self.max_length
        
        if text_type == 'lyrics':
            encoded = [self.encode_lyrics(text) for text in texts]
        elif text_type == 'caption':
            encoded = [self.encode_caption(text) for text in texts]
        else:
            raise ValueError(f"Unknown text_type: {text_type}")
        
        if padding:
            # 패딩
            padded = []
            for tokens in encoded:
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                else:
                    tokens = tokens + [self.SPECIAL_TOKENS['<PAD>']] * (max_length - len(tokens))
                padded.append(tokens)
            
            return torch.tensor(padded, dtype=torch.long)
        else:
            return encoded
    
    def create_attention_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """어텐션 마스크 생성"""
        return (token_ids != self.SPECIAL_TOKENS['<PAD>']).bool()
    
    def get_vocab_size(self) -> int:
        """어휘 크기 반환"""
        return self.vocab_size