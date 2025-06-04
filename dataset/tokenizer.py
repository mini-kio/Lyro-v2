# lyro/dataset/tokenizer.py
"""
LYRO Tokenizer Implementation with TTS Support
특수 토큰 처리 및 가사/전사본/오디오 토큰화
"""

import torch
import numpy as np
from typing import List, Dict, Union, Optional
import sentencepiece as spm
from pathlib import Path
import json


class LyroTokenizer:
    """
    LYRO 토크나이저 with TTS support
    
    텍스트(가사/전사본)와 오디오 토큰을 통합 관리하며,
    특수 토큰(EOS, TASK 등)을 처리합니다.
    """
    
    # 특수 토큰 정의 (TTS 추가)
    SPECIAL_TOKENS = {
        # EOS 토큰들
        '<EOA>': 32000,      # End of Audio (main EOS)
        '<EOD>': 32001,      # End of Document (final EOS)
        
        # 구조 토큰
        '<SOA>': 32002,      # Start of Audio
        '<TASK=SONG>': 32003,
        '<TASK=INST>': 32004,
        '<TASK=COVER>': 32005,
        '<TASK=TTS>': 32006,   # TTS 태스크 추가
        
        # 참조 토큰
        '<REF=10s>': 32007,   # 짧은 참조
        '<REF=30s>': 32008,   # 중간 참조
        '<REF=60s>': 32009,   # 긴 참조
        '<REF=180s>': 32010,  # 매우 긴 참조 (3분)
        '</REF>': 32011,      # 참조 종료
        
        # 편집 토큰
        '<MASK>': 32012,
        '</MASK>': 32013,
        
        # 오디오 코덱 토큰
        '<xcodec>': 32014,
        '<stage_1>': 32015,
        '<stage_2>': 32016,
        
        # 기타
        '<PAD>': 32017,
        '<UNK>': 32018,
        
        # 가사 구조 토큰
        '<verse>': 32019,
        '<chorus>': 32020,
        '<bridge>': 32021,
        '<outro>': 32022,
        
        # TTS 전용 토큰
        '<transcript>': 32023,    # 전사본 시작
        '</transcript>': 32024,   # 전사본 종료
        '<speech>': 32025,        # 음성 구간
        '</speech>': 32026,       # 음성 구간 종료
        '<pause>': 32027,         # 일시정지/무음
        '<breath>': 32028,        # 호흡음
        '<noise>': 32029,         # 배경 소음
    }
    
    VOCAB_SIZE = 32030  # 전체 vocabulary 크기 (증가)
    
    # 토큰 범위 검증 상수
    AUDIO_TOKEN_START = 30000
    AUDIO_TOKEN_END = 40000
    TEXT_TOKEN_MAX = 29999
    
    def __init__(
        self,
        text_tokenizer_path: Optional[str] = None,
        audio_vocab_size: int = 1024,
        num_codebooks: int = 8
    ):
        """
        Args:
            text_tokenizer_path: SentencePiece 모델 경로
            audio_vocab_size: 오디오 코드북 크기
            num_codebooks: 오디오 코드북 개수
        """
        # 텍스트 토크나이저 (SentencePiece)
        if text_tokenizer_path and Path(text_tokenizer_path).exists():
            # Fixed typo: use the correct alias 'spm'
            self.text_tokenizer = spm.SentencePieceProcessor()
            self.text_tokenizer.load(text_tokenizer_path)
        else:
            self.text_tokenizer = None
            print("Warning: Text tokenizer not loaded. Using character-level tokenization.")
            
        self.audio_vocab_size = audio_vocab_size
        self.num_codebooks = num_codebooks
        
        # 역방향 매핑
        self.id_to_token = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        
        # EOS 토큰 ID 목록
        self.eos_token_ids = [
            self.SPECIAL_TOKENS['<EOA>'],
            self.SPECIAL_TOKENS['<EOD>'],
            self.SPECIAL_TOKENS['</REF>'],
            self.SPECIAL_TOKENS['</MASK>'],
            self.SPECIAL_TOKENS['</transcript>'],
            self.SPECIAL_TOKENS['</speech>']
        ]
        
    def encode_text(self, text: str, add_special_tokens: bool = True, text_type: str = 'lyrics') -> List[int]:
        """
        텍스트를 토큰 ID로 변환 (가사 또는 전사본)
        
        Args:
            text: 입력 텍스트
            add_special_tokens: 특수 토큰 추가 여부
            text_type: 'lyrics' 또는 'transcript'
        """
        if not text:
            return []
        
        # 텍스트 타입에 따른 전처리
        if text_type == 'lyrics':
            text = self._process_lyric_structure(text)
        elif text_type == 'transcript':
            text = self._process_transcript_structure(text)
            
        if self.text_tokenizer:
            # SentencePiece 토큰화
            tokens = self.text_tokenizer.encode(text)
        else:
            # 문자 단위 토큰화 (fallback)
            tokens = [min(ord(c) % 1000, 999) for c in text]
            
        # 특수 토큰 오프셋 적용 (텍스트 토큰은 낮은 ID 사용)
        tokens = [t if t < self.TEXT_TOKEN_MAX else self.SPECIAL_TOKENS['<UNK>'] for t in tokens]
        
        return tokens
    
    def encode_transcript(self, transcript: str) -> List[int]:
        """전사본 전용 인코딩"""
        return self.encode_text(transcript, text_type='transcript')
    
    def encode_lyrics(self, lyrics: str) -> List[int]:
        """가사 전용 인코딩"""
        return self.encode_text(lyrics, text_type='lyrics')
    
    def encode_audio(self, audio_codes: np.ndarray) -> List[int]:
        """안전한 오디오 토큰 인코딩"""
        # 입력 검증
        if audio_codes.shape[0] != self.num_codebooks:
            raise ValueError(f"Expected {self.num_codebooks} codebooks, got {audio_codes.shape[0]}")
            
        tokens = []
        
        # 오디오 시작 토큰
        tokens.append(self.SPECIAL_TOKENS['<SOA>'])
        
        # 각 타임스텝별로 코드북 인터리빙
        for t in range(audio_codes.shape[1]):
            for cb in range(self.num_codebooks):
                code_val = int(audio_codes[cb, t])
                
                # 범위 검증 및 클리핑
                if code_val < 0:
                    code_val = 0
                elif code_val >= self.audio_vocab_size:
                    code_val = self.audio_vocab_size - 1
                    
                token_id = self.AUDIO_TOKEN_START + cb * self.audio_vocab_size + code_val
                
                # 최종 범위 검증
                if token_id >= self.AUDIO_TOKEN_END:
                    raise ValueError(f"Audio token ID {token_id} exceeds maximum {self.AUDIO_TOKEN_END}")
                    
                tokens.append(token_id)
                
        # 오디오 종료 토큰
        tokens.append(self.SPECIAL_TOKENS['<EOA>'])
        
        return tokens
    
    def decode_audio_tokens(self, token_ids: List[int]) -> np.ndarray:
        """오디오 토큰 디코딩"""
        audio_tokens = [t for t in token_ids 
                       if self.AUDIO_TOKEN_START <= t < self.AUDIO_TOKEN_END]
        
        if len(audio_tokens) % self.num_codebooks != 0:
            print(f"Warning: Audio tokens not divisible by codebooks")
            # Pad to make divisible
            pad_length = self.num_codebooks - (len(audio_tokens) % self.num_codebooks)
            audio_tokens.extend([self.AUDIO_TOKEN_START] * pad_length)
        
        time_steps = len(audio_tokens) // self.num_codebooks
        audio_codes = np.zeros((self.num_codebooks, time_steps), dtype=np.int32)
        
        for i, token_id in enumerate(audio_tokens):
            t = i // self.num_codebooks
            cb = i % self.num_codebooks
            code_val = token_id - self.AUDIO_TOKEN_START - cb * self.audio_vocab_size
            
            # 범위 검증
            if 0 <= code_val < self.audio_vocab_size:
                audio_codes[cb, t] = code_val
            else:
                print(f"Warning: Invalid code value {code_val} at position ({cb}, {t})")
                audio_codes[cb, t] = 0  # 기본값
            
        return audio_codes
    
    def create_sequence(
        self,
        task: str,
        text: Optional[str] = None,
        transcript: Optional[str] = None,  # TTS 전사본 추가
        audio_codes: Optional[np.ndarray] = None,
        reference_codes: Optional[np.ndarray] = None,
        reference_length: Optional[float] = None,
        style_prompt: Optional[str] = None
    ) -> List[int]:
        """
        완전한 시퀀스 생성 with TTS support
        
        태스크에 따라 적절한 토큰 시퀀스를 구성합니다.
        """
        sequence = []
        
        # 1. 태스크 토큰
        task_token = f'<TASK={task}>'
        if task_token in self.SPECIAL_TOKENS:
            sequence.append(self.SPECIAL_TOKENS[task_token])
        else:
            raise ValueError(f"Unknown task: {task}")
            
        # 2. 스타일 프롬프트 (있는 경우)
        if style_prompt:
            style_tokens = self.encode_text(style_prompt)
            sequence.extend(style_tokens)
            
        # 3. COVER 태스크의 경우 참조 처리
        if task == 'COVER' and reference_codes is not None:
            # 참조 길이에 따른 토큰 선택
            if reference_length is None or reference_length <= 10:
                ref_token = '<REF=10s>'
            elif reference_length <= 30:
                ref_token = '<REF=30s>'
            elif reference_length <= 60:
                ref_token = '<REF=60s>'
            else:
                ref_token = '<REF=180s>'
                
            sequence.append(self.SPECIAL_TOKENS[ref_token])
            
            # 참조 오디오 토큰
            try:
                ref_tokens = self.encode_audio(reference_codes)
                sequence.extend(ref_tokens)
            except Exception as e:
                print(f"Warning: Failed to encode reference audio: {e}")
                # 빈 참조로 대체
                sequence.extend([self.SPECIAL_TOKENS['<SOA>'], self.SPECIAL_TOKENS['<EOA>']])
            
            # 참조 종료
            sequence.append(self.SPECIAL_TOKENS['</REF>'])
            
        # 4. 텍스트 처리 (태스크별)
        if task == 'TTS':
            # TTS는 전사본 사용
            if transcript:
                sequence.append(self.SPECIAL_TOKENS['<transcript>'])
                transcript_tokens = self.encode_transcript(transcript)
                sequence.extend(transcript_tokens)
                sequence.append(self.SPECIAL_TOKENS['</transcript>'])
        elif task != 'INST':
            # SONG, COVER는 가사 사용
            if text:
                text_tokens = self.encode_lyrics(text)
                sequence.extend(text_tokens)
            
        # 5. 오디오 토큰
        if audio_codes is not None:
            try:
                if task == 'TTS':
                    # TTS는 음성 구간 표시
                    sequence.append(self.SPECIAL_TOKENS['<speech>'])
                    audio_tokens = self.encode_audio(audio_codes)
                    sequence.extend(audio_tokens)
                    sequence.append(self.SPECIAL_TOKENS['</speech>'])
                else:
                    # 일반 오디오
                    audio_tokens = self.encode_audio(audio_codes)
                    sequence.extend(audio_tokens)
            except Exception as e:
                print(f"Warning: Failed to encode audio: {e}")
                # 빈 오디오로 대체
                sequence.extend([self.SPECIAL_TOKENS['<SOA>'], self.SPECIAL_TOKENS['<EOA>']])
            
        # 6. 문서 종료 토큰
        sequence.append(self.SPECIAL_TOKENS['<EOD>'])
        
        return sequence
    
    def decode(self, token_ids: List[int]) -> Dict[str, Union[str, List[int]]]:
        """토큰 ID를 디코드하여 구조화된 형태로 반환 with TTS support"""
        result = {
            'task': None,
            'text': '',
            'transcript': '',     # TTS 전사본 추가
            'audio_tokens': [],
            'reference_tokens': [],
            'special_tokens': []
        }
        
        i = 0
        in_transcript = False
        
        while i < len(token_ids):
            token_id = token_ids[i]
            
            # 특수 토큰 확인
            if token_id in self.id_to_token:
                special_token = self.id_to_token[token_id]
                result['special_tokens'].append(special_token)
                
                # 태스크 토큰
                if special_token.startswith('<TASK='):
                    result['task'] = special_token
                
                # 전사본 구간 처리
                elif special_token == '<transcript>':
                    in_transcript = True
                elif special_token == '</transcript>':
                    in_transcript = False
                    
                # 참조 구간 처리
                elif special_token.startswith('<REF='):
                    i += 1
                    # 참조 오디오 토큰 수집
                    while i < len(token_ids) and token_ids[i] != self.SPECIAL_TOKENS['</REF>']:
                        result['reference_tokens'].append(token_ids[i])
                        i += 1
                        
            # 오디오 토큰 (30000번대)
            elif self.AUDIO_TOKEN_START <= token_id < self.AUDIO_TOKEN_END:
                result['audio_tokens'].append(token_id)
                
            # 텍스트 토큰
            elif token_id <= self.TEXT_TOKEN_MAX:
                if self.text_tokenizer:
                    # SentencePiece 디코드
                    try:
                        text = self.text_tokenizer.decode([token_id])
                        if in_transcript:
                            result['transcript'] += text
                        else:
                            result['text'] += text
                    except:
                        # 디코딩 실패 시 건너뜀
                        pass
                else:
                    # 문자 디코드
                    try:
                        char = chr(token_id % 256)
                        if in_transcript:
                            result['transcript'] += char
                        else:
                            result['text'] += char
                    except:
                        # 잘못된 문자 코드 무시
                        pass
                    
            i += 1
            
        return result
    
    def _process_lyric_structure(self, text: str) -> str:
        """가사 구조 태그를 특수 토큰으로 변환"""
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
        
        processed_text = text
        for pattern, replacement in structure_patterns:
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
        
        return processed_text
    
    def _process_transcript_structure(self, text: str) -> str:
        """전사본 구조 처리 (TTS 전용)"""
        import re
        
        # TTS 전사본 특수 표시 처리
        transcript_patterns = [
            (r'\[pause\]', '<pause>'),
            (r'\[breath\]', '<breath>'),
            (r'\[noise\]', '<noise>'),
            (r'\[.*?\]', ''),  # 기타 주석 제거
            (r'\(.*?\)', ''),  # 괄호 주석 제거
        ]
        
        processed_text = text
        for pattern, replacement in transcript_patterns:
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
        
        # 연속 공백 정리
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        return processed_text
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        text_type: str = 'lyrics'
    ) -> torch.Tensor:
        """배치 텍스트 인코딩 with TTS support"""
        encoded = []
        
        for text in texts:
            try:
                tokens = self.encode_text(text, text_type=text_type)
                if max_length and len(tokens) > max_length:
                    tokens = tokens[:max_length]
                encoded.append(tokens)
            except Exception as e:
                print(f"Warning: Failed to encode text '{text[:50]}...': {e}")
                encoded.append([self.SPECIAL_TOKENS['<UNK>']])
            
        # 패딩
        if padding and encoded:
            max_len = max(len(tokens) for tokens in encoded) if not max_length else max_length
            padded = []
            
            for tokens in encoded:
                if len(tokens) < max_len:
                    tokens = tokens + [self.SPECIAL_TOKENS['<PAD>']] * (max_len - len(tokens))
                elif len(tokens) > max_len:
                    tokens = tokens[:max_len]
                padded.append(tokens)
                
            return torch.tensor(padded)
        else:
            return encoded
    
    def batch_encode_mixed(
        self,
        lyrics_list: List[Optional[str]],
        transcript_list: List[Optional[str]],
        max_length: Optional[int] = None,
        padding: bool = True
    ) -> torch.Tensor:
        """가사와 전사본 혼합 배치 인코딩"""
        encoded = []
        
        for lyrics, transcript in zip(lyrics_list, transcript_list):
            try:
                if transcript:  # 전사본이 있으면 전사본 사용
                    tokens = self.encode_transcript(transcript)
                elif lyrics:   # 가사가 있으면 가사 사용
                    tokens = self.encode_lyrics(lyrics)
                else:          # 둘 다 없으면 빈 토큰
                    tokens = []
                    
                if max_length and len(tokens) > max_length:
                    tokens = tokens[:max_length]
                encoded.append(tokens)
            except Exception as e:
                print(f"Warning: Failed to encode text: {e}")
                encoded.append([self.SPECIAL_TOKENS['<UNK>']])
        
        # 패딩
        if padding and encoded:
            max_len = max(len(tokens) for tokens in encoded) if not max_length else max_length
            padded = []
            
            for tokens in encoded:
                if len(tokens) < max_len:
                    tokens = tokens + [self.SPECIAL_TOKENS['<PAD>']] * (max_len - len(tokens))
                elif len(tokens) > max_len:
                    tokens = tokens[:max_len]
                padded.append(tokens)
                
            return torch.tensor(padded)
        else:
            return encoded
    
    def get_eos_penalty_mask(self, current_length: int, min_length: int = 100) -> torch.Tensor:
        """
        EOS 토큰 페널티 마스크 생성 (TTS 지원)
        
        너무 짧은 생성을 방지하기 위해 초기에는 EOS 토큰에 페널티를 부여합니다.
        """
        if current_length < min_length:
            # EOS 토큰들에 대해 페널티 적용
            penalty = -float('inf')
        else:
            # 충분한 길이 후에는 페널티 제거
            penalty = 0.0
            
        mask = torch.zeros(self.VOCAB_SIZE)
        for eos_id in self.eos_token_ids:
            if eos_id < self.VOCAB_SIZE:  # 범위 검증
                mask[eos_id] = penalty
            
        return mask
    
    def validate_sequence(self, token_ids: List[int]) -> Dict[str, Union[bool, List[str]]]:
        """토큰 시퀀스 유효성 검증 with TTS support"""
        issues = []
        
        # 기본 검증
        if not token_ids:
            issues.append("Empty sequence")
            return {"valid": False, "issues": issues}
        
        # 토큰 범위 검증
        for i, token_id in enumerate(token_ids):
            if token_id < 0 or token_id >= self.VOCAB_SIZE:
                issues.append(f"Token {token_id} at position {i} is out of range")
        
        # 구조 검증
        has_task = any(self.id_to_token.get(t, '').startswith('<TASK=') for t in token_ids)
        if not has_task:
            issues.append("No task token found")
        
        # EOS 토큰 검증
        has_eos = any(t in self.eos_token_ids for t in token_ids)
        if not has_eos:
            issues.append("No EOS token found")
        
        # TTS 특화 검증
        task_tokens = [self.id_to_token.get(t, '') for t in token_ids if self.id_to_token.get(t, '').startswith('<TASK=')]
        if '<TASK=TTS>' in task_tokens:
            # TTS 태스크의 경우 전사본 구조 확인
            has_transcript = self.SPECIAL_TOKENS['<transcript>'] in token_ids
            if not has_transcript:
                issues.append("TTS task without transcript structure")
        
        # 오디오 구조 검증
        audio_tokens = [t for t in token_ids if self.AUDIO_TOKEN_START <= t < self.AUDIO_TOKEN_END]
        if audio_tokens:
            # SOA/EOA 쌍 검증
            soa_count = token_ids.count(self.SPECIAL_TOKENS['<SOA>'])
            eoa_count = token_ids.count(self.SPECIAL_TOKENS['<EOA>'])
            if soa_count != eoa_count:
                issues.append(f"Mismatched SOA/EOA tokens: {soa_count} SOA, {eoa_count} EOA")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def get_token_stats(self) -> Dict[str, int]:
        """토크나이저 통계 정보 반환 with TTS"""
        return {
            "vocab_size": self.VOCAB_SIZE,
            "special_tokens": len(self.SPECIAL_TOKENS),
            "audio_vocab_size": self.audio_vocab_size,
            "num_codebooks": self.num_codebooks,
            "audio_token_range": (self.AUDIO_TOKEN_START, self.AUDIO_TOKEN_END),
            "text_token_max": self.TEXT_TOKEN_MAX,
            "eos_tokens": len(self.eos_token_ids),
            "tts_support": True,
            "tts_special_tokens": 7  # TTS 관련 특수 토큰 개수
        }