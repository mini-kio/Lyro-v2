import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from transformers import AutoModel, AutoTokenizer
import warnings
import re
import json
import os

warnings.filterwarnings("ignore")






class UnifiedTextEncoder(nn.Module):
    """
    통합 텍스트 인코더: 가사와 스타일을 분리 입력하되 내부에서 상호작용
    
    입력:
    - lyrics: "[verse 1:piano] 가사 내용" (섹션 태그 + 가사)
    - style: "energetic pop rock, upbeat" (장르, 분위기, 스타일)
    
    내부 처리: Cross-attention으로 lyrics ↔ style 상호작용
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 768,
        hidden_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 12,
        max_lyrics_length: int = 256,
        max_style_length: int = 128,
        dropout: float = 0.1,
        pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_lyrics_length = max_lyrics_length
        self.max_style_length = max_style_length
        self.total_max_length = max_lyrics_length + max_style_length
        
        # 동적 vocab 초기화
        self.section_vocab = {}    # [verse], [chorus] 등
        self.style_vocab = {}      # piano, pop, energetic 등 (악기+장르+분위기 통합)
        self.max_section_types = 50
        self.max_style_types = 200  # 더 많은 스타일 지원
        
        # 임베딩 레이어들
        self.section_embedding = nn.Embedding(self.max_section_types, embed_dim // 4)
        self.style_embedding = nn.Embedding(self.max_style_types, embed_dim // 4)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 위치 임베딩 (lyrics + style 구분)
        self.lyrics_pos_embedding = nn.Parameter(torch.randn(1, max_lyrics_length, embed_dim) * 0.02)
        self.style_pos_embedding = nn.Parameter(torch.randn(1, max_style_length, embed_dim) * 0.02)
        
        # 타입 임베딩 (lyrics vs style 구분)
        self.type_embedding = nn.Embedding(2, embed_dim)  # 0: lyrics, 1: style
        
        # 사전 훈련된 텍스트 인코더 (fallback)
        try:
            self.pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.pretrained_encoder = AutoModel.from_pretrained(pretrained_model)
            for param in self.pretrained_encoder.parameters():
                param.requires_grad = False
            self.has_pretrained = True
            
            pretrained_dim = self.pretrained_encoder.config.hidden_size
            self.pretrained_proj = nn.Linear(pretrained_dim, embed_dim)
        except:
            self.has_pretrained = False
            
        # Cross-attention Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 상호작용 레이어들
        self.lyrics_style_cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.style_lyrics_cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 최종 출력 프로젝션
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # 어텐션 풀링 (최종 임베딩 생성)
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attention_pool = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def parse_lyrics(self, lyrics: str) -> Tuple[str, List[str], List[str]]:
        """
        가사 파싱: 다중 태그 지원
        
        지원 형식:
        - [verse 1:piano] content
        - [chorus:guitar:energetic] content  
        - [bridge] content
        - [piano] content
        - content (태그 없음)
        """
        if not lyrics or not lyrics.strip():
            return "", [], []
            
        # 패턴들 (우선순위 순)
        patterns = [
            r'\[([^:]+):([^:]+):([^]]+)\]\s*(.*)',  # [section:style:mood] content
            r'\[([^:]+):([^]]+)\]\s*(.*)',          # [section:style] content
            r'\[([^]]+)\]\s*(.*)',                  # [section or style] content
        ]
        
        sections = []
        styles = []
        content = lyrics.strip()
        
        for pattern in patterns:
            match = re.match(pattern, content)
            if match:
                groups = match.groups()
                if len(groups) == 4:  # [a:b:c] content
                    sections.append(groups[0].strip().lower())
                    styles.extend([groups[1].strip().lower(), groups[2].strip().lower()])
                    content = groups[3].strip()
                elif len(groups) == 3:  # [a:b] content
                    tag1, tag2 = groups[0].strip().lower(), groups[1].strip().lower()
                    # 첫 번째는 섹션, 두 번째는 스타일로 분류
                    sections.append(tag1)
                    styles.append(tag2)
                    content = groups[2].strip()
                elif len(groups) == 2:  # [a] content
                    tag = groups[0].strip().lower()
                    # 섹션인지 스타일인지 판단 (기본적으로 섹션으로)
                    if any(s in tag for s in ['verse', 'chorus', 'bridge', 'intro', 'outro', 'pre']):
                        sections.append(tag)
                    else:
                        styles.append(tag)
                    content = groups[1].strip()
                break
        
        return content, sections, styles
    
    def parse_style(self, style: str) -> List[str]:
        """
        스타일 텍스트 파싱: 콤마/공백으로 분리된 스타일들
        
        예시: "energetic pop rock, upbeat tempo" → ["energetic", "pop", "rock", "upbeat", "tempo"]
        """
        if not style or not style.strip():
            return []
            
        # 콤마로 분리 후 공백으로 재분리
        styles = []
        for part in style.split(','):
            for word in part.strip().split():
                word = word.strip().lower()
                if word and word not in styles:  # 중복 제거
                    styles.append(word)
        
        return styles
    
    def build_vocab_from_data(self, lyrics_list: List[str], style_list: List[str] = None):
        """데이터셋에서 섹션과 스타일 vocab 구축"""
        sections_found = set()
        styles_found = set()
        
        # 가사에서 섹션과 스타일 추출
        for lyrics in lyrics_list:
            _, sections, styles = self.parse_lyrics(lyrics)
            sections_found.update(sections)
            styles_found.update(styles)
        
        # 스타일 텍스트에서 스타일 추출
        if style_list:
            for style in style_list:
                styles = self.parse_style(style)
                styles_found.update(styles)
        
        # vocab 구축 (기존 vocab과 병합)
        for section in sections_found:
            if section not in self.section_vocab:
                new_idx = len(self.section_vocab)
                if new_idx < self.max_section_types:
                    self.section_vocab[section] = new_idx
        
        for style in styles_found:
            if style not in self.style_vocab:
                new_idx = len(self.style_vocab)
                if new_idx < self.max_style_types:
                    self.style_vocab[style] = new_idx
        
        print(f"Section vocab ({len(self.section_vocab)}): {list(self.section_vocab.keys())}")
        print(f"Style vocab ({len(self.style_vocab)}): {list(self.style_vocab.keys())}")
    
    def save_vocab(self, vocab_path: str):
        """vocab을 파일로 저장"""
        vocab_data = {
            'section_vocab': self.section_vocab,
            'style_vocab': self.style_vocab
        }
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"Vocab saved to {vocab_path}")
    
    def load_vocab(self, vocab_path: str):
        """파일에서 vocab 로드"""
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            self.section_vocab = vocab_data.get('section_vocab', {})
            self.style_vocab = vocab_data.get('style_vocab', {})
            # 하위 호환성
            if 'instrument_vocab' in vocab_data:
                self.style_vocab.update(vocab_data['instrument_vocab'])
            
            print(f"Vocab loaded from {vocab_path}")
            print(f"Section vocab ({len(self.section_vocab)}): {list(self.section_vocab.keys())}")
            print(f"Style vocab ({len(self.style_vocab)}): {list(self.style_vocab.keys())}")
        else:
            print(f"Vocab file not found: {vocab_path}")
    
    def get_vocab_stats(self) -> Dict[str, Any]:
        """vocab 통계 반환"""
        return {
            'section_vocab_size': len(self.section_vocab),
            'style_vocab_size': len(self.style_vocab),
            'section_coverage': len(self.section_vocab) / self.max_section_types,
            'style_coverage': len(self.style_vocab) / self.max_style_types,
            'sections': list(self.section_vocab.keys()),
            'styles': list(self.style_vocab.keys())
        }
    
    def forward(
        self, 
        lyrics: Optional[List[str]] = None,
        style: Optional[List[str]] = None,
        token_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        분리된 입력으로 상호작용하는 통합 인코딩
        
        Args:
            lyrics: 가사 리스트 (섹션 태그 포함) - ["[verse 1:piano] 내용"]
            style: 스타일 리스트 - ["energetic pop rock"]
            token_ids: 선택적 토큰 ID
            
        Returns:
            (B, embed_dim) 통합 임베딩
        """
        device = next(self.parameters()).device
        
        # 배치 크기 결정
        if lyrics is not None:
            batch_size = len(lyrics)
        elif style is not None:
            batch_size = len(style)
        else:
            raise ValueError("Either lyrics or style must be provided")
        
        # 기본값 설정
        if lyrics is None:
            lyrics = [""] * batch_size
        if style is None:
            style = [""] * batch_size
        
        # 1. 가사 처리
        lyrics_embeddings = []
        for i, lyric in enumerate(lyrics):
            content, sections, lyric_styles = self.parse_lyrics(lyric)
            
            # 텍스트 인코딩
            if self.has_pretrained and content.strip():
                content_embed = self._encode_with_pretrained([content])[0]
            else:
                content_embed = torch.zeros(self.embed_dim, device=device)
            
            # 섹션 임베딩
            section_embed = self._encode_sections(sections, device)
            
            # 가사 내 스타일 임베딩
            lyric_style_embed = self._encode_styles(lyric_styles, device)
            
            # 가사 통합 임베딩
            lyric_embed = content_embed + section_embed + lyric_style_embed
            lyrics_embeddings.append(lyric_embed)
        
        lyrics_tensor = torch.stack(lyrics_embeddings)  # (B, embed_dim)
        
        # 2. 스타일 처리  
        style_embeddings = []
        for style_text in style:
            parsed_styles = self.parse_style(style_text)
            
            # 스타일 텍스트 인코딩
            if self.has_pretrained and style_text.strip():
                style_content_embed = self._encode_with_pretrained([style_text])[0]
            else:
                style_content_embed = torch.zeros(self.embed_dim, device=device)
            
            # 스타일 태그 임베딩
            style_tag_embed = self._encode_styles(parsed_styles, device)
            
            # 스타일 통합 임베딩
            style_embed = style_content_embed + style_tag_embed
            style_embeddings.append(style_embed)
        
        style_tensor = torch.stack(style_embeddings)  # (B, embed_dim)
        
        # 3. 위치 및 타입 임베딩 추가
        lyrics_tensor = lyrics_tensor.unsqueeze(1)  # (B, 1, embed_dim)
        style_tensor = style_tensor.unsqueeze(1)   # (B, 1, embed_dim)
        
        # 타입 임베딩 (lyrics vs style 구분)
        lyrics_type = self.type_embedding(torch.zeros(batch_size, 1, dtype=torch.long, device=device))
        style_type = self.type_embedding(torch.ones(batch_size, 1, dtype=torch.long, device=device))
        
        lyrics_tensor = lyrics_tensor + lyrics_type
        style_tensor = style_tensor + style_type
        
        # 4. Cross-attention 상호작용
        # Lyrics가 Style을 참고
        lyrics_enhanced, _ = self.lyrics_style_cross_attention(
            query=lyrics_tensor,
            key=style_tensor,
            value=style_tensor
        )
        
        # Style이 Lyrics를 참고
        style_enhanced, _ = self.style_lyrics_cross_attention(
            query=style_tensor,
            key=lyrics_tensor,
            value=lyrics_tensor
        )
        
        # 5. 융합
        combined = torch.cat([
            lyrics_enhanced.squeeze(1),  # (B, embed_dim)
            style_enhanced.squeeze(1)    # (B, embed_dim)
        ], dim=-1)  # (B, embed_dim * 2)
        
        fused = self.fusion_layer(combined)  # (B, embed_dim)
        
        # 6. 최종 어텐션 풀링
        query = self.pool_query.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        
        # 전체 시퀀스로 어텐션 (lyrics + style)
        full_sequence = torch.cat([lyrics_enhanced, style_enhanced], dim=1)  # (B, 2, embed_dim)
        
        final_embed, _ = self.attention_pool(query, full_sequence, full_sequence)
        
        return self.output_proj(final_embed.squeeze(1))  # (B, embed_dim)
    
    def _encode_sections(self, sections: List[str], device: torch.device) -> torch.Tensor:
        """섹션 리스트를 임베딩으로 변환"""
        if not sections:
            return torch.zeros(self.embed_dim // 4, device=device)
        
        section_embeds = []
        for section in sections:
            if section in self.section_vocab:
                idx = self.section_vocab[section]
            else:
                idx = 0  # unknown
            section_embeds.append(self.section_embedding(torch.tensor(idx, device=device)))
        
        # 평균 또는 합계
        if section_embeds:
            combined = torch.stack(section_embeds).mean(dim=0)
            # embed_dim // 4 → embed_dim으로 확장
            expanded = torch.cat([combined, torch.zeros(self.embed_dim - self.embed_dim // 4, device=device)])
            return expanded
        else:
            return torch.zeros(self.embed_dim, device=device)
    
    def _encode_styles(self, styles: List[str], device: torch.device) -> torch.Tensor:
        """스타일 리스트를 임베딩으로 변환"""
        if not styles:
            return torch.zeros(self.embed_dim // 4, device=device)
        
        style_embeds = []
        for style in styles:
            if style in self.style_vocab:
                idx = self.style_vocab[style]
            else:
                idx = 0  # unknown
            style_embeds.append(self.style_embedding(torch.tensor(idx, device=device)))
        
        # 평균 또는 합계
        if style_embeds:
            combined = torch.stack(style_embeds).mean(dim=0)
            # embed_dim // 4 → embed_dim으로 확장
            expanded = torch.cat([combined, torch.zeros(self.embed_dim - self.embed_dim // 4, device=device)])
            return expanded
        else:
            return torch.zeros(self.embed_dim, device=device)
    
    def _encode_with_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """토큰 ID로 인코딩 (미사용 - 호환성 유지)"""
        B, L = token_ids.shape
        
        if L > self.max_lyrics_length:
            token_ids = token_ids[:, :self.max_lyrics_length]
            L = self.max_lyrics_length
            
        x = self.token_embedding(token_ids)
        x = x + self.lyrics_pos_embedding[:, :L, :]
        
        x = self.transformer(x)
        
        # 평균 풀링
        return x.mean(dim=1)
    
    def _encode_with_pretrained(self, texts: List[str]) -> torch.Tensor:
        """사전 훈련된 모델로 인코딩"""
        device = next(self.parameters()).device
        
        # 토크나이징
        encoded = self.pretrained_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # GPU로 이동
        for k, v in encoded.items():
            encoded[k] = v.to(device)
        
        # 인코딩
        with torch.no_grad():
            outputs = self.pretrained_encoder(**encoded)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # 평균 풀링
        
        return self.pretrained_proj(embeddings)
    
    def _encode_basic(self, texts: List[str]) -> torch.Tensor:
        """기본 인코딩 (간단한 문자 기반)"""
        device = next(self.parameters()).device
        batch_size = len(texts)
        
        # 간단한 문자 기반 토크나이징 (실제로는 더 정교한 토크나이저 필요)
        max_len = min(self.max_length, max(len(text) for text in texts) if texts else 1)
        
        token_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        
        for i, text in enumerate(texts):
            for j, char in enumerate(text[:max_len]):
                token_ids[i, j] = ord(char) % self.token_embedding.num_embeddings
        
        return self._encode_with_tokens(token_ids)




