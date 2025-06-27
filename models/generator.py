# lyro/models/generator.py
"""
LYRO SSM + Flow Matching Generator (1.5B 파라미터)
S6 State Space Model + Flow Matching 기반 음악 생성 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from einops import rearrange, repeat
from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    """Generator 설정"""
    # 모델 크기 (1.5B 타겟)
    d_model: int = 1536
    n_layers: int = 24
    n_heads: int = 24
    d_ff: int = 6144
    
    # SSM 설정
    d_state: int = 64
    d_conv: int = 4
    expand_factor: int = 2
    
    # 입출력 설정
    latent_channels: int = 16
    latent_time_steps: int = 128
    vocab_size: int = 32000
    
    # 컨디션 설정
    max_lyrics_length: int = 512
    max_caption_length: int = 256
    condition_dropout: float = 0.1
    
    # Flow Matching 설정
    flow_steps: int = 50
    sigma_min: float = 1e-4
    sigma_max: float = 1.0
    
    # 기타
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 주파수 계산
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 사전 계산된 cos, sin 캐시
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """캐시 업데이트"""
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq)
            
            self._cached_cos = torch.cos(freqs).to(dtype)
            self._cached_sin = torch.sin(freqs).to(dtype)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[-2]
        self._update_cache(seq_len, x.device, x.dtype)
        
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotary embedding 적용"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class S6Layer(nn.Module):
    """S6 State Space Layer"""
    
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand_factor
        self.d_inner = config.d_model * self.expand
        
        # 입력 투영
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # 컨볼루션
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=self.d_conv, 
            groups=self.d_inner,
            padding=self.d_conv - 1
        )
        
        # SSM 파라미터
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # A 매트릭스 (음수로 초기화)
        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A.float()))
        
        # D 매트릭스 (스킵 연결)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 출력 투영
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        
        # 정규화
        self.norm = nn.LayerNorm(self.d_inner, eps=config.layer_norm_eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # 입력 투영
        xz = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # (batch, seq_len, d_inner) each
        
        # 컨볼루션 (시퀀스 차원에서)
        x_conv = rearrange(x_inner, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[..., :seq_len]  # 패딩 제거
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        
        # 활성화
        x_conv = F.silu(x_conv)
        
        # SSM 연산
        dt = F.softplus(self.dt_proj(x_conv))  # (batch, seq_len, d_inner)
        
        BC = self.x_proj(x_conv)  # (batch, seq_len, 2 * d_state)
        B, C = BC.chunk(2, dim=-1)  # (batch, seq_len, d_state) each
        
        # A 매트릭스 (음수)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # 이산화
        dA = torch.einsum('bld,dn->bldn', dt, A)
        dB = torch.einsum('bld,bln->bldn', dt, B)
        
        # 상태 공간 연산 (간소화된 버전)
        # 실제로는 더 복잡한 연산이 필요하지만, 여기서는 근사
        y = torch.einsum('bldn,bln->bld', dB, C) + x_conv * self.D
        
        # 정규화 및 게이팅
        y = self.norm(y)
        y = y * F.silu(z)
        
        # 출력 투영
        output = self.out_proj(y)
        
        return output


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with RoPE"""
    
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        
        assert config.d_model % config.n_heads == 0
        
        # QKV 투영
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(self.d_head)
        
        # 스케일링
        self.scale = self.d_head ** -0.5
        
        # 드롭아웃
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        
        # QKV 계산
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 헤드 차원으로 reshape
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # RoPE 적용
        cos, sin = self.rope(x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention 계산
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Value와 곱하기
        out = torch.matmul(attn_weights, v)
        
        # 원래 형태로 복원
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        
        # 출력 투영
        out = self.o_proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """Transformer Block with S6 and Attention"""
    
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        
        # S6 레이어
        self.s6_layer = S6Layer(config)
        self.s6_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Attention 레이어
        self.attn_layer = MultiHeadAttention(config)
        self.attn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        self.ffn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # S6 블록
        x = x + self.s6_layer(self.s6_norm(x))
        
        # Attention 블록
        x = x + self.attn_layer(self.attn_norm(x), mask)
        
        # FFN 블록
        x = x + self.ffn(self.ffn_norm(x))
        
        return x


class FlowMatching(nn.Module):
    """Flow Matching for continuous generation"""
    
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
    
    def get_noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """노이즈 스케줄 계산"""
        # 코사인 스케줄
        return self.sigma_min + (self.sigma_max - self.sigma_min) * (1 - torch.cos(t * math.pi)) / 2
    
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """시간 샘플링"""
        return torch.rand(batch_size, device=device)
    
    def forward_process(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process for flow matching
        
        Args:
            x0: 노이즈 (batch, channels, time)
            x1: 데이터 (batch, channels, time)
            t: 시간 (batch,)
        
        Returns:
            xt: 보간된 상태
            target_v: 타겟 속도 벡터
        """
        # 시간 차원 확장
        t = t.view(-1, 1, 1)
        
        # 선형 보간
        xt = (1 - t) * x0 + t * x1
        
        # 노이즈 추가
        sigma_t = self.get_noise_schedule(t.squeeze())
        noise = torch.randn_like(xt) * sigma_t.view(-1, 1, 1)
        xt = xt + noise
        
        # 타겟 속도 (x1 - x0)
        target_v = x1 - x0
        
        return xt, target_v
    
    def reverse_process(
        self, 
        x: torch.Tensor, 
        velocity_fn, 
        num_steps: int = 50,
        cfg_scale: float = 1.0,
        **condition_kwargs
    ) -> torch.Tensor:
        """
        Reverse process for generation
        
        Args:
            x: 초기 노이즈
            velocity_fn: 속도 예측 함수
            num_steps: 생성 스텝 수
            cfg_scale: Classifier-free guidance 스케일
            **condition_kwargs: 조건 인자들
        
        Returns:
            생성된 샘플
        """
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            
            # 속도 예측
            if cfg_scale > 1.0:
                # Classifier-free guidance
                v_cond = velocity_fn(x, t, **condition_kwargs)
                v_uncond = velocity_fn(x, t, **{k: None for k in condition_kwargs.keys()})
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = velocity_fn(x, t, **condition_kwargs)
            
            # 상태 업데이트
            x = x + dt * v
        
        return x


class LyroGenerator(nn.Module):
    """
    LYRO SSM + Flow Matching Generator (1.5B 파라미터)
    """
    
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        
        # 입력 임베딩
        self.latent_embed = nn.Linear(config.latent_channels, config.d_model)
        
        # 시간 임베딩
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # 조건 처리
        self.condition_processor = ConditionProcessor(config)
        
        # Transformer 블록들
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # 출력 레이어
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.output_proj = nn.Linear(config.d_model, config.latent_channels)
        
        # Flow matching
        self.flow_matching = FlowMatching(config)
        
        # 파라미터 초기화
        self.apply(self._init_weights)
        
        print(f"LyroGenerator initialized with {self.count_parameters():,} parameters")
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def count_parameters(self) -> int:
        """파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        lyrics: Optional[torch.Tensor] = None,
        lyrics_mask: Optional[torch.Tensor] = None,
        captions: Optional[List[str]] = None,
        reference_audio: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            latents: (batch, channels, time) 잠재 벡터
            timesteps: (batch,) 시간스텝
            lyrics: (batch, seq_len) 가사 토큰
            lyrics_mask: (batch, seq_len) 가사 마스크
            captions: List[str] 캡션 텍스트
            reference_audio: (batch, channels, time) 참조 오디오
            task_type: 태스크 타입
        
        Returns:
            (batch, channels, time) 예측된 속도 벡터
        """
        batch_size = latents.shape[0]
        
        # 잠재 벡터를 시퀀스 형태로 변환
        latents = rearrange(latents, 'b c t -> b t c')  # (batch, time, channels)
        
        # 잠재 임베딩
        x = self.latent_embed(latents)  # (batch, time, d_model)
        
        # 시간 임베딩
        time_embed = self.time_embed(timesteps.unsqueeze(-1))  # (batch, d_model)
        time_embed = time_embed.unsqueeze(1)  # (batch, 1, d_model)
        x = x + time_embed
        
        # 조건 처리 및 임베딩
        condition_embed = self.condition_processor(
            lyrics=lyrics,
            lyrics_mask=lyrics_mask,
            captions=captions,
            reference_audio=reference_audio,
            task_type=task_type,
            batch_size=batch_size
        )  # (batch, 1, d_model)
        
        x = x + condition_embed
        
        # Transformer 레이어들
        for layer in self.layers:
            x = layer(x)
        
        # 출력 투영
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        # 원래 형태로 복원
        x = rearrange(x, 'b t c -> b c t')
        
        return x
    
    def training_loss(
        self,
        latents: torch.Tensor,
        lyrics: Optional[torch.Tensor] = None,
        lyrics_mask: Optional[torch.Tensor] = None,
        captions: Optional[List[str]] = None,
        reference_audio: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training loss 계산
        
        Args:
            latents: (batch, channels, time) 타겟 잠재 벡터
            기타: 조건들
        
        Returns:
            손실 딕셔너리
        """
        batch_size = latents.shape[0]
        device = latents.device
        
        # 시간 샘플링
        t = self.flow_matching.sample_time(batch_size, device)
        
        # 노이즈 샘플링
        noise = torch.randn_like(latents)
        
        # Forward process
        xt, target_v = self.flow_matching.forward_process(noise, latents, t)
        
        # 속도 예측
        predicted_v = self.forward(
            latents=xt,
            timesteps=t,
            lyrics=lyrics,
            lyrics_mask=lyrics_mask,
            captions=captions,
            reference_audio=reference_audio,
            task_type=task_type
        )
        
        # Flow matching loss
        flow_loss = F.mse_loss(predicted_v, target_v)
        
        return {
            'flow_loss': flow_loss,
            'predicted_v': predicted_v,
            'target_v': target_v
        }
    
    @torch.no_grad()
    def generate(
        self,
        shape: Tuple[int, int, int],
        lyrics: Optional[torch.Tensor] = None,
        lyrics_mask: Optional[torch.Tensor] = None,
        captions: Optional[List[str]] = None,
        reference_audio: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None,
        num_steps: int = 50,
        cfg_scale: float = 7.5,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate samples
        
        Args:
            shape: (batch, channels, time) 생성할 형태
            기타: 조건들
            num_steps: 생성 스텝 수
            cfg_scale: CFG 스케일
            device: 디바이스
        
        Returns:
            생성된 잠재 벡터
        """
        if device is None:
            device = next(self.parameters()).device
        
        # 초기 노이즈
        x = torch.randn(shape, device=device)
        
        # 속도 예측 함수
        def velocity_fn(latents, timesteps, **kwargs):
            return self.forward(
                latents=latents,
                timesteps=timesteps,
                **kwargs
            )
        
        # 생성
        generated = self.flow_matching.reverse_process(
            x=x,
            velocity_fn=velocity_fn,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            lyrics=lyrics,
            lyrics_mask=lyrics_mask,
            captions=captions,
            reference_audio=reference_audio,
            task_type=task_type
        )
        
        return generated


class ConditionProcessor(nn.Module):
    """조건 처리 모듈"""
    
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        
        # 가사 인코더
        self.lyrics_encoder = LyricsEncoder(config)
        
        # 캡션 인코더
        self.caption_encoder = CaptionEncoder(config)
        
        # 참조 오디오 인코더
        self.reference_encoder = ReferenceEncoder(config)
        
        # 태스크 임베딩
        self.task_embedding = nn.Embedding(10, config.d_model)  # 태스크 타입들
        
        # 융합 레이어
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model * 4, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model)
        )
    
    def forward(
        self,
        lyrics: Optional[torch.Tensor] = None,
        lyrics_mask: Optional[torch.Tensor] = None,
        captions: Optional[List[str]] = None,
        reference_audio: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        조건 처리
        
        Returns:
            (batch, 1, d_model) 조건 임베딩
        """
        device = next(self.parameters()).device
        
        # 각 조건 인코딩
        lyrics_embed = self.lyrics_encoder(lyrics, lyrics_mask) if lyrics is not None else torch.zeros(batch_size, self.config.d_model, device=device)
        
        caption_embed = self.caption_encoder(captions) if captions is not None else torch.zeros(batch_size, self.config.d_model, device=device)
        
        reference_embed = self.reference_encoder(reference_audio) if reference_audio is not None else torch.zeros(batch_size, self.config.d_model, device=device)
        
        # 태스크 임베딩
        task_id = self._get_task_id(task_type)
        task_embed = self.task_embedding(torch.tensor([task_id], device=device).expand(batch_size))
        
        # 모든 조건 융합
        all_embeds = torch.cat([lyrics_embed, caption_embed, reference_embed, task_embed], dim=-1)
        fused_embed = self.fusion(all_embeds)
        
        return fused_embed.unsqueeze(1)  # (batch, 1, d_model)
    
    def _get_task_id(self, task_type: Optional[str]) -> int:
        """태스크 타입을 ID로 변환"""
        task_map = {
            'SONG': 0,
            'INST': 1, 
            'COVER': 2,
            None: 3
        }
        return task_map.get(task_type, 3)


class LyricsEncoder(nn.Module):
    """가사 인코더"""
    
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_lyrics_length, config.d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(6)  # 작은 인코더
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        self.pool = nn.MultiheadAttention(config.d_model, 8, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
    
    def forward(self, lyrics: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = lyrics.shape
        
        # 임베딩
        x = self.embed(lyrics)
        x = x + self.pos_embed[:, :seq_len]
        
        # 인코딩
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        # 어텐션 풀링
        query = self.query.expand(batch_size, -1, -1)
        out, _ = self.pool(query, x, x, key_padding_mask=~mask if mask is not None else None)
        
        return out.squeeze(1)


class CaptionEncoder(nn.Module):
    """캡션 인코더 (MusicCaps 스타일)"""
    
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        # 실제로는 사전훈련된 텍스트 인코더 사용
        # 여기서는 간단한 구현
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
    
    def forward(self, captions: List[str]) -> torch.Tensor:
        # 간단한 구현 - 실제로는 토크나이저 + 사전훈련 모델 필요
        batch_size = len(captions) if captions else 1
        device = next(self.parameters()).device
        
        # 더미 구현
        return torch.randn(batch_size, self.embed.embedding_dim, device=device)


class ReferenceEncoder(nn.Module):
    """참조 오디오 인코더"""
    
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(config.latent_channels, config.d_model // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.d_model // 4, config.d_model // 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(config.d_model // 2, config.d_model, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.proj = nn.Linear(config.d_model, config.d_model)
    
    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        if reference is None:
            batch_size = 1
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.proj.out_features, device=device)
        
        # 컨볼루션 처리
        x = self.conv_layers(reference)  # (batch, d_model, 1)
        x = x.squeeze(-1)  # (batch, d_model)
        
        return self.proj(x)


def create_lyro_generator(config: GeneratorConfig = None) -> LyroGenerator:
    """LYRO Generator 생성"""
    if config is None:
        config = GeneratorConfig()
    
    return LyroGenerator(config)
