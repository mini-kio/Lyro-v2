# lyro/models/generator.py
"""
LYRO Generator - CFG 통합 1.5B 파라미터 모델
SSM + Flow Matching + CFG 샘플링
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from einops import rearrange, repeat
from dataclasses import dataclass

from .sampling import FlowMatchingSampler, create_sampler


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
    
    # Flow Matching + CFG 설정
    flow_steps: int = 50
    cfg_scale: float = 15.0
    cfg_min_scale: float = 3.0
    cfg_active_ratio: float = 0.5
    
    # 기타
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotary Position Embedding 적용"""
    def rotate_half(x):
        """벡터의 절반을 회전"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
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


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
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
        
        # A 매트릭스
        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A.float()))
        
        # D 매트릭스
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 출력 투영
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        
        # 정규화
        self.norm = nn.LayerNorm(self.d_inner, eps=config.layer_norm_eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        
        # 입력 투영
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # 컨볼루션
        x_conv = rearrange(x_inner, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[..., :seq_len]
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        
        # 활성화
        x_conv = F.silu(x_conv)
        
        # SSM 연산
        dt = F.softplus(self.dt_proj(x_conv))
        BC = self.x_proj(x_conv)
        B, C = BC.chunk(2, dim=-1)
        
        # A 매트릭스
        A = -torch.exp(self.A_log.float())
        
        # 이산화
        dA = torch.einsum('bld,dn->bldn', dt, A)
        dB = torch.einsum('bld,bln->bldn', dt, B)
        
        # 상태 공간 연산
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
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.rope = RotaryPositionalEmbedding(self.d_head)
        self.scale = self.d_head ** -0.5
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # RoPE 적용
        cos, sin = self.rope(x)
        
        # cos, sin을 q, k의 shape에 맞게 확장
        # cos, sin: (seq_len, d_head//2) -> (1, 1, seq_len, d_head//2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # d_head가 홀수인 경우를 대비해 repeat으로 확장
        if cos.shape[-1] != q.shape[-1]:
            cos = cos.repeat(1, 1, 1, 2)[:, :, :, :q.shape[-1]]
            sin = sin.repeat(1, 1, 1, 2)[:, :, :, :q.shape[-1]]
        
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        
        return self.o_proj(out)


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


class ConditionProcessor(nn.Module):
    """조건 처리 모듈"""
    
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        
        # 간단한 조건 인코더들
        self.lyrics_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.caption_proj = nn.Linear(768, config.d_model)  # 사전훈련된 텍스트 인코더 출력
        self.reference_proj = nn.Linear(config.latent_channels * config.latent_time_steps, config.d_model)
        
        # 태스크 임베딩
        self.task_embedding = nn.Embedding(10, config.d_model)
        
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
        device = next(self.parameters()).device
        
        # 실제 배치 크기 결정 (latent 텐서에서 추론)
        if lyrics is not None:
            actual_batch_size = lyrics.shape[0]
        else:
            actual_batch_size = batch_size
        
        # 가사 처리
        if lyrics is not None:
            lyrics_embed = self.lyrics_embed(lyrics)
            if lyrics_mask is not None:
                lyrics_embed = lyrics_embed * lyrics_mask.unsqueeze(-1)
            lyrics_embed = lyrics_embed.mean(dim=1)
        else:
            lyrics_embed = torch.zeros(actual_batch_size, self.config.d_model, device=device)
        
        # 캡션 처리 (간단한 더미 구현)
        if captions is not None:
            # 실제로는 사전훈련된 텍스트 인코더 필요
            caption_embed = torch.randn(actual_batch_size, 768, device=device)
            caption_embed = self.caption_proj(caption_embed)
        else:
            caption_embed = torch.zeros(actual_batch_size, self.config.d_model, device=device)
        
        # 참조 오디오 처리
        if reference_audio is not None:
            ref_flat = reference_audio.flatten(1)
            reference_embed = self.reference_proj(ref_flat)
        else:
            reference_embed = torch.zeros(actual_batch_size, self.config.d_model, device=device)
        
        # 태스크 임베딩
        task_id = self._get_task_id(task_type)
        task_embed = self.task_embedding(torch.tensor([task_id], device=device).expand(actual_batch_size))
        
        # 모든 조건 융합
        all_embeds = torch.cat([lyrics_embed, caption_embed, reference_embed, task_embed], dim=-1)
        fused_embed = self.fusion(all_embeds)
        
        return fused_embed.unsqueeze(1)
    
    def _get_task_id(self, task_type: Optional[str]) -> int:
        task_map = {'SONG': 0, 'INST': 1, 'COVER': 2, None: 3}
        return task_map.get(task_type, 3)


class LyroGenerator(nn.Module):
    """
    LYRO Generator with CFG Integration (1.5B parameters)
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
        
        # CFG 샘플러 초기화
        self.sampler = FlowMatchingSampler(
            cfg_scale=config.cfg_scale,
            cfg_min_scale=config.cfg_min_scale,
            cfg_active_ratio=config.cfg_active_ratio,
            steps=config.flow_steps,
            device="cpu"  # 나중에 모델과 함께 디바이스 이동
        )
        
        # 파라미터 초기화
        self.apply(self._init_weights)
        
        print(f"LyroGenerator initialized with {self.count_parameters():,} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def count_parameters(self) -> int:
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
        Forward pass for velocity prediction
        """
        batch_size = latents.shape[0]
        
        # 잠재 벡터를 시퀀스 형태로 변환 - 차원 확인
        if latents.dim() != 3:
            raise ValueError(f"Expected latents to have 3 dimensions (B, C, T), got {latents.shape}")
        
        # (B, C, T) -> (B, T, C)
        latents = rearrange(latents, 'b c t -> b t c')
        
        # 잠재 임베딩
        x = self.latent_embed(latents)
        
        # 시간 임베딩 - timesteps 차원 확인
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.dim() == 1 and timesteps.shape[0] != batch_size:
            timesteps = timesteps.expand(batch_size)
        
        time_embed = self.time_embed(timesteps.unsqueeze(-1))
        time_embed = time_embed.unsqueeze(1)
        x = x + time_embed
        
        # 조건 처리 및 임베딩
        condition_embed = self.condition_processor(
            lyrics=lyrics,
            lyrics_mask=lyrics_mask,
            captions=captions,
            reference_audio=reference_audio,
            task_type=task_type,
            batch_size=batch_size
        )
        
        # condition_embed shape: (B, 1, d_model) -> (B, T, d_model)로 브로드캐스트
        seq_len = x.shape[1]
        condition_embed = condition_embed.expand(-1, seq_len, -1)
        
        x = x + condition_embed
        
        # Transformer 레이어들
        for layer in self.layers:
            x = layer(x)
        
        # 출력 투영
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        # 원래 형태로 복원 - 차원 검증
        if x.dim() != 3:
            raise ValueError(f"Expected output to have 3 dimensions, got {x.shape}")
        
        # (B, T, C) -> (B, C, T)
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
        """Training loss calculation"""
        batch_size = latents.shape[0]
        device = latents.device
        
        # latents 차원 검증
        if latents.dim() != 3:
            raise ValueError(f"Expected latents to have 3 dimensions (B, C, T), got {latents.shape}")
        
        # 시간 샘플링
        t = torch.rand(batch_size, device=device)
        
        # 노이즈 샘플링
        noise = torch.randn_like(latents)
        
        # Forward process (Flow Matching)
        t_expanded = t.view(-1, 1, 1)
        xt = (1 - t_expanded) * noise + t_expanded * latents
        target_v = latents - noise
        
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
    def generate_with_cfg(
        self,
        shape: Tuple[int, int, int],
        lyrics: Optional[torch.Tensor] = None,
        lyrics_mask: Optional[torch.Tensor] = None,
        captions: Optional[List[str]] = None,
        reference_audio: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None,
        cfg_scale: Optional[float] = None,
        num_steps: Optional[int] = None,
        device: Optional[torch.device] = None,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        CFG를 사용한 고품질 생성
        """
        if device is None:
            device = next(self.parameters()).device
        
        # 조건 딕셔너리 준비
        conditions = {
            'lyrics': lyrics,
            'lyrics_mask': lyrics_mask,
            'captions': captions,
            'reference_audio': reference_audio,
            'task_type': task_type
        }
        
        # 샘플러 디바이스 업데이트
        if self.sampler.scheduler.t.device != device:
            self.sampler = FlowMatchingSampler(
                cfg_scale=cfg_scale or self.config.cfg_scale,
                steps=num_steps or self.config.flow_steps,
                device=device
            )
        
        # CFG 샘플링 실행
        generated = self.sampler.sample(
            model=self,
            shape=shape,
            conditions=conditions,
            device=device,
            verbose=verbose
        )
        
        return generated
    
    @torch.no_grad()
    def generate_fast(
        self,
        shape: Tuple[int, int, int],
        lyrics: Optional[torch.Tensor] = None,
        lyrics_mask: Optional[torch.Tensor] = None,
        captions: Optional[List[str]] = None,
        reference_audio: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None,
        num_steps: int = 20,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        빠른 생성 (CFG 없음)
        """
        if device is None:
            device = next(self.parameters()).device
        
        # 초기 노이즈
        x = torch.randn(shape, device=device)
        
        # 시간 스케줄
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t = torch.full((shape[0],), timesteps[i], device=device)
            
            # 예측
            pred = self.forward(
                latents=x,
                timesteps=t,
                lyrics=lyrics,
                lyrics_mask=lyrics_mask,
                captions=captions,
                reference_audio=reference_audio,
                task_type=task_type
            )
            
            # 업데이트
            dt = timesteps[i] - timesteps[i + 1]
            x = x - dt * pred
        
        return x


def create_lyro_generator(config: GeneratorConfig = None) -> LyroGenerator:
    """LYRO Generator 생성"""
    if config is None:
        config = GeneratorConfig()
    
    return LyroGenerator(config)


def load_pretrained_generator(
    checkpoint_path: str,
    config: GeneratorConfig = None,
    device: str = "auto"
) -> LyroGenerator:
    """프리트레인된 Generator 로드"""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config is None:
        config = GeneratorConfig()
    
    model = create_lyro_generator(config)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device).eval()
    
    print(f"Loaded pretrained generator from {checkpoint_path}")
    return model