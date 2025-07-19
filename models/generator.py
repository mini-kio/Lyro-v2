import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from einops import rearrange, repeat
from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    d_ff: int = 4096
    d_state: int = 64
    latent_channels: int = 16
    latent_time_steps: int = 128
    flow_steps: int = 50
    cfg_scale: float = 7.5
    dropout: float = 0.1


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return torch.cos(freqs), torch.sin(freqs)


class S6Layer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * 2
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, 4, groups=self.d_inner, padding=3)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A.float()))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_inner)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        x_conv = rearrange(x_inner, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[..., :L]
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)
        
        dt = F.softplus(self.dt_proj(x_conv))
        BC = self.x_proj(x_conv)
        B_ssm, C = BC.chunk(2, dim=-1)
        
        A = -torch.exp(self.A_log.float())
        dA = torch.einsum('bld,dn->bldn', dt, A)
        dB = torch.einsum('bld,bln->bldn', dt, B_ssm)
        
        y = torch.einsum('bldn,bln->bld', dB, C) + x_conv * self.D
        y = self.norm(y) * F.silu(z)
        return self.out_proj(y)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionalEmbedding(self.d_head)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        cos, sin = self.rope(x)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.s6_layer = S6Layer(d_model)
        self.attn_layer = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.s6_layer(self.norm1(x))
        x = x + self.attn_layer(self.norm2(x))
        x = x + self.ffn(self.norm3(x))
        return x


class ConditionProcessor(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.text_proj = nn.Linear(768, d_model)
        self.reference_proj = nn.Linear(16 * 128, d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)
    
    def forward(self, text_embed: Optional[torch.Tensor] = None, 
                reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        batch_size = 1
        
        if text_embed is not None:
            batch_size = text_embed.shape[0]
            text_cond = self.text_proj(text_embed)
        else:
            text_cond = torch.zeros(batch_size, self.text_proj.out_features, device=device)
        
        if reference is not None:
            ref_cond = self.reference_proj(reference.flatten(1))
        else:
            ref_cond = torch.zeros(batch_size, self.reference_proj.out_features, device=device)
        
        return self.fusion(torch.cat([text_cond, ref_cond], dim=-1))


class LyroGenerator(nn.Module):
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        
        self.latent_embed = nn.Linear(config.latent_channels, config.d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        self.condition_processor = ConditionProcessor(config.d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.d_ff, config.n_heads, config.dropout) 
            for _ in range(config.n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.latent_channels)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = latents.shape[0]
        
        latents = rearrange(latents, 'b c t -> b t c')
        x = self.latent_embed(latents)
        
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.dim() == 1 and timesteps.shape[0] != batch_size:
            timesteps = timesteps.expand(batch_size)
        
        time_embed = self.time_embed(timesteps.unsqueeze(-1)).unsqueeze(1)
        x = x + time_embed
        
        condition_embed = self.condition_processor(text_embed, reference)
        x = x + condition_embed.unsqueeze(1)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        return rearrange(x, 'b t c -> b c t')
    
    def training_loss(
        self,
        latents: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = latents.shape[0]
        device = latents.device
        
        t = torch.rand(batch_size, device=device)
        noise = torch.randn_like(latents)
        
        t_expanded = t.view(-1, 1, 1)
        xt = (1 - t_expanded) * noise + t_expanded * latents
        target_v = latents - noise
        
        predicted_v = self.forward(xt, t, text_embed, reference)
        flow_loss = F.mse_loss(predicted_v, target_v)
        
        return {'flow_loss': flow_loss}
    
    @torch.no_grad()
    def generate(
        self,
        shape: Tuple[int, int, int],
        text_embed: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        cfg_scale: float = 7.5
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device)
        
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t = torch.full((shape[0],), timesteps[i], device=device)
            
            if cfg_scale > 1.0:
                pred_cond = self.forward(x, t, text_embed, reference)
                pred_uncond = self.forward(x, t, None, None)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred = self.forward(x, t, text_embed, reference)
            
            dt = timesteps[i] - timesteps[i + 1]
            x = x - dt * pred
        
        return x
    


def create_lyro_generator(config: GeneratorConfig = None) -> LyroGenerator:
    if config is None:
        config = GeneratorConfig()
    return LyroGenerator(config)