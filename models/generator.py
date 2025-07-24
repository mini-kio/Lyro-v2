import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import hashlib
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
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[-2]
        t = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos_emb = torch.cos(freqs)
        sin_emb = torch.sin(freqs)
        # Ensure we have the right dimension
        if cos_emb.shape[-1] < self.dim:
            cos_emb = torch.cat([cos_emb, cos_emb], dim=-1)[:, :self.dim]
            sin_emb = torch.cat([sin_emb, sin_emb], dim=-1)[:, :self.dim]
        return cos_emb, sin_emb


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
        
        # Reference latent processing with 1D-Conv for temporal information
        self.reference_conv = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 16, kernel_size=3, padding=1)  # Back to original channels
        )
        
        # Cross-attention for reference latent temporal modeling
        self.reference_cross_attn = nn.MultiheadAttention(
            embed_dim=16, 
            num_heads=4, 
            batch_first=True
        )
        
        # Project processed reference to d_model
        self.reference_proj = nn.Linear(16 * 128, d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)
    
    def forward(self, text_embed: Optional[torch.Tensor] = None, 
                reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        batch_size = 1
        
        if text_embed is not None:
            batch_size = text_embed.shape[0]
        elif reference is not None:
            batch_size = reference.shape[0]
            
        if text_embed is not None:
            text_cond = self.text_proj(text_embed)
        else:
            text_cond = torch.zeros(batch_size, self.text_proj.out_features, device=device)
        
        if reference is not None:
            if reference.shape[0] != batch_size:
                # Handle batch size mismatch
                reference = reference[:batch_size] if reference.shape[0] > batch_size else reference.expand(batch_size, -1, -1)
            
            # Process reference with 1D convolution for temporal info
            # reference: (B, C, T) -> apply conv1d
            ref_conv = self.reference_conv(reference)  # (B, 16, 128)
            
            # Prepare for cross-attention: (B, T, C)
            ref_for_attn = ref_conv.transpose(1, 2)  # (B, 128, 16)
            
            # Self-attention on reference for temporal modeling
            ref_attended, _ = self.reference_cross_attn(
                query=ref_for_attn,
                key=ref_for_attn,
                value=ref_for_attn
            )  # (B, 128, 16)
            
            # Back to (B, C, T) for projection
            ref_processed = ref_attended.transpose(1, 2)  # (B, 16, 128)
            
            # Project to d_model
            ref_cond = self.reference_proj(ref_processed.flatten(1))
        else:
            ref_cond = torch.zeros(batch_size, self.reference_proj.out_features, device=device)
        
        return self.fusion(torch.cat([text_cond, ref_cond], dim=-1))


class LyroGenerator(nn.Module):
    def __init__(self, config: GeneratorConfig, enable_compile: bool = True):
        super().__init__()
        self.config = config
        self.enable_compile = enable_compile
        
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
        
        # REPA 관련 추가 (ACE-Step 스타일)
        self.enable_repa = True
        self.repa_layer_idx = 7  # 8번째 레이어
        self.repa_proj = nn.Linear(config.d_model, 1024)  # 1×1 conv 대신 Linear
        
        # REPA 관련 설정만 유지
        
        self.apply(self._init_weights)
        
        # torch.compile 적용
        if self.enable_compile:
            self._apply_compile_optimization()
    
    def _apply_compile_optimization(self):
        """torch.compile 최적화 적용"""
        try:
            import torch._dynamo as dynamo
            
            # Core transformer blocks 컴파일
            for i, layer in enumerate(self.layers):
                try:
                    self.layers[i] = torch.compile(
                        layer,
                        mode="reduce-overhead",
                        fullgraph=False,
                        dynamic=True
                    )
                except Exception as e:
                    print(f"Warning: Failed to compile layer {i}: {str(e)}")
            
            # Condition processor 컴파일
            try:
                self.condition_processor = torch.compile(
                    self.condition_processor,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True
                )
            except Exception as e:
                print(f"Warning: Failed to compile condition_processor: {str(e)}")
            
            print("torch.compile optimization applied successfully")
            
        except ImportError:
            print("torch.compile not available, skipping optimization")
        except Exception as e:
            print(f"Failed to apply torch.compile: {str(e)}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    
    
    def temporal_align(self, features_list):
        """ACE-Step 스타일 temporal alignment with nearest+lowpass"""
        if not features_list or len(features_list) == 0:
            return []
        
        max_time_steps = max(feat.shape[1] for feat in features_list if feat is not None)
        
        aligned_features = []
        for feat in features_list:
            if feat is None:
                aligned_features.append(None)
                continue
                
            B, T, D = feat.shape
            if T == max_time_steps:
                aligned_features.append(feat)
            else:
                feat_permuted = feat.permute(0, 2, 1)  # (B, D, T)
                
                if T < max_time_steps:  # 업샘플링
                    # nearest + low-pass
                    feat_nearest = F.interpolate(feat_permuted, size=max_time_steps, mode='nearest')
                    
                    # Gaussian smoothing
                    kernel_size = min(5, max_time_steps // T)
                    if kernel_size > 1 and kernel_size % 2 == 0:
                        kernel_size += 1
                    
                    if kernel_size > 1:
                        sigma = kernel_size / 6.0
                        x = torch.arange(kernel_size, device=feat.device, dtype=feat.dtype)
                        x = x - kernel_size // 2
                        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
                        kernel = kernel / kernel.sum()
                        kernel = kernel.view(1, 1, -1).expand(D, 1, -1)
                        
                        feat_aligned = F.conv1d(feat_nearest, kernel, padding=kernel_size//2, groups=D)
                    else:
                        feat_aligned = feat_nearest
                else:  # 다운샘플링
                    feat_aligned = F.interpolate(feat_permuted, size=max_time_steps, mode='linear', align_corners=False)
                
                feat_aligned = feat_aligned.permute(0, 2, 1)
                aligned_features.append(feat_aligned)
        
        return aligned_features

    def compute_repa_loss(
        self, 
        repa_features: torch.Tensor, 
        alignment_features: tuple = None,
        current_step: int = 0
    ) -> torch.Tensor:
        """
        ACE-Step 스타일 REPA loss 계산
        MERT(음악) + mHuBERT(가사) 두 표현을 동시에 맞춰 
        "가사 명료도 ↑ + 음악성 보존"을 노림
        """
        if not self.enable_repa or repa_features is None:
            return torch.tensor(0.0, device=repa_features.device if repa_features is not None else 'cpu')
        
        if alignment_features is None or len(alignment_features) != 2:
            return torch.tensor(0.0, device=repa_features.device)
        
        h_mert, h_mhubert = alignment_features
        if h_mert is None and h_mhubert is None:
            return torch.tensor(0.0, device=repa_features.device)
        
        device = repa_features.device
        
        try:
            # ACE-Step 스타일 dynamic weight 계산
            from ..models.losses import TrainingScheduler
            training_scheduler = TrainingScheduler()
            mhubert_weight = training_scheduler.get_mhubert_weight(current_step)
            mert_weight = 1.0  # MERT는 상시 1.0 유지
            
            # 3) h_mert, h_mhubert를 각각 75Hz→T′, 50Hz→T′로 선형보간
            h_dit = repa_features  # (B, T, 1024)
            h_dit, h_mert_aligned, h_mhubert_aligned = self.temporal_align([h_dit, h_mert, h_mhubert])
            
            total_loss = torch.tensor(0.0, device=device)
            
            # 4) 프레임별 cos sim 평균
            if h_mert_aligned is not None:
                h_dit_norm = F.normalize(h_dit, dim=-1, eps=1e-6)
                h_mert_norm = F.normalize(h_mert_aligned, dim=-1, eps=1e-6)
                mert_similarity = F.cosine_similarity(h_dit_norm, h_mert_norm, dim=-1)
                mert_loss = 1 - mert_similarity.mean()
                total_loss += mert_weight * mert_loss
            
            if h_mhubert_aligned is not None:
                h_dit_norm = F.normalize(h_dit, dim=-1, eps=1e-6)
                h_mhubert_norm = F.normalize(h_mhubert_aligned, dim=-1, eps=1e-6)
                mhubert_similarity = F.cosine_similarity(h_dit_norm, h_mhubert_norm, dim=-1)
                mhubert_loss = 1 - mhubert_similarity.mean()
                total_loss += mhubert_weight * mhubert_loss
            
            return total_loss
            
        except Exception as e:
            print(f"REPA loss computation failed: {e}")
            return torch.tensor(0.0, device=device)
    
    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        
        # 레이어 처리 및 REPA features 추출
        repa_features = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # REPA features 추출 (훈련 시에만) - ACE-Step 스타일
            if (i == self.repa_layer_idx and 
                self.training and 
                self.enable_repa):
                # 1) repa_layer의 hidden states: (B, T, d_model)
                h_dit = x  # (B, T, d_model)
                # 2) 1×1 conv or linear로 1024차원 투영
                repa_features = self.repa_proj(h_dit)  # (B, T, 1024)
        
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        return rearrange(x, 'b t c -> b c t'), repa_features
    
    def _create_deterministic_noise(self, shape: Tuple[int, ...], input_data: torch.Tensor) -> torch.Tensor:
        """입력 데이터 해시를 사용해 결정적 노이즈 생성"""
        device = input_data.device
        
        # 입력 데이터를 바이트로 변환
        input_bytes = input_data.detach().cpu().numpy().tobytes()
        
        # SHA256 해시 생성
        hash_obj = hashlib.sha256(input_bytes)
        seed = int.from_bytes(hash_obj.digest()[:8], byteorder='big')
        
        # 시드 기반 노이즈 생성
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        
        return torch.randn(shape, device=device, generator=generator)

    def training_loss(
        self,
        latents: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
        alignment_features: Optional[tuple] = None,
        current_step: int = 0
    ) -> Dict[str, torch.Tensor]:
        batch_size = latents.shape[0]
        device = latents.device
        
        t = torch.rand(batch_size, device=device)
        
        # 입력 기반 결정적 노이즈 생성
        if text_embed is not None:
            noise = self._create_deterministic_noise(latents.shape, text_embed)
        elif reference is not None:
            noise = self._create_deterministic_noise(latents.shape, reference)
        else:
            # 마지막 fallback - latents 자체를 시드로
            noise = self._create_deterministic_noise(latents.shape, latents)
        
        t_expanded = t.view(-1, 1, 1)
        xt = (1 - t_expanded) * noise + t_expanded * latents
        target_v = latents - noise
        
        predicted_v, repa_features = self.forward(xt, t, text_embed, reference)
        flow_loss = F.mse_loss(predicted_v, target_v) * 50.0  # Flow≈1e-2 -> 스케일 조정
        repa_loss = self.compute_repa_loss(repa_features, alignment_features, current_step)
        
        total_loss = flow_loss + repa_loss
        
        return {
            'flow_loss': flow_loss,
            'repa_loss': repa_loss,
            'total_loss': total_loss
        }
    
    @torch.no_grad()
    def generate(
        self,
        shape: Tuple[int, int, int],
        text_embed: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        cfg_scale: float = 7.5,
        use_deterministic_noise: bool = False
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        
        if use_deterministic_noise and (text_embed is not None or reference is not None):
            # 조건 입력 기반 결정적 노이즈
            if text_embed is not None:
                x = self._create_deterministic_noise(shape, text_embed)
            elif reference is not None:
                x = self._create_deterministic_noise(shape, reference)
            else:
                x = torch.randn(shape, device=device)
        else:
            x = torch.randn(shape, device=device)
        
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t = torch.full((shape[0],), timesteps[i], device=device)
            
            if cfg_scale > 1.0:
                pred_cond, _ = self.forward(x, t, text_embed, reference)
                pred_uncond, _ = self.forward(x, t, None, None)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred, _ = self.forward(x, t, text_embed, reference)
            
            dt = timesteps[i] - timesteps[i + 1]
            x = x - dt * pred
        
        return x
    


def create_lyro_generator(config: GeneratorConfig = None, enable_compile: bool = True) -> LyroGenerator:
    if config is None:
        config = GeneratorConfig()
    return LyroGenerator(config, enable_compile=enable_compile)