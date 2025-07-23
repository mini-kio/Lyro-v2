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
        
        # REPA 관련 추가
        self.enable_repa = True
        self.repa_layer_idx = 7  # 8번째 레이어
        self.repa_projection = nn.Linear(config.d_model, 1024)
        
        # 모델들 (lazy loading)
        self._mert_model = None
        self._wav2vec2_model = None
        
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
    
    def _get_mert_model(self):
        """MERT 모델 lazy loading"""
        if self._mert_model is None:
            try:
                from transformers import AutoModel
                self._mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M")
                self._mert_model.eval()
                for param in self._mert_model.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"Warning: MERT model not available: {e}")
                self.enable_repa = False
        return self._mert_model
    
    def _get_wav2vec2_model(self):
        """wav2vec2-xls-r-300m 모델 lazy loading"""
        if self._wav2vec2_model is None:
            try:
                from transformers import AutoModel
                self._wav2vec2_model = AutoModel.from_pretrained("facebook/wav2vec2-xls-r-300m")
                self._wav2vec2_model.eval()
                for param in self._wav2vec2_model.parameters():
                    param.requires_grad = False
            except Exception as e:
                print(f"Warning: wav2vec2 model not available: {e}")
                self.enable_repa = False
        return self._wav2vec2_model
    
    def _extract_audio_features(
        self, 
        audio_batch: torch.Tensor, 
        model: nn.Module, 
        target_sr: int = 16000,
        target_dim: int = 1024
    ) -> Optional[torch.Tensor]:
        """오디오 feature 추출 (배치 패딩 처리로 GPU 효율화)"""
        if model is None:
            return None
            
        device = audio_batch.device
        batch_size = audio_batch.shape[0]
        
        try:
            model.to(device)
            
            with torch.no_grad():
                # 배치 전처리 - 모노로 변환 및 길이 통일
                processed_batch = []
                max_length = 0
                
                for i in range(batch_size):
                    audio = audio_batch[i]
                    
                    # 모노로 변환
                    if audio.dim() > 1:
                        audio = audio.mean(dim=0)
                    
                    # 빈 오디오 처리
                    if audio.numel() == 0:
                        audio = torch.zeros(target_sr, device=device)  # 1초 기본값
                    
                    processed_batch.append(audio)
                    max_length = max(max_length, audio.shape[0])
                
                # 패딩으로 배치 생성
                padded_batch = torch.zeros(batch_size, max_length, device=device)
                attention_mask = torch.zeros(batch_size, max_length, device=device)
                
                for i, audio in enumerate(processed_batch):
                    length = audio.shape[0]
                    padded_batch[i, :length] = audio
                    attention_mask[i, :length] = 1.0
                
                try:
                    # 배치 단위로 모델 실행
                    outputs = model(padded_batch)
                    
                    if hasattr(outputs, 'last_hidden_state'):
                        features = outputs.last_hidden_state  # (B, T, D)
                    else:
                        features = outputs[0] if isinstance(outputs, tuple) else outputs
                    
                    # 어텐션 마스크 적용하여 평균 계산
                    if features.dim() == 3:  # (B, T, D)
                        # 마스크된 평균 계산
                        mask_expanded = attention_mask.unsqueeze(-1).expand_as(features)
                        masked_features = features * mask_expanded
                        lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                        features = masked_features.sum(dim=1) / lengths  # (B, D)
                    elif features.dim() == 2:  # (B, D)
                        pass  # 이미 올바른 형태
                    else:
                        features = features.mean(dim=1)
                    
                    # 차원 맞춤
                    if features.shape[-1] != target_dim:
                        if features.shape[-1] < target_dim:
                            pad_size = target_dim - features.shape[-1]
                            features = F.pad(features, (0, pad_size))
                        else:
                            features = features[..., :target_dim]
                    
                    return features  # (B, target_dim)
                    
                except Exception as e:
                    print(f"Batch model inference failed: {e}, falling back to individual processing")
                    # Fallback to individual processing
                    return self._extract_audio_features_individual(audio_batch, model, target_sr, target_dim)
            
        except Exception as e:
            print(f"Batch feature extraction failed: {e}")
            return None
    
    def _extract_audio_features_individual(
        self, 
        audio_batch: torch.Tensor, 
        model: nn.Module, 
        target_sr: int = 16000,
        target_dim: int = 1024
    ) -> Optional[torch.Tensor]:
        """개별 오디오 처리 (fallback)"""
        device = audio_batch.device
        batch_size = audio_batch.shape[0]
        features_list = []
        
        with torch.no_grad():
            for i in range(batch_size):
                audio = audio_batch[i]
                
                # 오디오 전처리
                if audio.dim() > 1:
                    audio = audio.mean(dim=0)  # 모노로 변환
                
                # 모델 입력
                if audio.numel() == 0:
                    features_list.append(torch.zeros(1, target_dim, device=device))
                    continue
                
                try:
                    outputs = model(audio.unsqueeze(0))
                    if hasattr(outputs, 'last_hidden_state'):
                        feat = outputs.last_hidden_state.mean(dim=1)  # (1, D)
                    else:
                        feat = outputs[0].mean(dim=1) if isinstance(outputs, tuple) else outputs.mean(dim=1)
                    
                    # 차원 맞춤
                    if feat.shape[-1] != target_dim:
                        if feat.shape[-1] < target_dim:
                            pad_size = target_dim - feat.shape[-1]
                            feat = F.pad(feat, (0, pad_size))
                        else:
                            feat = feat[..., :target_dim]
                    
                    features_list.append(feat)
                    
                except Exception as e:
                    print(f"Feature extraction failed for sample {i}: {e}")
                    features_list.append(torch.zeros(1, target_dim, device=device))
        
        return torch.cat(features_list, dim=0)  # (B, target_dim)
    
    def compute_repa_loss(
        self, 
        repa_features: torch.Tensor, 
        training_audio: torch.Tensor
    ) -> torch.Tensor:
        """REPA loss 계산 (wav2vec2 사용)"""
        if not self.enable_repa or repa_features is None or training_audio is None:
            return torch.tensor(0.0, device=repa_features.device if repa_features is not None else 'cpu')
        
        device = repa_features.device
        
        try:
            # MERT features
            mert_model = self._get_mert_model()
            mert_features = self._extract_audio_features(
                training_audio, mert_model, target_sr=24000, target_dim=1024
            )
            
            # wav2vec2 features
            wav2vec2_model = self._get_wav2vec2_model()
            wav2vec2_features = self._extract_audio_features(
                training_audio, wav2vec2_model, target_sr=16000, target_dim=1024
            )
            
            # 코사인 유사도 loss 계산
            total_loss = torch.tensor(0.0, device=device)
            
            if mert_features is not None:
                mert_similarity = F.cosine_similarity(repa_features, mert_features, dim=-1)
                total_loss += 1 - mert_similarity.mean()
            
            if wav2vec2_features is not None:
                wav2vec2_similarity = F.cosine_similarity(repa_features, wav2vec2_features, dim=-1)
                total_loss += 0.5 * (1 - wav2vec2_similarity.mean())  # 0.5 가중치
            
            return total_loss
            
        except Exception as e:
            print(f"REPA loss computation failed: {e}")
            return torch.tensor(0.0, device=device)
    
    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
        training_audio: Optional[torch.Tensor] = None
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
            
            # REPA features 추출 (훈련 시에만)
            if (i == self.repa_layer_idx and 
                self.training and 
                training_audio is not None and 
                self.enable_repa):
                repa_features = self.repa_projection(x.mean(dim=1))
        
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
        training_audio: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = latents.shape[0]
        device = latents.device
        
        t = torch.rand(batch_size, device=device)
        
        # 입력 기반 결정적 노이즈 생성 (fallback 시)
        if training_audio is not None:
            noise = self._create_deterministic_noise(latents.shape, training_audio)
        elif text_embed is not None:
            noise = self._create_deterministic_noise(latents.shape, text_embed)
        elif reference is not None:
            noise = self._create_deterministic_noise(latents.shape, reference)
        else:
            # 마지막 fallback - latents 자체를 시드로
            noise = self._create_deterministic_noise(latents.shape, latents)
        
        t_expanded = t.view(-1, 1, 1)
        xt = (1 - t_expanded) * noise + t_expanded * latents
        target_v = latents - noise
        
        predicted_v, repa_features = self.forward(xt, t, text_embed, reference, training_audio)
        flow_loss = F.mse_loss(predicted_v, target_v)
        repa_loss = self.compute_repa_loss(repa_features, training_audio)
        
        return {
            'flow_loss': flow_loss,
            'repa_loss': repa_loss,
            'total_loss': flow_loss + repa_loss
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