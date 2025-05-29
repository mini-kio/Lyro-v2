# lyro/ssm/model.py
"""
Complete Lyro SSM Implementation
Advanced State Space Model with Linear Attention for Long Sequence Processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
import numpy as np


# ==================== State Space Model Core ====================

class StateSpaceKernel(nn.Module):
    """
    Advanced State Space Model Kernel
    Implements efficient convolution-based SSM without external dependencies
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = True,
        conv_bias: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        
        dt_rank = dt_rank or math.ceil(d_model / 16)
        
        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        
        # Initialize dt projection
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SSM
        
        Args:
            x: (B, L, D) input sequence
        Returns:
            output: (B, L, D) processed sequence
        """
        B, L, D = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)
        
        # Convolution (causal)
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[..., :L]  # Truncate to original length
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # Activation
        x = F.silu(x)
        
        # SSM computation
        x = self.ssm(x)
        
        # Gating
        y = x * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Core SSM computation"""
        B, L, D = x.shape
        
        # Compute dt, B, C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        
        # Compute dt
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt + self.dt_proj.bias)
        
        # Compute A
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretization
        A_discrete, B_discrete = self.discretize(A, B, dt)
        
        # SSM step
        y = self.ssm_step(x, A_discrete, B_discrete, C, self.D)
        
        return y
    
    def discretize(self, A, B, dt):
        """Discretize continuous SSM parameters"""
        # A: (d_inner, d_state)
        # B: (B, L, d_state) 
        # dt: (B, L, d_inner)
        
        B_batch, L, _ = dt.shape
        
        # Expand dt for broadcasting
        dt = dt.unsqueeze(-1)  # (B, L, d_inner, 1)
        A = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)
        
        # Zero-order hold discretization
        A_discrete = torch.exp(dt * A)  # (B, L, d_inner, d_state)
        B_discrete = (A_discrete - 1) / A * B.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        return A_discrete, B_discrete
    
    def ssm_step(self, x, A, B, C, D):
        """Perform SSM recurrence"""
        B_batch, L, d_inner = x.shape
        d_state = A.shape[-1]
        
        # Initialize state
        h = torch.zeros(B_batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        for t in range(L):
            # Current inputs
            x_t = x[:, t]  # (B, d_inner)
            A_t = A[:, t]  # (B, d_inner, d_state)
            B_t = B[:, t]  # (B, d_inner, d_state)
            C_t = C[:, t]  # (B, d_state)
            
            # State update
            h = A_t * h + B_t * x_t.unsqueeze(-1)
            
            # Output
            y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1) + D * x_t
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        
        return y


class SSMBlock(nn.Module):
    """Complete SSM Block with normalization and skip connections"""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        self.ssm = StateSpaceKernel(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input sequence
        Returns:
            output: (B, L, D) processed sequence
        """
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        return x + residual


# ==================== Advanced Multi-Scale SSM ====================

class MultiScaleSSM(nn.Module):
    """Multi-scale SSM for different temporal patterns"""
    
    def __init__(
        self,
        d_model: int,
        scales: List[int] = [1, 2, 4, 8],
        d_state: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.scales = scales
        self.ssm_blocks = nn.ModuleList([
            SSMBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=4 * scale,  # Larger conv for larger scales
                dropout=dropout,
            ) for scale in scales
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(d_model * len(scales), d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input with multiple SSM scales"""
        B, L, D = x.shape
        
        scale_outputs = []
        
        for scale, ssm_block in zip(self.scales, self.ssm_blocks):
            if scale == 1:
                # No downsampling
                scale_out = ssm_block(x)
            else:
                # Downsample, process, upsample
                # Simple downsampling
                x_down = x[:, ::scale, :]
                scale_out_down = ssm_block(x_down)
                
                # Upsample back to original length
                scale_out = F.interpolate(
                    scale_out_down.transpose(1, 2),
                    size=L,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            scale_outputs.append(scale_out)
        
        # Fuse multi-scale outputs
        fused = torch.cat(scale_outputs, dim=-1)
        output = self.fusion(fused)
        output = self.fusion_norm(output)
        
        return output


# ==================== Conditional Embeddings ====================

class AdvancedConditionalEmbedding(nn.Module):
    """Advanced conditioning with multiple modalities"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Task embeddings
        self.task_embed = nn.Embedding(10, d_model)
        
        # Time embeddings (sinusoidal)
        self.time_embed = SinusoidalEmbedding(d_model)
        
        # Text/Lyrics encoder
        self.text_encoder = nn.Sequential(
            nn.Embedding(32000, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Style prompt encoder
        self.style_encoder = nn.Sequential(
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # ICL reference encoder
        self.icl_encoder = nn.Sequential(
            nn.Conv1d(8, d_model // 4, 1),
            nn.GroupNorm(8, d_model // 4),
            nn.GELU(),
            nn.Conv1d(d_model // 4, d_model, 1),
        )
        
        # Cross-attention for ICL
        self.icl_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, conditions: Dict) -> torch.Tensor:
        """
        Args:
            conditions: Dict containing:
                - task_token: (B,) task IDs
                - time: (B,) time steps  
                - lyrics: (B, L_text) text tokens
                - style_prompt: (B, 512) style vectors
                - icl_reference: (B, 8, L_ref) reference latents
        """
        B = conditions['task_token'].shape[0]
        device = conditions['task_token'].device
        
        embeddings = []
        
        # 1. Task embedding
        task_emb = self.task_embed(conditions['task_token'])  # (B, D)
        embeddings.append(task_emb)
        
        # 2. Time embedding
        time_emb = self.time_embed(conditions['time'])  # (B, D)
        embeddings.append(time_emb)
        
        # 3. Text embedding
        if 'lyrics' in conditions and conditions['lyrics'] is not None:
            lyrics_tokens = conditions['lyrics']  # (B, L_text)
            # Mean pooling over sequence length
            text_emb = self.text_encoder(lyrics_tokens).mean(dim=1)  # (B, D)
        else:
            text_emb = torch.zeros(B, self.d_model, device=device)
        embeddings.append(text_emb)
        
        # 4. Style embedding
        if 'style_prompt' in conditions and conditions['style_prompt'] is not None:
            style_emb = self.style_encoder(conditions['style_prompt'])  # (B, D)
        else:
            style_emb = torch.zeros(B, self.d_model, device=device)
        embeddings.append(style_emb)
        
        # 5. ICL reference processing
        if 'icl_reference' in conditions and conditions['icl_reference'] is not None:
            icl_ref = conditions['icl_reference']  # (B, 8, L_ref)
            icl_encoded = self.icl_encoder(icl_ref)  # (B, D, L_ref)
            icl_encoded = icl_encoded.transpose(1, 2)  # (B, L_ref, D)
            
            # Cross-attention with other embeddings
            query = torch.stack(embeddings, dim=1)  # (B, 4, D)
            icl_attended, _ = self.icl_cross_attn(
                query, icl_encoded, icl_encoded
            )
            icl_emb = icl_attended.mean(dim=1)  # (B, D)
        else:
            icl_emb = torch.zeros(B, self.d_model, device=device)
        
        embeddings.append(icl_emb)
        
        # 6. Fusion
        combined = torch.cat(embeddings, dim=1)  # (B, 5*D)
        condition_emb = self.fusion(combined)  # (B, D)
        
        return condition_emb


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal position embedding"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B,) time steps
        Returns:
            emb: (B, dim) embeddings
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
            
        return emb


# ==================== U-Net SSM Architecture ====================

class SSMUNetBlock(nn.Module):
    """SSM + U-Net style block with skip connections"""
    
    def __init__(
        self,
        d_model: int,
        num_ssm_layers: int = 2,
        d_state: int = 64,
        use_multiscale: bool = True,
        skip_connection: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.skip_connection = skip_connection
        self.use_multiscale = use_multiscale
        
        if use_multiscale:
            self.ssm_layers = nn.ModuleList([
                MultiScaleSSM(
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                ) for _ in range(num_ssm_layers)
            ])
        else:
            self.ssm_layers = nn.ModuleList([
                SSMBlock(
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                ) for _ in range(num_ssm_layers)
            ])
        
        # Condition injection
        self.condition_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Skip connection projection
        if skip_connection:
            self.skip_proj = nn.Linear(d_model * 2, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        condition_emb: torch.Tensor,
        skip_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input sequence
            condition_emb: (B, D) condition embedding
            skip_input: (B, L, D) skip connection input
        """
        # Condition injection
        condition_projected = self.condition_proj(condition_emb)  # (B, D)
        condition_projected = condition_projected.unsqueeze(1)  # (B, 1, D)
        
        # Apply SSM layers
        for ssm_layer in self.ssm_layers:
            x = ssm_layer(x + condition_projected)
        
        # Skip connection
        if self.skip_connection and skip_input is not None:
            x = torch.cat([x, skip_input], dim=-1)  # (B, L, 2*D)
            x = self.skip_proj(x)  # (B, L, D)
        
        x = self.norm(x)
        return x


# ==================== Complete Lyro SSM U-Net ====================

class LyroSSMUNet(nn.Module):
    """
    Complete Lyro SSM + U-Net Architecture
    Combines state space models with U-Net for efficient long sequence processing
    """
    
    def __init__(
        self,
        input_channels: int = 8,
        hidden_dims: List[int] = [128, 256, 384, 512],
        ssm_layers: List[int] = [2, 3, 4, 4],
        d_state: int = 64,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        use_multiscale_ssm: bool = True,
    ):
        super().__init__()
        
        assert len(hidden_dims) == len(ssm_layers)
        
        self.num_stages = len(hidden_dims)
        self.hidden_dims = hidden_dims
        self.max_seq_len = max_seq_len
        
        # Condition embedding
        self.condition_embedding = AdvancedConditionalEmbedding(hidden_dims[0])
        
        # Input projection
        self.input_proj = nn.Conv1d(input_channels, hidden_dims[0], 1)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dims[0]) * 0.02
        )
        
        # Encoder stages
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        for i in range(self.num_stages - 1):
            encoder = SSMUNetBlock(
                d_model=hidden_dims[i],
                num_ssm_layers=ssm_layers[i],
                d_state=d_state,
                use_multiscale=use_multiscale_ssm,
                skip_connection=False,
                dropout=dropout,
            )
            self.encoders.append(encoder)
            
            # Downsampling
            downsampler = nn.Sequential(
                nn.LayerNorm(hidden_dims[i]),
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.Conv1d(hidden_dims[i+1], hidden_dims[i+1], 
                         kernel_size=3, stride=2, padding=1)
            )
            self.downsamplers.append(downsampler)
        
        # Bottleneck
        self.bottleneck = SSMUNetBlock(
            d_model=hidden_dims[-1],
            num_ssm_layers=ssm_layers[-1],
            d_state=d_state,
            use_multiscale=use_multiscale_ssm,
            skip_connection=False,
            dropout=dropout,
        )
        
        # Decoder stages  
        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for i in range(self.num_stages - 1, 0, -1):
            # Upsampling
            upsampler = nn.Sequential(
                nn.ConvTranspose1d(
                    hidden_dims[i], 
                    hidden_dims[i-1],
                    kernel_size=3, 
                    stride=2, 
                    padding=1,
                    output_padding=1
                ),
                nn.LayerNorm(hidden_dims[i-1])
            )
            self.upsamplers.append(upsampler)
            
            # Decoder with skip connections
            decoder = SSMUNetBlock(
                d_model=hidden_dims[i-1],
                num_ssm_layers=ssm_layers[i-1],
                d_state=d_state,
                use_multiscale=use_multiscale_ssm,
                skip_connection=True,
                dropout=dropout,
            )
            self.decoders.append(decoder)
        
        # Output projection
        self.output_proj = nn.Conv1d(hidden_dims[0], input_channels, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        conditions: Dict
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (B, C, T) input latent
            time: (B,) time steps
            conditions: conditioning dict
        Returns:
            velocity: (B, C, T) predicted velocity
        """
        B, C, T = x.shape
        
        # Generate condition embedding
        conditions = conditions.copy()
        conditions['time'] = time
        condition_emb = self.condition_embedding(conditions)  # (B, D)
        
        # Input projection and transpose
        x = self.input_proj(x)  # (B, D, T)
        x = x.transpose(1, 2)  # (B, T, D)
        
        # Add positional encoding
        if T <= self.max_seq_len:
            pos_emb = self.pos_embed[:, :T, :]
            x = x + pos_emb
        
        # U-Net forward pass
        skip_connections = []
        
        # Encoder
        for i, (encoder, downsampler) in enumerate(zip(self.encoders, self.downsamplers)):
            x = encoder(x, condition_emb)
            skip_connections.append(x)
            
            # Downsample
            x = x.transpose(1, 2)  # (B, D, T)
            x = downsampler[0](x.transpose(1, 2))  # LayerNorm
            x = downsampler[1](x)  # Linear
            x = x.transpose(1, 2)  # (B, T, D)
            x = downsampler[2](x.transpose(1, 2))  # Conv1d downsample
            x = x.transpose(1, 2)  # (B, T//2, D)
        
        # Bottleneck
        x = self.bottleneck(x, condition_emb)
        
        # Decoder
        for i, (upsampler, decoder) in enumerate(zip(self.upsamplers, self.decoders)):
            # Upsample
            x = x.transpose(1, 2)  # (B, D, T)
            x = upsampler[0](x)  # ConvTranspose1d
            x = x.transpose(1, 2)  # (B, T*2, D)
            x = upsampler[1](x)  # LayerNorm
            
            # Skip connection
            skip_input = skip_connections[-(i+1)]
            # Handle size mismatch
            if x.shape[1] != skip_input.shape[1]:
                min_len = min(x.shape[1], skip_input.shape[1])
                x = x[:, :min_len]
                skip_input = skip_input[:, :min_len]
            
            x = decoder(x, condition_emb, skip_input)
        
        # Output projection
        x = x.transpose(1, 2)  # (B, D, T)
        velocity = self.output_proj(x)  # (B, C, T)
        
        return velocity


# ==================== Task Controllers ====================

class TaskController:
    """Task control and condition preparation"""
    
    TASK_TOKENS = {
        'SONG': 0,
        'INST': 1, 
        'COVER': 2,
        'INPAINT': 3,
        'EXTEND': 4
    }
    
    @classmethod
    def create_conditions(
        cls,
        task_type: str,
        lyrics: Optional[torch.Tensor] = None,
        style_prompt: Optional[torch.Tensor] = None,
        icl_reference: Optional[torch.Tensor] = None,
        device: torch.device = None,
        **kwargs
    ) -> Dict:
        """Create conditions dict for given task"""
        
        if device is None:
            device = torch.device('cpu')
        
        conditions = {
            'task_token': torch.tensor([cls.TASK_TOKENS[task_type]], device=device),
            'lyrics': lyrics,
            'style_prompt': style_prompt,
            'icl_reference': icl_reference
        }
        
        # Task-specific processing
        if task_type == 'INST':
            conditions['lyrics'] = None
        elif task_type == 'COVER':
            if icl_reference is None:
                raise ValueError("COVER task requires ICL reference")
        
        return conditions


# ==================== EOS Token Handling ====================

class EOSTokenHandler:
    """Enhanced EOS token system"""
    
    EOS_TOKENS = {
        'EOA': 32000,    # End of Audio
        'EOD': 32001,    # End of Document  
        'REF_END': 32007,  # Reference end
        'MASK_END': 32009, # Mask end
    }
    
    @classmethod
    def check_eos(cls, token_ids: torch.Tensor) -> Tuple[Optional[str], Optional[int]]:
        """Check for EOS tokens in sequence"""
        for token_name, token_id in cls.EOS_TOKENS.items():
            if token_id in token_ids:
                return token_name, token_id
        return None, None
    
    @classmethod
    def apply_eos_penalty(
        cls,
        logits: torch.Tensor,
        current_length: int,
        min_length: int = 100,
        penalty: float = -float('inf')
    ) -> torch.Tensor:
        """Apply EOS penalty for early stopping prevention"""
        
        if current_length < min_length:
            for token_id in cls.EOS_TOKENS.values():
                if token_id < logits.size(-1):
                    logits[..., token_id] = penalty
        
        return logits


# ==================== Model Factory ====================

def create_lyro_ssm_model(
    input_channels: int = 8,
    model_size: str = "base",  # "small", "base", "large"
    max_seq_len: int = 8192,
    **kwargs
) -> LyroSSMUNet:
    """Factory function to create Lyro SSM models"""
    
    if model_size == "small":
        config = {
            "hidden_dims": [96, 192, 288, 384],
            "ssm_layers": [2, 2, 3, 3],
            "d_state": 32,
        }
    elif model_size == "base":
        config = {
            "hidden_dims": [128, 256, 384, 512], 
            "ssm_layers": [2, 3, 4, 4],
            "d_state": 64,
        }
    elif model_size == "large":
        config = {
            "hidden_dims": [256, 512, 768, 1024],
            "ssm_layers": [3, 4, 6, 6], 
            "d_state": 128,
        }
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # Override with kwargs
    config.update(kwargs)
    
    return LyroSSMUNet(
        input_channels=input_channels,
        max_seq_len=max_seq_len,
        **config
    )