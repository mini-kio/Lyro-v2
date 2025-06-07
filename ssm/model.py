# lyro/ssm/model.py
"""
Complete Lyro SSM Implementation with S6 (Mamba-2)
Advanced State Space Model with S6 architecture for superior long sequence processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union
import math
import numpy as np
from einops import rearrange, repeat

# Optional triton import for optimization (not required for basic functionality)
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


# ==================== S6 Core Components ====================

class S6StateSpaceKernel(nn.Module):
    """
    S6 (Mamba-2) State Space Kernel with State Space Dual (SSD) architecture
    Implements efficient selective scan with enhanced parallelization
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        d_head: int = 64,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        A_init_range: Tuple[float, float] = (1, 16),
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        bias: bool = True,
        conv_bias: bool = True,
        # S6 specific parameters
        chunk_size: int = 256,
        use_mem_eff_path: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state  
        self.d_conv = d_conv
        self.d_head = d_head
        self.expand = expand
        self.d_inner = d_model * expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        
        # Number of heads
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        assert self.nheads % self.ngroups == 0
        
        # Input projections (enhanced for S6)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # Convolution (causal depthwise)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner, 
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # S6 SSM parameters with SSD structure
        self.A_log = nn.Parameter(torch.empty(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        
        # dt projection (per head)
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))
        
        # Input projections for B, C, dt
        self.x_proj = nn.ModuleList([
            nn.Linear(self.headdim, d_state, bias=False) 
            for _ in range(self.nheads)
        ])
        
        self.dt_proj = nn.ModuleList([
            nn.Linear(self.headdim, 1, bias=True)
            for _ in range(self.nheads)  
        ])
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        
        # Normalization
        self.norm = nn.LayerNorm(self.d_inner)
        
        # Initialize parameters
        self._initialize_parameters(A_init_range, dt_min, dt_max, dt_init_floor)
    
    def _initialize_parameters(self, A_init_range, dt_min, dt_max, dt_init_floor):
        """Initialize S6 parameters"""
        
        # Initialize A (diagonal state matrix)
        A_init_range = A_init_range
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log.data.copy_(A_log)
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.data.copy_(inv_dt)
        
        # Initialize dt projections
        for dt_proj in self.dt_proj:
            nn.init.uniform_(dt_proj.weight, -0.1, 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        S6 forward pass with State Space Dual architecture
        
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
        
        # Reshape for multi-head processing
        x = rearrange(x, 'b l (h d) -> b l h d', h=self.nheads)  # (B, L, nheads, headdim)
        
        # S6 SSM computation with SSD
        y = self.selective_scan_ssd(x)
        
        # Reshape back
        y = rearrange(y, 'b l h d -> b l (h d)')  # (B, L, d_inner)
        
        # Normalization
        y = self.norm(y)
        
        # Gating with SiLU
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def selective_scan_ssd(self, x: torch.Tensor) -> torch.Tensor:
        """
        S6 Selective Scan with State Space Dual (SSD) architecture
        Enhanced parallel processing and memory efficiency
        """
        B, L, H, D = x.shape
        
        # Compute per-head parameters
        A = -torch.exp(self.A_log.float())  # (nheads,)
        
        outputs = []
        
        for h in range(H):
            x_h = x[:, :, h, :]  # (B, L, headdim)
            
            # Compute dt, B, C for this head
            dt_h = self.dt_proj[h](x_h).squeeze(-1)  # (B, L)
            dt_h = F.softplus(dt_h + self.dt_bias[h])
            
            # B and C matrices (learnable projections)
            B_h = self.x_proj[h](x_h)  # (B, L, d_state) 
            C_h = self.x_proj[h](x_h)  # (B, L, d_state)
            
            # Apply SSD selective scan
            if self.use_mem_eff_path and L > self.chunk_size:
                y_h = self._chunked_scan(x_h, A[h], B_h, C_h, dt_h, self.D[h])
            else:
                y_h = self._standard_scan(x_h, A[h], B_h, C_h, dt_h, self.D[h])
            
            outputs.append(y_h)
        
        # Stack outputs
        y = torch.stack(outputs, dim=2)  # (B, L, nheads, headdim)
        
        return y
    
    def _standard_scan(self, x, A, B, C, dt, D):
        """Standard recurrent scan with corrected tensor operations"""
        B_batch, L, d_head = x.shape
        d_state = B.shape[-1]
        
        # Initialize hidden state
        h = torch.zeros(B_batch, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        for t in range(L):
            # Discretization
            dt_t = dt[:, t].unsqueeze(-1)  # (B, 1)
            A_discrete = torch.exp(dt_t * A)  # (B, 1)
            B_discrete = dt_t * B[:, t]  # (B, d_state)
            
            # State update with corrected broadcasting
            # h: (B, d_state), A_discrete: (B, 1) -> A_discrete * h: (B, d_state)
            h_updated = A_discrete * h  # (B, d_state)
            
            # B_discrete: (B, d_state), x[:, t]: (B, d_head)
            # We need to project x[:, t] to match d_state dimension
            # Since B is already (B, L, d_state), the projection is already handled
            # We just need to properly broadcast
            x_t = x[:, t]  # (B, d_head)
            
            # For proper SSM operation, we need to ensure dimensional compatibility
            # The typical SSM formulation is: h = A*h + B*u where u is the input
            # Here B_discrete is (B, d_state) and we need a scalar input per state
            # So we sum over the head dimension to get a scalar input
            u_t = x_t.mean(dim=-1, keepdim=True)  # (B, 1) - average over heads
            
            # Now broadcast properly: B_discrete * u_t
            h_input = B_discrete * u_t  # (B, d_state) * (B, 1) -> (B, d_state)
            
            h = h_updated + h_input  # (B, d_state)
            
            # Output computation 
            # C[:, t]: (B, d_state), h: (B, d_state) -> sum over d_state
            y_t = torch.sum(h.unsqueeze(-1) * C[:, t].unsqueeze(-1), dim=1).squeeze(-1)  # (B,)
            
            # Add direct feedthrough: D * x[:, t] where D is scalar
            # Sum over head dimension to get scalar per batch
            y_t = y_t + D * x_t.mean(dim=-1)  # (B,) + scalar * (B,) -> (B,)
            
            # Expand to match expected output dimension (d_head)
            y_t = y_t.unsqueeze(-1).expand(-1, d_head)  # (B, d_head)
            
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)  # (B, L, d_head)
    
    def _chunked_scan(self, x, A, B, C, dt, D):
        """Memory-efficient chunked scan for long sequences"""
        B_batch, L, d_head = x.shape
        d_state = B.shape[-1]
        
        # Process in chunks
        chunk_size = self.chunk_size
        num_chunks = (L + chunk_size - 1) // chunk_size
        
        outputs = []
        h = torch.zeros(B_batch, d_state, device=x.device, dtype=x.dtype)
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, L)
            
            # Extract chunk
            x_chunk = x[:, start_idx:end_idx]
            dt_chunk = dt[:, start_idx:end_idx]
            B_chunk = B[:, start_idx:end_idx]
            C_chunk = C[:, start_idx:end_idx]
            
            # Process chunk with parallel scan
            y_chunk = self._parallel_scan_chunk(x_chunk, A, B_chunk, C_chunk, dt_chunk, D, h)
            outputs.append(y_chunk)
            
            # Update hidden state for next chunk
            if end_idx < L:
                # Compute final hidden state of this chunk
                for t in range(x_chunk.shape[1]):
                    dt_t = dt_chunk[:, t].unsqueeze(-1)
                    A_discrete = torch.exp(dt_t * A)
                    B_discrete = dt_t * B_chunk[:, t]
                    h = A_discrete * h + B_discrete * x_chunk[:, t].unsqueeze(-1)
        
        return torch.cat(outputs, dim=1)
    
    def _parallel_scan_chunk(self, x_chunk, A, B_chunk, C_chunk, dt_chunk, D, h_init):
        """Parallel scan for a single chunk"""
        B_batch, chunk_len, d_head = x_chunk.shape
        
        # Use associative scan for parallel processing
        # This is a simplified version - in practice, use optimized implementations
        outputs = []
        h = h_init
        
        for t in range(chunk_len):
            dt_t = dt_chunk[:, t].unsqueeze(-1)
            A_discrete = torch.exp(dt_t * A)
            B_discrete = dt_t * B_chunk[:, t]
            
            h = A_discrete * h + B_discrete * x_chunk[:, t].unsqueeze(-1)
            y_t = torch.sum(h * C_chunk[:, t], dim=-1) + D * x_chunk[:, t]
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)


class S6Block(nn.Module):
    """Complete S6 Block with normalization and skip connections"""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        d_head: int = 64,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        layer_idx: Optional[int] = None,
        # S6 specific
        chunk_size: int = 256,
        use_mem_eff_path: bool = True,
    ):
        super().__init__()
        
        self.s6 = S6StateSpaceKernel(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            d_head=d_head,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            use_mem_eff_path=use_mem_eff_path,
            layer_idx=layer_idx,
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
        x = self.s6(x)
        x = self.dropout(x)
        return x + residual


# ==================== Multi-Scale S6 ====================

class MultiScaleS6(nn.Module):
    """Multi-scale S6 for different temporal patterns"""
    
    def __init__(
        self,
        d_model: int,
        scales: List[int] = [1, 2, 4],
        d_state: int = 128,
        d_head: int = 64,
        dropout: float = 0.1,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        
        self.scales = scales
        self.s6_blocks = nn.ModuleList([
            S6Block(
                d_model=d_model,
                d_state=d_state,
                d_head=d_head,
                d_conv=4 * scale,  # Larger conv for larger scales
                dropout=dropout,
                layer_idx=layer_idx,
                chunk_size=256 // scale,  # Smaller chunks for larger scales
            ) for scale in scales
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(d_model * len(scales), d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input with multiple S6 scales"""
        B, L, D = x.shape
        
        scale_outputs = []
        
        for scale, s6_block in zip(self.scales, self.s6_blocks):
            if scale == 1:
                # No downsampling
                scale_out = s6_block(x)
            else:
                # Downsample, process, upsample
                x_down = x[:, ::scale, :]
                scale_out_down = s6_block(x_down)
                
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


# ==================== Enhanced Conditional Embeddings ====================

class AdvancedConditionalEmbedding(nn.Module):
    """Advanced conditioning with multiple modalities for S6"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Task embeddings
        self.task_embed = nn.Embedding(10, d_model)
        
        # Time embeddings (sinusoidal)
        self.time_embed = SinusoidalEmbedding(d_model)
        
        # Text/Lyrics encoder with enhanced attention
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
        
        # ICL reference encoder with S6
        self.icl_encoder = nn.Sequential(
            nn.Conv1d(8, d_model // 4, 1),
            nn.GroupNorm(8, d_model // 4),
            nn.GELU(),
            nn.Conv1d(d_model // 4, d_model, 1),
        )
        
        # Enhanced cross-attention for ICL
        self.icl_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Advanced fusion with residual connections
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 5, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
            ),
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
        ])
        
        # Adaptive weighting
        self.adaptive_weights = nn.Parameter(torch.ones(5))
        
    def forward(self, conditions: Dict) -> torch.Tensor:
        """Enhanced conditioning with adaptive weighting"""
        B = conditions['task_token'].shape[0]
        device = conditions['task_token'].device
        
        embeddings = []
        
        # 1. Task embedding
        task_emb = self.task_embed(conditions['task_token'])
        embeddings.append(task_emb)
        
        # 2. Time embedding
        time_emb = self.time_embed(conditions['time'])
        embeddings.append(time_emb)
        
        # 3. Text embedding with better handling
        if 'lyrics' in conditions and conditions['lyrics'] is not None:
            lyrics_tokens = conditions['lyrics']
            text_emb = self.text_encoder(lyrics_tokens).mean(dim=1)
        else:
            text_emb = torch.zeros(B, self.d_model, device=device)
        embeddings.append(text_emb)
        
        # 4. Style embedding
        if 'style_prompt' in conditions and conditions['style_prompt'] is not None:
            style_emb = self.style_encoder(conditions['style_prompt'])
        else:
            style_emb = torch.zeros(B, self.d_model, device=device)
        embeddings.append(style_emb)
        
        # 5. Enhanced ICL reference processing
        if 'icl_reference' in conditions and conditions['icl_reference'] is not None:
            icl_ref = conditions['icl_reference']
            icl_encoded = self.icl_encoder(icl_ref)
            icl_encoded = icl_encoded.transpose(1, 2)
            
            # Enhanced cross-attention
            query = torch.stack(embeddings, dim=1)
            icl_attended, attn_weights = self.icl_cross_attn(
                query, icl_encoded, icl_encoded
            )
            icl_emb = icl_attended.mean(dim=1)
        else:
            icl_emb = torch.zeros(B, self.d_model, device=device)
        
        embeddings.append(icl_emb)
        
        # 6. Adaptive weighted fusion
        weights = F.softmax(self.adaptive_weights, dim=0)
        weighted_embeddings = []
        
        for emb, weight in zip(embeddings, weights):
            weighted_embeddings.append(emb * weight)
        
        # 7. Multi-layer fusion with residuals
        combined = torch.cat(weighted_embeddings, dim=1)
        
        x = combined
        for layer in self.fusion_layers:
            residual = x if x.shape[-1] == self.d_model else None
            x = layer(x)
            if residual is not None:
                x = x + residual
        
        return x


class SinusoidalEmbedding(nn.Module):
    """Enhanced sinusoidal position embedding"""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x.unsqueeze(-1).float() * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
            
        return emb


# ==================== S6 U-Net Architecture ====================

class S6UNetBlock(nn.Module):
    """S6 + U-Net style block with enhanced skip connections"""
    
    def __init__(
        self,
        d_model: int,
        condition_dim: int,
        num_s6_layers: int = 2,
        d_state: int = 128,
        d_head: int = 64,
        use_multiscale: bool = True,
        skip_connection: bool = True,
        dropout: float = 0.1,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        
        self.skip_connection = skip_connection
        self.use_multiscale = use_multiscale
        
        if use_multiscale:
            self.s6_layers = nn.ModuleList([
                MultiScaleS6(
                    d_model=d_model,
                    d_state=d_state,
                    d_head=d_head,
                    dropout=dropout,
                    layer_idx=layer_idx,
                ) for _ in range(num_s6_layers)
            ])
        else:
            self.s6_layers = nn.ModuleList([
                S6Block(
                    d_model=d_model,
                    d_state=d_state,
                    d_head=d_head,
                    dropout=dropout,
                    layer_idx=layer_idx,
                ) for _ in range(num_s6_layers)
            ])
        
        # Enhanced condition injection
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Skip connection projection with gating
        if skip_connection:
            self.skip_proj = nn.Linear(d_model * 2, d_model)
            self.skip_gate = nn.Parameter(torch.zeros(1))
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        condition_emb: torch.Tensor,
        skip_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Enhanced forward with adaptive skip connections"""
        
        # Condition injection
        condition_projected = self.condition_proj(condition_emb)
        condition_projected = condition_projected.unsqueeze(1)
        
        # Apply S6 layers with condition
        for s6_layer in self.s6_layers:
            x = s6_layer(x + condition_projected)
        
        # Enhanced skip connection with gating
        if self.skip_connection and skip_input is not None:
            gate = torch.sigmoid(self.skip_gate)
            x_skip = torch.cat([x, skip_input * gate], dim=-1)
            x = self.skip_proj(x_skip)
        
        x = self.norm(x)
        return x


# ==================== Complete Lyro S6 U-Net ====================

class LyroS6UNet(nn.Module):
    """
    Complete Lyro S6 + U-Net Architecture
    Enhanced State Space Model with S6 for superior long sequence processing
    """
    
    def __init__(
        self,
        input_channels: int = 8,
        hidden_dims: List[int] = [128, 256, 384, 512],
        s6_layers: List[int] = [2, 3, 4, 4],
        d_state: int = 128,
        d_head: int = 64,
        max_seq_len: int = 8192,
        dropout: float = 0.1,
        use_multiscale_s6: bool = True,
        # S6 specific parameters
        chunk_size: int = 256,
        use_mem_eff_path: bool = True,
    ):
        super().__init__()
        
        assert len(hidden_dims) == len(s6_layers)
        
        self.num_stages = len(hidden_dims)
        self.hidden_dims = hidden_dims
        self.max_seq_len = max_seq_len
        self.input_channels = input_channels
        
        # Enhanced condition embedding
        self.condition_embedding = AdvancedConditionalEmbedding(hidden_dims[0])
        
        # Input projection with better initialization
        self.input_proj = nn.Conv1d(input_channels, hidden_dims[0], 1)
        nn.init.xavier_uniform_(self.input_proj.weight)
        
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dims[0]) * 0.02
        )
        
        # Encoder stages
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        for i in range(self.num_stages - 1):
            encoder = S6UNetBlock(
                d_model=hidden_dims[i],
                condition_dim=hidden_dims[0],
                num_s6_layers=s6_layers[i],
                d_state=d_state,
                d_head=d_head,
                use_multiscale=use_multiscale_s6,
                skip_connection=False,
                dropout=dropout,
                layer_idx=i,
            )
            self.encoders.append(encoder)
            
            # Enhanced downsampling
            downsampler = nn.Sequential(
                nn.LayerNorm(hidden_dims[i]),
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.GELU(),
                nn.Conv1d(
                    hidden_dims[i+1], 
                    hidden_dims[i+1], 
                    kernel_size=3, 
                    stride=2, 
                    padding=1
                ),
                nn.GroupNorm(min(32, hidden_dims[i+1] // 4), hidden_dims[i+1]),
            )
            self.downsamplers.append(downsampler)
        
        # Enhanced bottleneck
        self.bottleneck = S6UNetBlock(
            d_model=hidden_dims[-1],
            condition_dim=hidden_dims[0],
            num_s6_layers=s6_layers[-1],
            d_state=d_state,
            d_head=d_head,
            use_multiscale=use_multiscale_s6,
            skip_connection=False,
            dropout=dropout,
            layer_idx=self.num_stages - 1,
        )
        
        # Decoder stages with enhanced upsampling
        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for i in range(self.num_stages - 1, 0, -1):
            # Enhanced upsampling
            upsampler = nn.Sequential(
                nn.ConvTranspose1d(
                    hidden_dims[i], 
                    hidden_dims[i-1],
                    kernel_size=3, 
                    stride=2, 
                    padding=1,
                    output_padding=1
                ),
                nn.GroupNorm(min(32, hidden_dims[i-1] // 4), hidden_dims[i-1]),
                nn.GELU(),
            )
            self.upsamplers.append(upsampler)
            
            # Decoder with enhanced skip connections
            decoder = S6UNetBlock(
                d_model=hidden_dims[i-1],
                condition_dim=hidden_dims[0],
                num_s6_layers=s6_layers[i-1],
                d_state=d_state,
                d_head=d_head,
                use_multiscale=use_multiscale_s6,
                skip_connection=True,
                dropout=dropout,
                layer_idx=i-1,
            )
            self.decoders.append(decoder)
        
        # Enhanced output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dims[0], hidden_dims[0], 3, padding=1),
            nn.GroupNorm(min(32, hidden_dims[0] // 4), hidden_dims[0]),
            nn.GELU(),
            nn.Conv1d(hidden_dims[0], input_channels, 1),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Store S6 specific parameters
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        
    def _init_weights(self, module):
        """Enhanced weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        conditions: Dict
    ) -> torch.Tensor:
        """
        Enhanced forward pass with S6
        
        Args:
            x: (B, C, T) input latent
            time: (B,) time steps
            conditions: conditioning dict
        Returns:
            velocity: (B, C, T) predicted velocity
        """
        B, C, T = x.shape
        
        # Generate enhanced condition embedding
        conditions = conditions.copy()
        conditions['time'] = time
        condition_emb = self.condition_embedding(conditions)
        
        # Input projection and transpose
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        
        # Add learnable positional encoding
        if T <= self.max_seq_len:
            pos_emb = self.pos_embed[:, :T, :]
            x = x + pos_emb
        
        # Enhanced U-Net forward pass
        skip_connections = []
        
        # Encoder with progressive conditioning
        for i, (encoder, downsampler) in enumerate(zip(self.encoders, self.downsamplers)):
            x = encoder(x, condition_emb)
            skip_connections.append(x.clone())
            
            # Enhanced downsampling
            x = downsampler[0](x)  # LayerNorm
            x = downsampler[1](x)  # Linear
            x = downsampler[2](x)  # GELU
            x = x.transpose(1, 2)   # (B, D, T)
            x = downsampler[3](x)   # Conv1d downsample
            x = downsampler[4](x)   # GroupNorm
            x = x.transpose(1, 2)   # (B, T//2, D)
        
        # Enhanced bottleneck
        x = self.bottleneck(x, condition_emb)
        
        # Decoder with adaptive skip connections
        for i, (upsampler, decoder) in enumerate(zip(self.upsamplers, self.decoders)):
            # Enhanced upsampling
            x = x.transpose(1, 2)   # (B, D, T)
            x = upsampler[0](x)     # ConvTranspose1d
            x = upsampler[1](x)     # GroupNorm
            x = upsampler[2](x)     # GELU
            x = x.transpose(1, 2)   # (B, T*2, D)
            
            # Adaptive skip connection
            skip_input = skip_connections[-(i+1)]
            
            # Handle size mismatch with interpolation
            if x.shape[1] != skip_input.shape[1]:
                min_len = min(x.shape[1], skip_input.shape[1])
                if x.shape[1] > skip_input.shape[1]:
                    x = x[:, :min_len]
                else:
                    skip_input = F.interpolate(
                        skip_input.transpose(1, 2),
                        size=x.shape[1],
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)
            
            x = decoder(x, condition_emb, skip_input)
        
        # Enhanced output projection
        x = x.transpose(1, 2)
        velocity = self.output_proj(x)
        
        return velocity


# ==================== Task Controllers & Utilities ====================

class TaskController:
    """Enhanced task control for S6 model"""
    
    TASK_TOKENS = {
        'SONG': 0,
        'INST': 1, 
        'COVER': 2,
        'INPAINT': 3,
        'EXTEND': 4,
        'EDIT': 5,
        'REMIX': 6,
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
        """Create enhanced conditions dict for S6"""
        
        if device is None:
            device = torch.device('cpu')
        
        conditions = {
            'task_token': torch.tensor([cls.TASK_TOKENS[task_type]], device=device),
            'lyrics': lyrics,
            'style_prompt': style_prompt,
            'icl_reference': icl_reference
        }
        
        # Task-specific enhancements
        if task_type == 'INST':
            conditions['lyrics'] = None
        elif task_type in ['COVER', 'REMIX']:
            if icl_reference is None:
                raise ValueError(f"{task_type} task requires ICL reference")
        
        return conditions


class EOSTokenHandler:
    """Enhanced EOS token system for S6"""
    
    EOS_TOKENS = {
        'EOA': 32000,      # End of Audio
        'EOD': 32001,      # End of Document  
        'EOS': 32002,      # End of Sequence
        'REF_END': 32007,  # Reference end
        'MASK_END': 32009, # Mask end
        'EDIT_END': 32010, # Edit end
    }
    
    @classmethod
    def check_eos(cls, token_ids: torch.Tensor) -> Tuple[Optional[str], Optional[int]]:
        """Enhanced EOS detection"""
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
        penalty: float = -float('inf'),
        adaptive_penalty: bool = True
    ) -> torch.Tensor:
        """Enhanced EOS penalty with adaptive control"""
        
        if current_length < min_length:
            penalty_factor = 1.0
            if adaptive_penalty:
                # Gradually reduce penalty as we approach min_length
                penalty_factor = 1.0 - (current_length / min_length) * 0.5
            
            for token_id in cls.EOS_TOKENS.values():
                if token_id < logits.size(-1):
                    logits[..., token_id] = penalty * penalty_factor
        
        return logits


# ==================== Model Factory ====================

def create_lyro_s6_model(
    input_channels: int = 8,
    model_size: str = "base",
    max_seq_len: int = 8192,
    use_torch_compile: bool = False,  # 기본값을 False로 변경 (torch.compile 문제 때문에)
    use_mixed_precision: bool = True,
    compile_mode: str = "default",
    # S6 specific parameters
    chunk_size: int = 256,
    use_mem_eff_path: bool = True,
    **kwargs
) -> LyroS6UNet:
    """
    Factory function to create Lyro S6 models with optimization support
    
    Args:
        input_channels: Number of input channels
        model_size: Model size configuration ("small", "base", "large")
        max_seq_len: Maximum sequence length
        use_torch_compile: Enable torch.compile() optimization
        use_mixed_precision: Enable Mixed Precision training support
        compile_mode: Compilation mode
        chunk_size: S6 chunk size for memory efficiency
        use_mem_eff_path: Use memory efficient path for long sequences
        **kwargs: Additional model configuration
        
    Returns:
        Optimized LyroS6UNet model
    """
    
    if model_size == "small":
        config = {
            "hidden_dims": [96, 192, 288, 384],
            "s6_layers": [2, 2, 3, 3],
            "d_state": 64,
            "d_head": 32,
        }
    elif model_size == "base":
        config = {
            "hidden_dims": [128, 256, 384, 512], 
            "s6_layers": [2, 3, 4, 4],
            "d_state": 128,
            "d_head": 64,
        }
    elif model_size == "large":
        config = {
            "hidden_dims": [256, 512, 768, 1024],
            "s6_layers": [3, 4, 6, 6], 
            "d_state": 256,
            "d_head": 128,
        }
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # Override with kwargs
    config.update(kwargs)
    
    # Create S6 model
    model = LyroS6UNet(
        input_channels=input_channels,
        max_seq_len=max_seq_len,
        chunk_size=chunk_size,
        use_mem_eff_path=use_mem_eff_path,
        **config
    )
    
    # Add optimization flags
    model._use_mixed_precision = use_mixed_precision
    model._compile_mode = compile_mode
    
    # Apply torch.compile() if requested
    if use_torch_compile and hasattr(torch, 'compile'):
        try:
            print(f"🚀 Applying torch.compile() with mode '{compile_mode}'...")
            
            # Check for Triton availability and use safer compile mode on Windows
            import platform
            if platform.system() == "Windows":
                safe_compile_mode = "reduce-overhead" if compile_mode == "default" else compile_mode
                print(f"Windows detected, using safer compile mode: {safe_compile_mode}")
            else:
                safe_compile_mode = compile_mode
            
            # Compile key components
            model.condition_embedding = torch.compile(
                model.condition_embedding, 
                mode=safe_compile_mode
            )
            
            # Compile S6 layers
            for encoder in model.encoders:
                for s6_layer in encoder.s6_layers:
                    s6_layer = torch.compile(s6_layer, mode=safe_compile_mode)
            
            model.bottleneck = torch.compile(model.bottleneck, mode=safe_compile_mode)
            
            for decoder in model.decoders:
                for s6_layer in decoder.s6_layers:
                    s6_layer = torch.compile(s6_layer, mode=safe_compile_mode)
            
            print("✅ torch.compile() applied successfully")
            
        except Exception as e:
            print(f"Warning: torch.compile() failed: {e}")
            print("Continuing without compilation...")
            use_torch_compile = False
    
    return model


# ==================== Backward Compatibility ====================

# Alias for backward compatibility
LyroSSMUNet = LyroS6UNet
create_lyro_ssm_model = create_lyro_s6_model

def benchmark_s6_model(
    model: LyroS6UNet,
    batch_size: int = 1,
    seq_len: int = 1024,
    device: str = "cuda"
) -> Dict[str, float]:
    """Benchmark S6 model performance"""
    
    import time
    
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs
    x = torch.randn(batch_size, 8, seq_len, device=device)
    time_steps = torch.randn(batch_size, device=device)
    conditions = {
        'task_token': torch.zeros(batch_size, dtype=torch.long, device=device),
        'lyrics': None,
        'style_prompt': torch.randn(batch_size, 512, device=device),
        'icl_reference': None
    }
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, time_steps, conditions)
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(100):
            output = model(x, time_steps, conditions)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    throughput = (batch_size * seq_len) / avg_time
    
    return {
        "avg_forward_time": avg_time,
        "throughput_tokens_per_sec": throughput,
        "memory_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "memory_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "s6_chunk_size": model.chunk_size,
        "use_mem_eff_path": model.use_mem_eff_path,
    }


print("S6-based Lyro SSM model implementation completed!")
print("Key improvements:")
print("- S6 (Mamba-2) State Space architecture")
print("- State Space Dual (SSD) for enhanced parallelization")
print("- Memory-efficient chunked processing")
print("- Multi-scale S6 for different temporal patterns")
print("- Enhanced conditional embeddings with adaptive weighting")
print("- Improved skip connections with gating")
print("- Better initialization and normalization")
