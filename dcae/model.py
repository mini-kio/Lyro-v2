# lyro/dcae/model.py
"""
SSM-based LYRO DCAE Implementation with Memory Optimization - ENHANCED VERSION
State Space Model based Diffusion ConvNet AutoEncoder for high-quality audio compression
Enhanced with chunked processing and expanded gradient checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchaudio
import torchaudio.functional as F_audio
import numpy as np
from typing import Tuple, Optional, List, Dict, Union
import math
from pathlib import Path

# Import SSM components from SSM module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssm.model import MultiScaleSSM, SinusoidalEmbedding


# ==================== Memory-Optimized State Space Kernel ====================

class MemoryOptimizedStateSpaceKernel(nn.Module):
    """
    Memory-Optimized State Space Model Kernel with chunked processing
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
        # Memory optimization parameters
        chunk_size: int = 1024,  # Process in chunks to reduce memory
        use_checkpointing: bool = True,
        memory_efficient: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.chunk_size = chunk_size
        self.use_checkpointing = use_checkpointing
        self.memory_efficient = memory_efficient
        
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
        Memory-optimized forward pass with chunked processing and checkpointing
        
        Args:
            x: (B, L, D) input sequence
        Returns:
            output: (B, L, D) processed sequence
        """
        B, L, D = x.shape
        
        # Apply checkpointing to reduce memory
        if self.use_checkpointing and self.training:
            return checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation with memory optimization"""
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
        
        # Memory-optimized SSM computation
        if self.memory_efficient and L > self.chunk_size:
            x = self._chunked_ssm(x)
        else:
            x = self.ssm(x)
        
        # Gating
        y = x * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output
    
    def _chunked_ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Process SSM in chunks to reduce peak memory"""
        B, L, D = x.shape
        
        # Split into chunks
        chunks = []
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            chunk = x[:, start:end]
            
            # Process chunk
            processed_chunk = self.ssm(chunk)
            chunks.append(processed_chunk)
        
        # Concatenate results
        return torch.cat(chunks, dim=1)
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Core SSM computation with memory-optimized discretization"""
        B, L, D = x.shape
        
        # Compute dt, B, C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)
        
        # Compute dt
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt + self.dt_proj.bias)
        
        # Compute A
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Memory-optimized discretization
        A_discrete, B_discrete = self.memory_efficient_discretize(A, B, dt)
        
        # SSM step
        y = self.ssm_step(x, A_discrete, B_discrete, C, self.D)
        
        return y
    
    def memory_efficient_discretize(self, A, B, dt):
        """
        Memory-optimized discretization with chunked processing
        
        Args:
            A: (d_inner, d_state)
            B: (B, L, d_state) 
            dt: (B, L, d_inner)
        """
        B_batch, L, _ = dt.shape
        
        # Expand dt for broadcasting
        dt = dt.unsqueeze(-1)  # (B, L, d_inner, 1)
        A = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)
        
        # Process in chunks to reduce memory usage
        chunk_size = min(self.chunk_size, L)
        
        A_discrete_chunks = []
        B_discrete_chunks = []
        
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            
            # Chunk data
            dt_chunk = dt[:, start:end]  # (B, chunk_size, d_inner, 1)
            B_chunk = B[:, start:end]    # (B, chunk_size, d_state)
            
            # Zero-order hold discretization for chunk
            # Use torch.clamp to prevent overflow
            dt_A = torch.clamp(dt_chunk * A, min=-10, max=10)
            A_discrete_chunk = torch.exp(dt_A)  # (B, chunk_size, d_inner, d_state)
            
            # Compute B_discrete more efficiently
            # B_discrete = (A_discrete - 1) / A * B
            # Use numerically stable computation
            with torch.no_grad():
                # For small dt_A, use Taylor expansion: exp(x) - 1 ≈ x + x²/2
                small_mask = torch.abs(dt_A) < 0.1
            
            B_discrete_chunk = torch.where(
                small_mask,
                # Taylor expansion for numerical stability
                dt_chunk * (1 + 0.5 * dt_A) * B_chunk.unsqueeze(2),
                # Standard computation
                (A_discrete_chunk - 1) / (dt_A + 1e-8) * B_chunk.unsqueeze(2)
            )
            
            A_discrete_chunks.append(A_discrete_chunk)
            B_discrete_chunks.append(B_discrete_chunk)
            
            # Clear intermediate tensors
            del dt_chunk, B_chunk, dt_A, A_discrete_chunk, B_discrete_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate chunks
        A_discrete = torch.cat(A_discrete_chunks, dim=1)
        B_discrete = torch.cat(B_discrete_chunks, dim=1)
        
        return A_discrete, B_discrete
    
    def ssm_step(self, x, A, B, C, D):
        """Perform SSM recurrence with memory optimization"""
        B_batch, L, d_inner = x.shape
        d_state = A.shape[-1]
        
        # Process in chunks if sequence is too long
        if L > self.chunk_size and self.memory_efficient:
            return self._chunked_ssm_step(x, A, B, C, D)
        
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
    
    def _chunked_ssm_step(self, x, A, B, C, D):
        """Process SSM step in chunks for memory efficiency"""
        B_batch, L, d_inner = x.shape
        d_state = A.shape[-1]
        
        # Initialize state
        h = torch.zeros(B_batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        # Process in chunks
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            
            chunk_outputs = []
            
            for t in range(start, end):
                # Current inputs
                x_t = x[:, t]
                A_t = A[:, t]
                B_t = B[:, t]
                C_t = C[:, t]
                
                # State update
                h = A_t * h + B_t * x_t.unsqueeze(-1)
                
                # Output
                y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1) + D * x_t
                chunk_outputs.append(y_t)
            
            outputs.extend(chunk_outputs)
        
        y = torch.stack(outputs, dim=1)
        return y


class CheckpointedSSMBlock(nn.Module):
    """SSM Block with enhanced gradient checkpointing"""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        chunk_size: int = 1024,
        use_checkpointing: bool = True,
        memory_efficient: bool = True,
    ):
        super().__init__()
        
        self.use_checkpointing = use_checkpointing
        
        self.ssm = MemoryOptimizedStateSpaceKernel(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            chunk_size=chunk_size,
            use_checkpointing=use_checkpointing,
            memory_efficient=memory_efficient,
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
        if self.use_checkpointing and self.training:
            return checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        return x + residual


# ==================== Helper Functions ====================

def safe_group_norm(num_channels: int, base_groups: int = 8) -> nn.Module:
    """Create safe GroupNorm that handles small channel counts"""
    if num_channels <= 1:
        return nn.Identity()
    elif num_channels < base_groups:
        return nn.GroupNorm(1, num_channels)
    else:
        num_groups = min(base_groups, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_channels)


# ==================== Memory-Optimized SSM Components ====================

class MemoryOptimizedSSMEncoder(nn.Module):
    """Memory-optimized SSM-based encoder with chunked processing"""
    
    def __init__(
        self,
        input_channels: int = 2,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4, 8, 16],
        ssm_layers: List[int] = [2, 2, 3, 3, 2],
        latent_channels: int = 8,
        d_state: int = 64,
        dropout: float = 0.1,
        use_multiscale_ssm: bool = True,
        use_weight_norm: bool = True,
        # Memory optimization parameters
        chunk_size: int = 1024,
        use_checkpointing: bool = True,
        memory_efficient: bool = True,
    ):
        super().__init__()
        
        assert len(channel_multipliers) == len(ssm_layers)
        
        self.num_stages = len(channel_multipliers)
        self.base_channels = base_channels
        self.use_checkpointing = use_checkpointing
        self.chunk_size = chunk_size
        self.memory_efficient = memory_efficient
        
        # Initial convolution
        stem_conv = nn.Conv1d(input_channels, base_channels, 7, padding=3, padding_mode="reflect")
        if use_weight_norm:
            stem_conv = nn.utils.weight_norm(stem_conv)
        
        self.stem = nn.Sequential(
            stem_conv,
            safe_group_norm(base_channels),
            nn.SiLU()
        )
        
        # Encoder stages with checkpointing
        self.stages = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(self.num_stages):
            out_channels = base_channels * channel_multipliers[i]
            
            # Downsampling layer (except first stage)
            if i == 0:
                downsample = nn.Identity()
            else:
                if use_weight_norm:
                    downsample = nn.utils.weight_norm(
                        nn.Conv1d(current_channels, out_channels, 3, stride=2, padding=1)
                    )
                else:
                    downsample = nn.Conv1d(current_channels, out_channels, 3, stride=2, padding=1)
                
                downsample = nn.Sequential(
                    downsample,
                    safe_group_norm(out_channels),
                    nn.SiLU()
                )
            
            # Memory-optimized SSM blocks
            if use_multiscale_ssm:
                ssm_processor = MultiScaleSSM(
                    d_model=out_channels,
                    scales=[1, 2, 4] if out_channels >= 128 else [1, 2],
                    d_state=d_state,
                    dropout=dropout
                )
            else:
                ssm_blocks = nn.ModuleList([
                    CheckpointedSSMBlock(
                        d_model=out_channels,
                        d_state=d_state,
                        dropout=dropout,
                        chunk_size=chunk_size,
                        use_checkpointing=use_checkpointing,
                        memory_efficient=memory_efficient,
                    ) for _ in range(ssm_layers[i])
                ])
                ssm_processor = lambda x: self._apply_checkpointed_ssm_blocks(x, ssm_blocks)
            
            stage = nn.ModuleDict({
                'downsample': downsample,
                'ssm_processor': ssm_processor
            })
            
            self.stages.append(stage)
            current_channels = out_channels
        
        # Final projection to latent space
        final_conv1 = nn.Conv1d(current_channels, latent_channels * 2, 3, padding=1)
        final_conv2 = nn.Conv1d(latent_channels * 2, latent_channels, 1)
        
        if use_weight_norm:
            final_conv1 = nn.utils.weight_norm(final_conv1)
            final_conv2 = nn.utils.weight_norm(final_conv2)
        
        self.final_conv = nn.Sequential(
            final_conv1,
            safe_group_norm(latent_channels * 2),
            nn.SiLU(),
            final_conv2
        )
        
        self.apply(self._init_weights)
    
    def _apply_checkpointed_ssm_blocks(self, x, ssm_blocks):
        """Apply multiple checkpointed SSM blocks"""
        B, C, T = x.shape
        x = x.transpose(1, 2)  # (B, T, C)
        
        for ssm_block in ssm_blocks:
            if self.use_checkpointing and self.training:
                x = checkpoint.checkpoint(ssm_block, x, use_reentrant=False)
            else:
                x = ssm_block(x)
            
        x = x.transpose(1, 2)  # (B, C, T)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Memory-optimized encoding with checkpointing
        
        Args:
            x: (B, 2, T) stereo audio
        Returns:
            latent: (B, 8, T//32) compressed latent
            skip_features: List of skip connection features for decoder
        """
        x = self.stem(x)
        
        skip_features = []
        
        for i, stage in enumerate(self.stages):
            # Apply downsampling
            x = stage['downsample'](x)
            
            # Convert to sequence format for SSM with checkpointing
            B, C, T = x.shape
            
            if self.use_checkpointing and self.training:
                # Checkpoint the entire stage processing
                x = checkpoint.checkpoint(
                    self._process_stage_ssm, 
                    x, stage['ssm_processor'], 
                    use_reentrant=False
                )
            else:
                x = self._process_stage_ssm(x, stage['ssm_processor'])
            
            # Store skip connection features
            skip_features.append(x.clone())
        
        latent = self.final_conv(x)
        
        return latent, skip_features
    
    def _process_stage_ssm(self, x, ssm_processor):
        """Process SSM stage with proper tensor format handling"""
        B, C, T = x.shape
        x_seq = x.transpose(1, 2)  # (B, T, C)
        
        if hasattr(ssm_processor, '__call__'):
            if isinstance(ssm_processor, MultiScaleSSM):
                x_seq = ssm_processor(x_seq)
            else:
                x_seq = ssm_processor(x_seq)
        
        return x_seq.transpose(1, 2)  # (B, C, T)


class MemoryOptimizedSSMUpsampling(nn.Module):
    """Memory-optimized SSM-enhanced upsampling with checkpointing"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        kernel_size: int = 8,
        use_weight_norm: bool = True,
        sample_rate: int = 44100,
        use_checkpointing: bool = True,
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.use_checkpointing = use_checkpointing
        
        # Transposed convolution
        padding = (kernel_size - scale_factor) // 2
        conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=scale_factor,
            padding=padding
        )
        
        if use_weight_norm:
            conv_transpose = nn.utils.weight_norm(conv_transpose)
        
        self.upsample = conv_transpose
        
        # Anti-aliasing filter
        if scale_factor > 1:
            try:
                from torchaudio.transforms import LowpassBiquad
                cutoff_freq = 0.9 * (sample_rate / 2) / scale_factor
                self.anti_alias_filter = LowpassBiquad(
                    sample_rate=sample_rate,
                    cutoff_freq=cutoff_freq
                )
            except ImportError:
                self.anti_alias_filter = nn.Identity()
        else:
            self.anti_alias_filter = nn.Identity()
        
        self.norm = safe_group_norm(out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        if self.use_checkpointing and self.training:
            return checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        x = self.upsample(x)
        
        # Apply anti-aliasing filter after upsampling
        if self.scale_factor > 1 and not isinstance(self.anti_alias_filter, nn.Identity):
            x_filtered = []
            for i in range(x.shape[1]):
                x_ch = self.anti_alias_filter(x[:, i:i+1])
                x_filtered.append(x_ch)
            x = torch.cat(x_filtered, dim=1)
        
        x = self.norm(x)
        x = self.activation(x)
        
        return x


class MemoryOptimizedSSMDecoder(nn.Module):
    """Memory-optimized SSM-based decoder with enhanced checkpointing"""
    
    def __init__(
        self,
        latent_channels: int = 8,
        base_channels: int = 64,
        channel_multipliers: List[int] = [16, 8, 4, 2, 1],
        kernel_sizes: List[int] = [8, 8, 6, 6, 7],
        scale_factors: List[int] = [2, 2, 2, 2, 2],
        ssm_layers: List[int] = [2, 3, 3, 2, 2],
        output_channels: int = 2,
        d_state: int = 64,
        use_multiscale_ssm: bool = True,
        dropout: float = 0.1,
        use_weight_norm: bool = True,
        sample_rate: int = 44100,
        # Memory optimization parameters
        chunk_size: int = 1024,
        use_checkpointing: bool = True,
        memory_efficient: bool = True,
    ):
        super().__init__()
        
        assert len(channel_multipliers) == len(kernel_sizes) == len(scale_factors) == len(ssm_layers)
        
        self.num_stages = len(channel_multipliers)
        self.use_checkpointing = use_checkpointing
        self.chunk_size = chunk_size
        self.memory_efficient = memory_efficient
        
        # Initial projection from latent space
        initial_channels = base_channels * channel_multipliers[0]
        initial_conv = nn.Conv1d(latent_channels, initial_channels, 3, padding=1)
        
        if use_weight_norm:
            initial_conv = nn.utils.weight_norm(initial_conv)
        
        self.initial_conv = nn.Sequential(
            initial_conv,
            safe_group_norm(initial_channels),
            nn.SiLU()
        )
        
        # Skip connection projection layers
        self.skip_projections = nn.ModuleList()
        
        # Decoder stages with checkpointing
        self.stages = nn.ModuleList()
        current_channels = initial_channels
        
        for i in range(self.num_stages):
            if i == self.num_stages - 1:
                out_channels = output_channels
            else:
                out_channels = base_channels * channel_multipliers[i + 1]
            
            # Skip connection projection (except for last stage)
            if i < self.num_stages - 1:
                skip_proj = nn.Conv1d(current_channels * 2, current_channels, 1)
                if use_weight_norm:
                    skip_proj = nn.utils.weight_norm(skip_proj)
                self.skip_projections.append(skip_proj)
            else:
                self.skip_projections.append(nn.Identity())
            
            # Memory-optimized upsampling layer
            upsample = MemoryOptimizedSSMUpsampling(
                current_channels, out_channels,
                scale_factor=scale_factors[i],
                kernel_size=kernel_sizes[i],
                use_weight_norm=use_weight_norm,
                sample_rate=sample_rate,
                use_checkpointing=use_checkpointing
            )
            
            # Memory-optimized SSM processing (except for last stage)
            if i < self.num_stages - 1:
                if use_multiscale_ssm:
                    ssm_processor = MultiScaleSSM(
                        d_model=out_channels,
                        scales=[1, 2] if out_channels >= 128 else [1],
                        d_state=d_state,
                        dropout=dropout
                    )
                else:
                    ssm_blocks = nn.ModuleList([
                        CheckpointedSSMBlock(
                            d_model=out_channels,
                            d_state=d_state,
                            dropout=dropout,
                            chunk_size=chunk_size,
                            use_checkpointing=use_checkpointing,
                            memory_efficient=memory_efficient,
                        ) for _ in range(ssm_layers[i])
                    ])
                    ssm_processor = lambda x: self._apply_checkpointed_ssm_blocks(x, ssm_blocks)
            else:
                ssm_processor = nn.Identity()
            
            stage = nn.ModuleDict({
                'upsample': upsample,
                'ssm_processor': ssm_processor
            })
            
            self.stages.append(stage)
            current_channels = out_channels
        
        # Final output layer
        final_conv = nn.Conv1d(output_channels, output_channels, 7, padding=3, padding_mode="reflect")
        if use_weight_norm:
            final_conv = nn.utils.weight_norm(final_conv)
        
        self.final_conv = nn.Sequential(
            final_conv,
            nn.Tanh()
        )
        
        self.apply(self._init_weights)
    
    def _apply_checkpointed_ssm_blocks(self, x, ssm_blocks):
        """Apply multiple checkpointed SSM blocks"""
        B, C, T = x.shape
        x = x.transpose(1, 2)  # (B, T, C)
        
        for ssm_block in ssm_blocks:
            if self.use_checkpointing and self.training:
                x = checkpoint.checkpoint(ssm_block, x, use_reentrant=False)
            else:
                x = ssm_block(x)
            
        x = x.transpose(1, 2)  # (B, C, T)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, latent: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Memory-optimized decoding with checkpointing
        
        Args:
            latent: (B, 8, T//32) compressed latent
            skip_features: List of skip connection features from encoder
        Returns:
            audio: (B, 2, T) reconstructed stereo audio
        """
        # Calculate expected output length based on latent
        expected_length = latent.shape[-1] * 32  # 32x upsampling ratio
        
        x = self.initial_conv(latent)
        
        # Reverse skip features order (decoder goes from deep to shallow)
        skip_features = skip_features[::-1]
        
        for i, stage in enumerate(self.stages):
            # Apply skip connections (except for last stage)
            if i < self.num_stages - 1 and i < len(skip_features):
                skip_feature = skip_features[i]
                
                # Handle size mismatch by interpolation
                if x.shape[-1] != skip_feature.shape[-1]:
                    skip_feature = F.interpolate(
                        skip_feature, size=x.shape[-1], 
                        mode='linear', align_corners=False
                    )
                
                # Concatenate and project
                x = torch.cat([x, skip_feature], dim=1)
                x = self.skip_projections[i](x)
            
            # Apply upsampling with checkpointing
            x = stage['upsample'](x)
            
            # Apply SSM processing with checkpointing
            if not isinstance(stage['ssm_processor'], nn.Identity):
                if self.use_checkpointing and self.training:
                    x = checkpoint.checkpoint(
                        self._process_stage_ssm, 
                        x, stage['ssm_processor'], 
                        use_reentrant=False
                    )
                else:
                    x = self._process_stage_ssm(x, stage['ssm_processor'])
        
        audio = self.final_conv(x)
        
        # Ensure exact length match by trimming or padding
        current_length = audio.shape[-1]
        if current_length != expected_length:
            if current_length > expected_length:
                # Trim excess
                audio = audio[..., :expected_length]
            else:
                # Pad if needed
                pad_length = expected_length - current_length
                audio = F.pad(audio, (0, pad_length), mode='reflect')
        
        return audio
    
    def _process_stage_ssm(self, x, ssm_processor):
        """Process SSM stage with proper tensor format handling"""
        B, C, T = x.shape
        x_seq = x.transpose(1, 2)  # (B, T, C)
        
        if isinstance(ssm_processor, MultiScaleSSM):
            x_seq = ssm_processor(x_seq)
        else:
            x_seq = ssm_processor(x_seq)
        
        return x_seq.transpose(1, 2)  # (B, C, T)


# ==================== Enhanced Multi-Resolution STFT Loss ====================

class MultiResolutionSTFTLoss(nn.Module):
    """Enhanced Multi-resolution STFT loss with extended frequency coverage"""
    
    def __init__(
        self,
        fft_sizes: List[int] = [4096, 2048, 1024, 512, 256, 128, 64],
        hop_sizes: Optional[List[int]] = None,
        win_sizes: Optional[List[int]] = None,
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.5,
        w_phasediff: float = 0.0
    ):
        super().__init__()
        
        if hop_sizes is None:
            hop_sizes = [f // 4 for f in fft_sizes]
        if win_sizes is None:
            win_sizes = fft_sizes
            
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phasediff = w_phasediff
        
        # Pre-compute windows
        self.windows = nn.ParameterDict()
        for i, win_size in enumerate(win_sizes):
            self.windows[f'window_{i}'] = nn.Parameter(
                torch.hann_window(win_size), requires_grad=False
            )
    
    def stft(self, x, fft_size, hop_size, win_size, window):
        """Compute STFT"""
        return torch.stft(
            x, fft_size, hop_size, win_size, window,
            return_complex=True, normalized=False, center=True, pad_mode='reflect'
        )
    
    def forward(self, x, y):
        """
        Compute enhanced multi-resolution STFT loss
        
        Args:
            x: (B, C, T) predicted audio
            y: (B, C, T) target audio
        """
        # Ensure both tensors have the same length
        min_length = min(x.shape[-1], y.shape[-1])
        x = x[..., :min_length]
        y = y[..., :min_length]
        
        total_loss = 0.0
        
        for i, (fft_size, hop_size, win_size) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_sizes)
        ):
            window = self.windows[f'window_{i}'].to(x.device)
            
            # Compute STFT for each channel
            x_stft_list = []
            y_stft_list = []
            
            for c in range(x.shape[1]):
                x_stft = self.stft(x[:, c], fft_size, hop_size, win_size, window)
                y_stft = self.stft(y[:, c], fft_size, hop_size, win_size, window)
                x_stft_list.append(x_stft)
                y_stft_list.append(y_stft)
            
            # Stack channel results
            x_stft = torch.stack(x_stft_list, dim=1)
            y_stft = torch.stack(y_stft_list, dim=1)
            
            # Magnitude spectra
            x_mag = torch.abs(x_stft)
            y_mag = torch.abs(y_stft)
            
            # Ensure STFT results have the same shape (additional safety check)
            if x_mag.shape != y_mag.shape:
                min_time_frames = min(x_mag.shape[-1], y_mag.shape[-1])
                x_mag = x_mag[..., :min_time_frames]
                y_mag = y_mag[..., :min_time_frames]
            
            # Spectral convergence loss
            if self.w_sc > 0:
                sc_loss = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-8)
                total_loss += self.w_sc * sc_loss
            
            # Log magnitude loss
            if self.w_log_mag > 0:
                log_mag_loss = F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))
                total_loss += self.w_log_mag * log_mag_loss
            
            # Linear magnitude loss
            if self.w_lin_mag > 0:
                lin_mag_loss = F.l1_loss(x_mag, y_mag)
                total_loss += self.w_lin_mag * lin_mag_loss
            
            # Frequency-weighted loss for better high-frequency preservation
            if i < 3:  # Apply to higher resolution STFTs
                freq_weights = torch.linspace(0.5, 2.0, x_mag.shape[2], device=x_mag.device)
                freq_weights = freq_weights.view(1, 1, -1, 1)
                
                weighted_loss = F.l1_loss(x_mag * freq_weights, y_mag * freq_weights)
                total_loss += 0.1 * weighted_loss
        
        return total_loss / len(self.fft_sizes)


# ==================== Complete Memory-Optimized DCAE Model ====================

class MemoryOptimizedLyroMusicDCAE(nn.Module):
    """
    Memory-Optimized SSM-based Lyro Music DCAE with enhanced features
    
    Features:
    - Chunked processing in StateSpaceKernel discretization
    - Expanded gradient checkpointing
    - Memory-efficient SSM operations
    - Configurable chunk sizes and memory optimization levels
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        latent_channels: int = 8,
        use_vector_quantization: bool = False,
        vq_num_embeddings: int = 1024,
        vq_commitment_cost: float = 0.25,
        dual_channel_processing: bool = True,
        encoder_base_channels: int = 64,
        decoder_base_channels: int = 64,
        dropout: float = 0.1,
        use_weight_norm: bool = True,
        use_multiscale_ssm: bool = True,
        d_state: int = 64,
        # Memory optimization parameters
        chunk_size: int = 1024,
        use_checkpointing: bool = True,
        memory_efficient: bool = True,
        checkpointing_segments: int = 4,  # Number of segments for checkpointing
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.latent_channels = latent_channels
        self.use_vq = use_vector_quantization
        self.dual_channel_processing = dual_channel_processing
        self.chunk_size = chunk_size
        self.use_checkpointing = use_checkpointing
        self.memory_efficient = memory_efficient
        self.checkpointing_segments = checkpointing_segments
        
        # Memory-optimized SSM-based Encoder with enhanced checkpointing
        self.encoder = MemoryOptimizedSSMEncoder(
            input_channels=2,
            base_channels=encoder_base_channels,
            latent_channels=latent_channels,
            d_state=d_state,
            dropout=dropout,
            use_multiscale_ssm=use_multiscale_ssm,
            use_weight_norm=use_weight_norm,
            chunk_size=chunk_size,
            use_checkpointing=use_checkpointing,
            memory_efficient=memory_efficient,
        )
        
        # Vector quantization (optional)
        if use_vector_quantization:
            from .vq import VectorQuantizer
            self.quantizer = VectorQuantizer(
                num_embeddings=vq_num_embeddings,
                embedding_dim=latent_channels,
                commitment_cost=vq_commitment_cost
            )
        
        # Memory-optimized SSM-based Decoder with enhanced checkpointing
        self.decoder = MemoryOptimizedSSMDecoder(
            latent_channels=latent_channels,
            base_channels=decoder_base_channels,
            output_channels=2,
            d_state=d_state,
            dropout=dropout,
            use_multiscale_ssm=use_multiscale_ssm,
            use_weight_norm=use_weight_norm,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            use_checkpointing=use_checkpointing,
            memory_efficient=memory_efficient,
        )
        
        # Dual-channel processing (learnable channel importance)
        if dual_channel_processing:
            self.vocal_weight = nn.Parameter(torch.ones(4))
            self.inst_weight = nn.Parameter(torch.ones(4))
        
        # Enhanced Loss function
        self.stft_loss = MultiResolutionSTFTLoss()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Memory-optimized encoding with chunked processing and checkpointing
        
        Args:
            audio: (B, 2, T) stereo audio at 44.1kHz
        Returns:
            latent: (B, 8, T//32) compressed latent
            skip_features: List of skip connection features
        """
        # Apply segmented checkpointing for very long audio
        if (self.use_checkpointing and self.training and 
            audio.shape[-1] > self.chunk_size * self.checkpointing_segments):
            
            return self._segmented_encode(audio)
        else:
            return self._encode_impl(audio)
    
    def _segmented_encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode with segmented checkpointing for very long audio"""
        B, C, T = audio.shape
        segment_length = T // self.checkpointing_segments
        
        latent_segments = []
        skip_features_segments = []
        
        for i in range(self.checkpointing_segments):
            start = i * segment_length
            end = (i + 1) * segment_length if i < self.checkpointing_segments - 1 else T
            
            audio_segment = audio[:, :, start:end]
            
            # Use checkpointing for each segment
            latent_seg, skip_seg = checkpoint.checkpoint(
                self._encode_impl, audio_segment, use_reentrant=False
            )
            
            latent_segments.append(latent_seg)
            skip_features_segments.append(skip_seg)
        
        # Combine segments
        latent = torch.cat(latent_segments, dim=-1)
          # Combine skip features
        combined_skip_features = []
        for layer_idx in range(len(skip_features_segments[0])):
            layer_features = [seg[layer_idx] for seg in skip_features_segments]
            combined_layer = torch.cat(layer_features, dim=-1)
            combined_skip_features.append(combined_layer)
        
        return latent, combined_skip_features
    
    def _encode_impl(self, audio: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Implementation of encoding"""
        latent, skip_features = self.encoder(audio)
        
        # Apply dual-channel weighting - dynamically handle different latent channel counts
        if self.dual_channel_processing:
            # For latent_channels that are not divisible by 2, only apply weighting to available channels
            if self.latent_channels >= 8:
                # Standard case: 8+ channels, split 4+4
                vocal_channels = latent[:, :4] * self.vocal_weight.view(1, 4, 1)
                inst_channels = latent[:, 4:8] * self.inst_weight.view(1, 4, 1)
                # Keep remaining channels unchanged
                if self.latent_channels > 8:
                    remaining_channels = latent[:, 8:]
                    latent = torch.cat([vocal_channels, inst_channels, remaining_channels], dim=1)
                else:
                    latent = torch.cat([vocal_channels, inst_channels], dim=1)
            elif self.latent_channels == 6:
                # Small model case: 6 channels, split 3+3
                half_channels = self.latent_channels // 2
                vocal_weight_adjusted = self.vocal_weight[:half_channels].view(1, half_channels, 1)
                inst_weight_adjusted = self.inst_weight[:half_channels].view(1, half_channels, 1)
                
                vocal_channels = latent[:, :half_channels] * vocal_weight_adjusted
                inst_channels = latent[:, half_channels:] * inst_weight_adjusted
                latent = torch.cat([vocal_channels, inst_channels], dim=1)
            else:
                # For other cases, apply simple uniform weighting
                pass  # Keep latent unchanged
        
        return latent, skip_features
    
    def decode(self, latent: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Memory-optimized decoding with chunked processing and checkpointing
        
        Args:
            latent: (B, 8, T//32) compressed latent
            skip_features: List of skip connection features from encoder
        Returns:
            audio: (B, 2, T) reconstructed stereo audio
        """
        # Apply segmented checkpointing for very long latents
        if (self.use_checkpointing and self.training and 
            latent.shape[-1] > self.chunk_size // 32 * self.checkpointing_segments):
            
            return self._segmented_decode(latent, skip_features)
        else:
            return self.decoder(latent, skip_features)
    
    def _segmented_decode(self, latent: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """Decode with segmented checkpointing for very long latents"""
        B, C, T = latent.shape
        segment_length = T // self.checkpointing_segments
        
        audio_segments = []
        
        for i in range(self.checkpointing_segments):
            start = i * segment_length
            end = (i + 1) * segment_length if i < self.checkpointing_segments - 1 else T
            
            latent_segment = latent[:, :, start:end]
            
            # Segment skip features accordingly
            skip_segment = []
            for skip_feature in skip_features:
                # Calculate corresponding segment for each skip feature layer
                skip_start = start * (skip_feature.shape[-1] // T)
                skip_end = end * (skip_feature.shape[-1] // T)
                skip_segment.append(skip_feature[:, :, skip_start:skip_end])
            
            # Use checkpointing for each segment
            audio_seg = checkpoint.checkpoint(
                self.decoder, latent_segment, skip_segment, use_reentrant=False
            )
            
            audio_segments.append(audio_seg)
        
        # Combine segments
        audio = torch.cat(audio_segments, dim=-1)
        
        return audio
    
    def forward(
        self,
        audio: torch.Tensor,
        return_loss: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Complete forward pass with memory optimization
        
        Args:
            audio: (B, 2, T) input stereo audio
            return_loss: whether to return loss components
        """
        # Store original length
        original_length = audio.shape[-1]
        
        # Memory-optimized encode with enhanced checkpointing
        latent, skip_features = self.encode(audio)
        
        # Vector quantization (if enabled)
        vq_loss = torch.tensor(0.0, device=audio.device)
        if self.use_vq:
            latent, vq_loss, _ = self.quantizer(latent)
        
        # Memory-optimized decode with enhanced checkpointing
        reconstructed = self.decode(latent, skip_features)
        
        # Ensure reconstructed audio has the same length as input
        if reconstructed.shape[-1] != original_length:
            if reconstructed.shape[-1] > original_length:
                # Trim if longer
                reconstructed = reconstructed[..., :original_length]
            else:
                # Pad if shorter
                pad_length = original_length - reconstructed.shape[-1]
                reconstructed = F.pad(reconstructed, (0, pad_length), mode='reflect')
        
        if return_loss:
            # Enhanced reconstruction loss
            stft_loss = self.stft_loss(reconstructed, audio)
            
            # Time domain loss
            time_loss = F.l1_loss(reconstructed, audio)
              # Total loss
            total_loss = stft_loss + 0.1 * time_loss + 0.02 * vq_loss
            
            loss_dict = {
                'total_loss': total_loss,
                'stft_loss': stft_loss,
                'time_loss': time_loss,
                'vq_loss': vq_loss
            }
            
            return reconstructed, loss_dict
        
        return reconstructed
    
    def separate_channels(
        self, 
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate vocal and instrumental latent channels"""
        if self.latent_channels >= 8:
            # Standard case: split into 4+4 channels
            vocal_latent = latent[:, :4]
            inst_latent = latent[:, 4:8]
        else:
            # For smaller models, split evenly
            half_channels = self.latent_channels // 2
            vocal_latent = latent[:, :half_channels]
            inst_latent = latent[:, half_channels:half_channels*2]
        
        return vocal_latent, inst_latent
    
    def encode_with_separation(
        self, 
        audio: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Encode with explicit channel separation"""
        full_latent, skip_features = self.encode(audio)
        vocal_latent, inst_latent = self.separate_channels(full_latent)
        
        return {
            'full': full_latent,
            'vocal': vocal_latent,
            'instrumental': inst_latent,
            'skip_features': skip_features
        }
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio"""
        return 32.0
    
    def estimate_latent_shape(self, audio_length: int) -> Tuple[int, int]:
        """Estimate latent shape for given audio length"""
        latent_length = audio_length // 32
        return (self.latent_channels, latent_length)
    
    def get_memory_stats(self) -> Dict[str, str]:
        """Get memory optimization configuration"""
        return {
            'chunk_size': self.chunk_size,
            'use_checkpointing': self.use_checkpointing,
            'memory_efficient': self.memory_efficient,
            'checkpointing_segments': self.checkpointing_segments,
            'chunked_discretization': 'enabled',
            'enhanced_checkpointing': 'enabled'
        }


# ==================== Model Factory ====================

def create_memory_optimized_lyro_dcae(
    model_size: str = "base",
    sample_rate: int = 44100,
    use_vq: bool = False,
    use_weight_norm: bool = True,
    encoder_base_channels: int = None,
    decoder_base_channels: int = None,
    dropout: float = 0.1,
    use_multiscale_ssm: bool = True,
    d_state: int = 64,
    # Memory optimization parameters  
    chunk_size: int = 1024,
    use_checkpointing: bool = True,
    memory_efficient: bool = True,
    checkpointing_segments: int = 4,
    **kwargs
) -> MemoryOptimizedLyroMusicDCAE:
    """Create Memory-Optimized SSM-based Lyro DCAE model"""
    
    if model_size == "small":
        base_config = {
            "encoder_base_channels": encoder_base_channels or 48,
            "decoder_base_channels": decoder_base_channels or 48,
            "latent_channels": 6,
        }
        effective_d_state = 32
        effective_chunk_size = 512  # Smaller chunks for small model
    elif model_size == "base":
        base_config = {
            "encoder_base_channels": encoder_base_channels or 64,
            "decoder_base_channels": decoder_base_channels or 64,
            "latent_channels": 8,
        }
        effective_d_state = d_state
        effective_chunk_size = chunk_size
    elif model_size == "large":
        base_config = {
            "encoder_base_channels": encoder_base_channels or 96,
            "decoder_base_channels": decoder_base_channels or 96,
            "latent_channels": 12,
        }
        effective_d_state = 128
        effective_chunk_size = chunk_size * 2  # Larger chunks for large model
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # Get all parameters that MemoryOptimizedLyroMusicDCAE.__init__ accepts
    model_init_params = {
        'sample_rate', 'latent_channels', 'use_vector_quantization', 'vq_num_embeddings',
        'vq_commitment_cost', 'dual_channel_processing', 'encoder_base_channels', 
        'decoder_base_channels', 'dropout', 'use_weight_norm', 'use_multiscale_ssm',
        'd_state', 'chunk_size', 'use_checkpointing', 'memory_efficient', 'checkpointing_segments'
    }
    
    # Combine base config with kwargs, filtering out non-model parameters
    final_config = {}
    final_config.update(base_config)
    final_config.update({k: v for k, v in kwargs.items() if k in model_init_params})
    
    return MemoryOptimizedLyroMusicDCAE(
        sample_rate=sample_rate,
        use_vector_quantization=use_vq,
        use_weight_norm=use_weight_norm,
        dropout=dropout,
        use_multiscale_ssm=use_multiscale_ssm,
        d_state=effective_d_state,
        chunk_size=effective_chunk_size,
        use_checkpointing=use_checkpointing,
        memory_efficient=memory_efficient,
        checkpointing_segments=checkpointing_segments,
        **final_config
    )


# For backward compatibility
def create_enhanced_lyro_dcae(*args, **kwargs):
    """Backward compatibility wrapper"""
    return create_memory_optimized_lyro_dcae(*args, **kwargs)


LyroMusicDCAE = MemoryOptimizedLyroMusicDCAE  # Backward compatibility alias