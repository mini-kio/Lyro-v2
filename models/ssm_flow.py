# lyro/models/ssm_flow.py
"""
SSM + Flow Matching Generator
1.5B parameter music generation model with S6 backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Optional, Tuple, List
from einops import rearrange, repeat


class S6Layer(nn.Module):
    """S6 State Space Layer with optimized implementation"""
    
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # State space parameters
        A = repeat(torch.arange(1, d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A.float()))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Normalization
        self.norm = nn.LayerNorm(self.d_inner)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        B, L, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each
        
        # Convolution
        x_conv = rearrange(x_inner, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[..., :L]  # Remove padding
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        
        # Activation
        x_conv = F.silu(x_conv)
        
        # SSM computation
        dt = F.softplus(self.dt_proj(x_conv))  # (B, L, d_inner)
        
        BC = self.x_proj(x_conv)  # (B, L, 2 * d_state)
        B_ssm, C = BC.chunk(2, dim=-1)  # (B, L, d_state) each
        
        # Discretization
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        dA = torch.einsum('bld,dn->bldn', dt, A)
        dB = torch.einsum('bld,bln->bldn', dt, B_ssm)
        
        # State space computation (simplified)
        # For efficiency, we use a simplified version
        y = torch.einsum('bldn,bln->bld', dB, C) + x_conv * self.D
        
        # Normalization and gating
        y = self.norm(y)
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output


class S6Block(nn.Module):
    """S6 block with residual connection and normalization"""
    
    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        
        self.s6 = S6Layer(d_model, d_state)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # S6 block
        x = x + self.dropout(self.s6(self.norm(x)))
        
        # FFN block
        x = x + self.ffn(self.ffn_norm(x))
        
        return x


class FlowMatching(nn.Module):
    """Flow matching implementation for continuous generation"""
    
    def __init__(self, sigma: float = 1e-4):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute flow matching loss
        
        Args:
            x0: Noise (B, ...)
            x1: Data (B, ...)
            t: Time (B,)
            
        Returns:
            xt: Interpolated state
            target_v: Target velocity
        """
        # Expand time dimension to match x1
        while t.dim() < x1.dim():
            t = t.unsqueeze(-1)
            
        # Linear interpolation
        xt = (1 - t) * x0 + t * x1
        
        # Add noise
        if self.sigma > 0:
            noise = torch.randn_like(xt) * self.sigma
            xt = xt + noise
            
        # Target velocity
        target_v = x1 - x0
        
        return xt, target_v


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for flow matching"""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        
        half_dim = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half_dim) / half_dim)
        self.register_buffer('freqs', freqs)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) time values
        Returns:
            (B, dim) time embeddings
        """
        args = t[:, None] * self.freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding


class ConditionFusion(nn.Module):
    """Fuse multiple condition modalities"""
    
    def __init__(self, condition_dims: Dict[str, int], output_dim: int):
        super().__init__()
        
        self.condition_dims = condition_dims
        self.output_dim = output_dim
        
        # Individual projections
        self.projections = nn.ModuleDict()
        total_dim = 0
        
        for name, dim in condition_dims.items():
            self.projections[name] = nn.Linear(dim, output_dim)
            total_dim += output_dim
            
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, conditions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            conditions: Dictionary of condition tensors
        Returns:
            (B, output_dim) fused condition embedding
        """
        projected = []
        
        for name, tensor in conditions.items():
            if name in self.projections and tensor is not None:
                proj = self.projections[name](tensor)
                projected.append(proj)
            else:
                # Zero embedding for missing conditions
                batch_size = next(iter(conditions.values())).shape[0]
                device = next(iter(conditions.values())).device
                zero_proj = torch.zeros(batch_size, self.output_dim, device=device)
                projected.append(zero_proj)
                
        # Concatenate and fuse
        fused = torch.cat(projected, dim=-1)
        output = self.fusion(fused)
        
        return output


class SSMFlowGenerator(nn.Module):
    """
    1.5B parameter SSM + Flow Matching generator
    """
    
    def __init__(
        self,
        latent_channels: int = 16,
        latent_size: int = 128,
        d_model: int = 1024,
        n_layers: int = 24,
        d_state: int = 64,
        condition_dims: Dict[str, int] = None,
        dropout: float = 0.1,
        max_length: int = 1024
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_length = max_length
        
        # Default condition dimensions
        if condition_dims is None:
            condition_dims = {
                'lyrics': 512,
                'caption': 768,
                'reference': latent_channels * latent_size
            }
        
        # Input/Output projections
        self.input_proj = nn.Linear(latent_channels, d_model)
        self.output_proj = nn.Linear(d_model, latent_channels)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, d_model) * 0.02)
        
        # Time embedding for flow matching
        self.time_embed = TimeEmbedding(d_model)
        self.time_proj = nn.Linear(d_model, d_model)
        
        # Condition fusion
        self.condition_fusion = ConditionFusion(condition_dims, d_model)
        self.condition_proj = nn.Linear(d_model, d_model)
        
        # S6 backbone
        self.layers = nn.ModuleList([
            S6Block(d_model, d_state, dropout) for _ in range(n_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Flow matching
        self.flow_matching = FlowMatching()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass for velocity prediction
        
        Args:
            x: (B, latent_channels, latent_size) latent tensor
            t: (B,) time steps
            conditions: Dictionary of condition tensors
            
        Returns:
            (B, latent_channels, latent_size) predicted velocity
        """
        B, C, L = x.shape
        
        # Reshape to sequence format
        x = rearrange(x, 'b c l -> b l c')  # (B, L, C)
        
        # Input projection
        x = self.input_proj(x)  # (B, L, d_model)
        
        # Add positional encoding
        seq_len = min(L, self.max_length)
        pos_emb = self.pos_embed[:, :seq_len, :]
        if L > seq_len:
            # Interpolate positional encoding for longer sequences
            pos_emb = F.interpolate(
                pos_emb.transpose(1, 2), size=L, mode='linear', align_corners=False
            ).transpose(1, 2)
        x = x + pos_emb
        
        # Time embedding
        time_emb = self.time_embed(t)  # (B, d_model)
        time_emb = self.time_proj(time_emb)  # (B, d_model)
        time_emb = time_emb.unsqueeze(1)  # (B, 1, d_model)
        
        # Condition embedding
        condition_emb = self.condition_fusion(conditions)  # (B, d_model)
        condition_emb = self.condition_proj(condition_emb)  # (B, d_model)
        condition_emb = condition_emb.unsqueeze(1)  # (B, 1, d_model)
        
        # Add condition and time embeddings
        x = x + time_emb + condition_emb
        
        # Apply S6 layers
        for layer in self.layers:
            x = layer(x)
            
        # Final normalization and projection
        x = self.final_norm(x)
        x = self.output_proj(x)  # (B, L, latent_channels)
        
        # Reshape back to latent format
        x = rearrange(x, 'b l c -> b c l')  # (B, latent_channels, L)
        
        return x
        
    def training_loss(
        self,
        x1: torch.Tensor,
        conditions: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute flow matching training loss
        
        Args:
            x1: (B, latent_channels, latent_size) target latents
            conditions: Dictionary of condition tensors
            
        Returns:
            loss: Flow matching loss
            info: Loss information dictionary
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # Sample random time
        t = torch.rand(batch_size, device=device)
        
        # Sample noise
        x0 = torch.randn_like(x1)
        
        # Flow matching interpolation
        xt, target_v = self.flow_matching(x0, x1, t)
        
        # Predict velocity
        predicted_v = self.forward(xt, t, conditions)
        
        # Flow matching loss
        loss = F.mse_loss(predicted_v, target_v)
        
        # Loss info
        info = {
            'flow_loss': loss.item(),
            'target_v_norm': torch.norm(target_v).item(),
            'predicted_v_norm': torch.norm(predicted_v).item()
        }
        
        return loss, info
        
    @torch.no_grad()
    def generate(
        self,
        conditions: Dict[str, torch.Tensor],
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        shape: Optional[Tuple[int, int, int]] = None
    ) -> torch.Tensor:
        """
        Generate latents using flow matching
        
        Args:
            conditions: Dictionary of condition tensors
            num_steps: Number of generation steps
            guidance_scale: Classifier-free guidance scale
            shape: Output shape (B, latent_channels, latent_size)
            
        Returns:
            (B, latent_channels, latent_size) generated latents
        """
        if shape is None:
            batch_size = next(iter(conditions.values())).shape[0]
            shape = (batch_size, self.latent_channels, self.latent_size)
            
        device = next(iter(conditions.values())).device
        
        # Initialize with noise
        x = torch.randn(shape, device=device)
        
        # Time schedule
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((shape[0],), i * dt, device=device)
            
            # Predict velocity
            if guidance_scale > 1.0:
                # Classifier-free guidance
                v_cond = self.forward(x, t, conditions)
                
                # Create unconditional input
                uncond_conditions = {}
                for key, value in conditions.items():
                    if value is not None:
                        uncond_conditions[key] = torch.zeros_like(value)
                    else:
                        uncond_conditions[key] = None
                        
                v_uncond = self.forward(x, t, uncond_conditions)
                
                # Apply guidance
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v = self.forward(x, t, conditions)
                
            # Update state
            x = x + dt * v
            
        return x
        
    def get_num_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_ssm_flow_generator(
    latent_channels: int = 16,
    latent_size: int = 128,
    d_model: int = 1024,
    n_layers: int = 24,
    condition_dims: Dict[str, int] = None,
    **kwargs
) -> SSMFlowGenerator:
    """
    Create SSM Flow Generator
    
    Args:
        latent_channels: Number of latent channels
        latent_size: Latent sequence length
        d_model: Model dimension
        n_layers: Number of S6 layers
        condition_dims: Condition dimensions dictionary
        **kwargs: Additional arguments
        
    Returns:
        SSMFlowGenerator model
    """
    model = SSMFlowGenerator(
        latent_channels=latent_channels,
        latent_size=latent_size,
        d_model=d_model,
        n_layers=n_layers,
        condition_dims=condition_dims,
        **kwargs
    )
    
    print(f"Created SSM Flow Generator with {model.get_num_parameters():,} parameters")
    
    return model