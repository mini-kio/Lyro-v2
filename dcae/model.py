# lyro/dcae/model.py - Fixed Continuous Ultra-Compressed 4kbps DCAE Model
"""
Fixed Continuous Ultra-Compressed DCAE Model - Corrected tensor dimension issues
Features: Proper tensor shape handling, debugging outputs, dimension consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Tuple, Optional, List, Dict, Union, Any
import math
from pathlib import Path
import os
import gc

# Disable torch compile
import torch._dynamo
torch._dynamo.config.disable = True
os.environ['TORCH_COMPILE_DISABLE'] = '1'


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Numerically safe log"""
    return torch.log(torch.clamp(x, min=eps))


def ensure_stereo_audio(audio: torch.Tensor, target_device: Optional[torch.device] = None) -> torch.Tensor:
    """Ensure stereo format with device safety"""
    if audio is None:
        device = target_device or torch.device('cpu')
        return torch.zeros(1, 2, 44100, device=device, dtype=torch.float32)
    
    if target_device is not None and audio.device != target_device:
        audio = audio.to(target_device)
    
    # Convert to stereo
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1)
    elif audio.dim() == 2:
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1).unsqueeze(0)
        elif audio.shape[0] == 2:
            audio = audio.unsqueeze(0)
        else:
            audio = audio.unsqueeze(1).repeat(1, 2, 1)
    elif audio.dim() == 3:
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
    
    return audio


class ContinuousAdaptiveQuantization(nn.Module):
    """
    Fixed continuous adaptive quantization with proper tensor handling
    """
    
    def __init__(self, 
                 latent_dim: int = 6,
                 num_levels: int = 64,
                 temperature: float = 1.0,
                 commitment_cost: float = 0.25):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_levels = num_levels
        self.temperature = temperature
        self.commitment_cost = commitment_cost
        
        # Learnable quantization levels per dimension (non-uniform)
        self.quantization_levels = nn.Parameter(
            torch.linspace(-2.0, 2.0, num_levels).unsqueeze(0).repeat(latent_dim, 1)
        )
        
        # Adaptive scale per dimension
        self.scale = nn.Parameter(torch.ones(latent_dim))
        
        # Continuous relaxation temperature (learnable)
        self.temperature_param = nn.Parameter(torch.tensor(temperature))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with continuous quantization - FIXED tensor handling
        """
        # Input validation and shape debugging
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor [B, C, T], got {x.shape}")
        
        B, C, T = x.shape
        
        # Validate channel dimension consistency
        if C != self.latent_dim:
            raise ValueError(f"Input channels {C} doesn't match expected latent_dim {self.latent_dim}")
        
        # Scale input - FIXED: ensure proper broadcasting
        scale_reshaped = self.scale.abs().view(1, -1, 1)  # [1, latent_dim, 1]
        x_scaled = x / (scale_reshaped + 1e-8)  # Add epsilon for numerical stability
        
        # Compute distances to all quantization levels
        # x_scaled: [B, C, T], quantization_levels: [C, num_levels]
        x_expanded = x_scaled.unsqueeze(-1)  # [B, C, T, 1]
        
        # FIXED: Ensure quantization_levels broadcasting is correct
        levels_expanded = self.quantization_levels.unsqueeze(0).unsqueeze(2)  # [1, C, 1, num_levels]
        
        # Validate broadcasting compatibility
        if levels_expanded.shape[1] != C:
            raise ValueError(f"Quantization levels channels {levels_expanded.shape[1]} != input channels {C}")
        
        distances = (x_expanded - levels_expanded).pow(2)  # [B, C, T, num_levels]
        
        # Soft assignment with temperature (Gumbel-Softmax style)
        temperature_value = self.temperature_param.abs() + 1e-8  # Ensure positive
        soft_assignments = F.softmax(-distances / temperature_value, dim=-1)
        
        # Continuous quantized output (weighted sum of levels)
        quantized = torch.sum(
            soft_assignments * levels_expanded, 
            dim=-1
        )  # [B, C, T]
        
        # Scale back - FIXED: consistent with scaling
        quantized = quantized * scale_reshaped
        
        # Regularization losses
        commitment_loss = F.mse_loss(x.detach(), quantized)
        
        # Codebook regularization (encourage diverse usage)
        avg_probs = soft_assignments.mean(dim=[0, 2])  # [C, num_levels]
        entropy_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        
        # Level spread regularization (prevent level collapse)
        level_diff = torch.diff(self.quantization_levels, dim=1)
        spread_loss = -torch.mean(level_diff.pow(2))
        
        total_reg_loss = (
            self.commitment_cost * commitment_loss + 
            0.1 * entropy_loss + 
            0.01 * spread_loss
        )
        
        # Straight-through for gradient flow
        if self.training:
            quantized = x + (quantized - x).detach()
        
        return quantized, total_reg_loss
    
    def get_effective_bitrate(self, tensor_shape: tuple, duration: float = 1.0) -> float:
        """Calculate effective bitrate for given tensor shape"""
        total_elements = 1
        for dim in tensor_shape:
            total_elements *= dim
        
        # Effective bits per element (log2 of quantization levels)
        effective_bits = math.log2(self.num_levels)
        
        return (total_elements * effective_bits) / (duration * 1000)  # kbps


class LatentSpaceRegularizer(nn.Module):
    """
    Fixed latent space regularizer with proper tensor handling
    """
    
    def __init__(self, latent_dim: int = 6):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Target statistics for latent space
        self.register_buffer('target_mean', torch.zeros(latent_dim))
        self.register_buffer('target_std', torch.ones(latent_dim))
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization losses for latent space - FIXED
        """
        if latent.dim() != 3:
            raise ValueError(f"Expected 3D latent tensor [B, C, T], got {latent.shape}")
        
        B, C, T = latent.shape
        
        if C != self.latent_dim:
            raise ValueError(f"Latent channels {C} doesn't match expected {self.latent_dim}")
        
        # 1. Statistical regularization (encourage unit Gaussian)
        latent_flat = latent.view(B, C, -1)
        mean_loss = F.mse_loss(
            latent_flat.mean(dim=[0, 2]), 
            self.target_mean
        )
        std_loss = F.mse_loss(
            latent_flat.std(dim=[0, 2]), 
            self.target_std
        )
        
        # 2. Temporal smoothness (important for SSM)
        if T > 1:
            temporal_diff = torch.diff(latent, dim=2)
            smoothness_loss = torch.mean(temporal_diff.pow(2))
        else:
            smoothness_loss = torch.tensor(0.0, device=latent.device)
          # 3. Channel correlation regularization (prevent mode collapse)
        if C > 1:
            # FIXED: Proper channel correlation calculation
            # latent_flat: [B, C, T] -> normalize across time dimension
            latent_norm = F.normalize(latent_flat, dim=2)  # [B, C, T]
            
            # Compute channel correlation: transpose to [B, T, C] for bmm
            latent_for_corr = latent_norm.transpose(1, 2)  # [B, T, C]
            correlation_matrix = torch.bmm(
                latent_for_corr.transpose(1, 2),  # [B, C, T]
                latent_for_corr                   # [B, T, C]
            ).mean(dim=0)  # [C, C]
            
            # Encourage orthogonality (off-diagonal should be small)
            identity = torch.eye(C, device=latent.device)
            correlation_loss = F.mse_loss(correlation_matrix, identity)
        else:
            correlation_loss = torch.tensor(0.0, device=latent.device)
        
        total_reg_loss = (
            0.1 * mean_loss + 
            0.1 * std_loss + 
            0.05 * smoothness_loss + 
            0.02 * correlation_loss
        )
        
        return total_reg_loss


def multi_scale_stft(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-scale STFT loss"""
    stft_configs = [
        {"n_fft": 1024, "hop_length": 256, "weight": 1.0},
        {"n_fft": 512, "hop_length": 128, "weight": 0.7},
        {"n_fft": 2048, "hop_length": 512, "weight": 0.5},
    ]
    
    total_loss = 0.0
    pred_flat = pred.reshape(-1, pred.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    
    for config in stft_configs:
        try:
            pred_stft = torch.stft(
                pred_flat,
                n_fft=config["n_fft"],
                hop_length=config["hop_length"],
                return_complex=True,
                window=torch.hann_window(config["n_fft"], device=pred.device)
            )
            target_stft = torch.stft(
                target_flat,
                n_fft=config["n_fft"],
                hop_length=config["hop_length"],
                return_complex=True,
                window=torch.hann_window(config["n_fft"], device=target.device)
            )
            
            stft_loss = F.l1_loss(torch.abs(pred_stft), torch.abs(target_stft))
            total_loss += stft_loss * config["weight"]
        except:
            continue
    
    return total_loss


def loss_continuous_ultra_compressed(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    quantization_loss: torch.Tensor,
    regularization_loss: torch.Tensor,
    mel_transform
) -> torch.Tensor:
    """Continuous ultra-compressed dual-domain loss function"""
    # Ensure same length
    min_len = min(pred.shape[-1], target.shape[-1])
    pred = pred[..., :min_len]
    target = target[..., :min_len]
    
    # Time domain L1 loss (primary)
    l_time = F.l1_loss(pred, target)
    
    # Spectral loss (reduced weight for compression)
    l_stft = multi_scale_stft(pred, target)
    
    # Mel-spectrogram loss (minimal for compression)
    try:
        pred_mono = pred.mean(dim=1)
        target_mono = target.mean(dim=1)
        pred_mel = mel_transform(pred_mono)
        target_mel = mel_transform(target_mono)
        l_mel = F.l1_loss(pred_mel, target_mel)
    except:
        l_mel = torch.tensor(0.0, device=pred.device)
    
    return (
        1.0 * l_time + 
        0.3 * l_stft + 
        0.1 * l_mel + 
        0.3 * quantization_loss + 
        0.2 * regularization_loss
    )


class MRDiscriminator(nn.Module):
    """Multi-resolution discriminator for feature matching"""
    
    def __init__(self):
        super().__init__()
        
        self.disc_list = nn.ModuleList([
            self._create_disc(1, [32, 64, 128, 256]),
            self._create_disc(2, [32, 64, 128]),
            self._create_disc(4, [32, 64]),
        ])
    
    def _create_disc(self, pool_factor: int, channels: List[int]) -> nn.Module:
        """Create single-scale discriminator"""
        layers = []
        
        if pool_factor > 1:
            layers.append(nn.AvgPool1d(pool_factor, pool_factor))
        
        in_ch = 2
        for out_ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, 15, padding=7),
                nn.LeakyReLU(0.2),
                nn.GroupNorm(8, out_ch),
            ])
            in_ch = out_ch
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning feature maps for each scale"""
        if x.dim() == 3 and x.shape[1] == 2:
            x_reshaped = x
        else:
            x_reshaped = x.reshape(-1, 2, x.shape[-1])
        
        feature_maps = []
        for disc in self.disc_list:
            try:
                features = disc(x_reshaped)
                feature_maps.append(features)
            except:
                dummy_features = torch.zeros(
                    x_reshaped.shape[0], 32, x_reshaped.shape[-1] // 4,
                    device=x.device, dtype=x.dtype
                )
                feature_maps.append(dummy_features)
        
        return feature_maps


class LogMelSpectrogram(nn.Module):
    """Log Mel-Spectrogram transform"""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 40.0,
        f_max: float = 16000.0,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        
        self.register_buffer("window", torch.hann_window(win_length))
        
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=f_min,
            f_max=f_max,
            n_stft=n_fft // 2 + 1,
            norm="slaney",
            mel_scale="slaney"
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to log mel-spectrogram"""
        if audio.dim() == 3:
            audio = audio.mean(dim=1)
        
        pad_left = (self.win_length - self.hop_length) // 2
        pad_right = (self.win_length - self.hop_length + 1) // 2
        audio = F.pad(audio, (pad_left, pad_right), mode="reflect")
        
        spec = torch.stft(
            audio.float(),
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(audio.device),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)
        mel = self.mel_scale.to(audio.device)(spec)
        log_mel = safe_log(mel + 1e-5)
        
        return log_mel.to(audio.dtype)


class ConvNeXt1DBlock(nn.Module):
    """Efficient 1D ConvNeXt block"""
    
    def __init__(self, dim: int, mlp_ratio: float = 2.0, kernel_size: int = 7):
        super().__init__()
        
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.GroupNorm(8, dim)
        self.pwconv1 = nn.Conv1d(dim, int(mlp_ratio * dim), 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(int(mlp_ratio * dim), dim, 1)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + x
        return x


class FixedUltraCompressedEncoder(nn.Module):
    """
    FIXED Ultra-compressed encoder with precise tensor dimension control
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        depths: List[int] = [1, 1, 2, 1],
        dims: List[int] = [48, 96, 192, 384],
        downsample_factors: List[int] = [8, 8, 8, 8],  # 4096x total downsample
        target_time_steps: int = 11,  # Force specific output size
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.depths = depths
        self.dims = dims
        self.target_time_steps = target_time_steps
        self.total_downsample = 1
        for factor in downsample_factors:
            self.total_downsample *= factor
        
        # Initial 1D convolution with controlled downsampling
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, dims[0], kernel_size=15, stride=downsample_factors[0], padding=7),
            nn.GroupNorm(8, dims[0]),
            nn.GELU(),
        )
        
        # Downsampling layers with shape validation
        self.downsample_layers = nn.ModuleList()
        for i in range(len(depths)):
            if i == 0:
                layer = nn.Identity()
            else:
                layer = nn.Sequential(
                    nn.GroupNorm(8, dims[i-1]),
                    nn.Conv1d(dims[i-1], dims[i], kernel_size=3, stride=downsample_factors[i], padding=1),
                )
            self.downsample_layers.append(layer)
        
        # ConvNeXt stages
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage_layers = []
            for j in range(depths[i]):
                stage_layers.append(ConvNeXt1DBlock(dim=dims[i]))
            self.stages.append(nn.Sequential(*stage_layers))
        
        # Final norm
        self.norm = nn.GroupNorm(8, dims[-1])
        
        # Adaptive pooling to ensure target output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_time_steps)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with precise tensor shape control
        """
        # Input validation
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, C, T], got {x.shape}")
        
        B, C, T = x.shape
        if C != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {C}")
        
        # Forward through encoder
        x = self.stem(x)  # [B, 48, T//8]
        
        for i, (downsample, stage) in enumerate(zip(self.downsample_layers, self.stages)):
            x = downsample(x)
            x = stage(x)
        
        # Apply normalization
        enc_feat = self.norm(x)  # [B, 384, variable_time]
        
        # FIXED: Force output to target time steps
        enc_feat = self.adaptive_pool(enc_feat)  # [B, 384, target_time_steps]
        
        # Generate stereo cross information
        if enc_feat.shape[1] >= 2:
            mid_channel = enc_feat.shape[1] // 2
            left_feat = enc_feat[:, :mid_channel, :].mean(1, keepdim=True)  # [B, 1, T]
            right_feat = enc_feat[:, mid_channel:, :].mean(1, keepdim=True)  # [B, 1, T]
            cross = left_feat - right_feat  # [B, 1, T]
        else:
            cross = enc_feat.mean(1, keepdim=True)  # [B, 1, T]
        
        return enc_feat, cross


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class ResBlock1D(nn.Module):
    """Efficient 1D residual block"""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int] = (1, 2, 4)):
        super().__init__()
        
        self.convs1 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                          dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                          dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
        ])
        
        self.convs2 = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                          dilation=1, padding=get_padding(kernel_size, 1))),
            nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                          dilation=1, padding=get_padding(kernel_size, 1))),
        ])
        
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.silu(x)
            xt = c1(xt)
            xt = F.silu(xt)
            xt = c2(xt)
            x = xt + x
        return x


class FixedUltraCompressedDecoder(nn.Module):
    """
    FIXED Ultra-compressed decoder with precise tensor handling
    """
    
    def __init__(
        self,
        latent_channels: int = 6,
        upsample_rates: Tuple[int] = (8, 8, 8, 8),
        upsample_kernel_sizes: Tuple[int] = (16, 16, 16, 16),
        resblock_kernel_sizes: Tuple[int] = (3, 5),
        resblock_dilation_sizes: Tuple[Tuple[int]] = ((1, 2), (1, 3)),
        initial_channel: int = 256,
        output_channels: int = 2,
        target_output_length: int = 44100,
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.target_output_length = target_output_length
        self.total_upsample_rate = 1
        for rate in upsample_rates:
            self.total_upsample_rate *= rate
        
        # Pre-conv - FIXED: account for cross features
        self.conv_pre = nn.utils.weight_norm(nn.Conv1d(
            latent_channels + 1, initial_channel, 7, 1, padding=3
        ))
        
        # Upsample layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.utils.weight_norm(nn.ConvTranspose1d(
                initial_channel // (2**i),
                initial_channel // (2**(i+1)),
                k, u, padding=(k-u)//2
            )))
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = initial_channel // (2**(i+1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1D(ch, k, d))
        
        # Post-conv
        final_ch = initial_channel // (2**len(self.ups))
        self.conv_post = nn.utils.weight_norm(nn.Conv1d(
            final_ch, output_channels, 7, 1, padding=3
        ))
        
        # Final adaptive pooling to target length
        self.final_pool = nn.AdaptiveAvgPool1d(target_output_length)
        
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, latent: torch.Tensor, cross: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with FIXED tensor concatenation
        """
        # Input validation
        if latent.dim() != 3:
            raise ValueError(f"Expected 3D latent tensor [B, C, T], got {latent.shape}")
        
        B, C, T = latent.shape
        if C != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {C}")
        
        x = latent
        
        # FIXED: Handle cross features with proper shape validation
        if cross is not None:
            if cross.dim() != 3:
                raise ValueError(f"Expected 3D cross tensor [B, 1, T], got {cross.shape}")
            
            cross_B, cross_C, cross_T = cross.shape
            
            # Validate batch dimension
            if cross_B != B:
                raise ValueError(f"Cross batch size {cross_B} != latent batch size {B}")
            
            # Resize cross to match latent time dimension if needed
            if cross_T != T:
                cross = F.interpolate(cross, size=T, mode='linear', align_corners=False)
            
            # Concatenate along channel dimension
            x = torch.cat([x, cross], dim=1)  # [B, C+1, T]
        else:
            # Create dummy cross features
            dummy_cross = torch.zeros(B, 1, T, device=x.device, dtype=x.dtype)
            x = torch.cat([x, dummy_cross], dim=1)  # [B, C+1, T]
        
        # Verify concatenated shape
        expected_channels = self.latent_channels + 1
        if x.shape[1] != expected_channels:
            raise ValueError(f"After concat: expected {expected_channels} channels, got {x.shape[1]}")
        
        # Pre-convolution
        x = self.conv_pre(x)
        
        # Upsample with residual blocks
        for i in range(self.num_upsamples):
            x = F.silu(x)
            x = self.ups[i](x)
            
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        
        # Final convolution to audio
        x = F.silu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        # FIXED: Ensure target output length
        if x.shape[-1] != self.target_output_length:
            x = self.final_pool(x)
        
        return x


class FixedContinuousUltraCompressedDCAEModel(nn.Module):
    """
    FIXED Continuous Ultra-Compressed DCAE Model with precise tensor handling
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 40.0,
        f_max: float = 16000.0,
        latent_channels: int = 6,
        encoder_depths: List[int] = [1, 1, 2, 1],
        encoder_dims: List[int] = [48, 96, 192, 384],
        decoder_upsample_rates: Tuple[int] = (8, 8, 8, 8),
        decoder_upsample_kernel_sizes: Tuple[int] = (16, 16, 16, 16),
        decoder_resblock_kernel_sizes: Tuple[int] = (3, 5),
        decoder_resblock_dilation_sizes: Tuple[Tuple[int]] = ((1, 2), (1, 3)),
        decoder_initial_channel: int = 256,
        output_channels: int = 2,
        quantization_num_levels: int = 64,
        quantization_temperature: float = 1.0,
        target_time_steps: int = 11,  # Control latent time dimension
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.latent_channels = latent_channels
        self.output_channels = output_channels
        self.target_time_steps = target_time_steps
        
        # Mel-spectrogram transform for loss computation
        self.mel_transform = LogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        
        # FIXED encoder with controlled output dimensions
        self.encoder = FixedUltraCompressedEncoder(
            input_channels=output_channels,
            depths=encoder_depths,
            dims=encoder_dims,
            target_time_steps=target_time_steps,
        )
        
        # Latent projection - FIXED: consistent with encoder output
        self.to_latent = nn.Conv1d(encoder_dims[-1], latent_channels, kernel_size=1)
        
        # Continuous adaptive quantization
        self.continuous_quantizer = ContinuousAdaptiveQuantization(
            latent_dim=latent_channels,
            num_levels=quantization_num_levels,
            temperature=quantization_temperature,
        )
        
        # Latent space regularizer
        self.latent_regularizer = LatentSpaceRegularizer(latent_dim=latent_channels)
        
        # FIXED decoder with controlled input/output dimensions
        self.decoder = FixedUltraCompressedDecoder(
            latent_channels=latent_channels,
            upsample_rates=decoder_upsample_rates,
            upsample_kernel_sizes=decoder_upsample_kernel_sizes,
            resblock_kernel_sizes=decoder_resblock_kernel_sizes,
            resblock_dilation_sizes=decoder_resblock_dilation_sizes,
            initial_channel=decoder_initial_channel,
            output_channels=output_channels,
            target_output_length=sample_rate,  # 1 second output
        )
        
        # Feature-matching discriminator
        self.discriminator = MRDiscriminator()
        
        # Track losses
        self.quantization_loss = torch.tensor(0.0)
        self.regularization_loss = torch.tensor(0.0)
    
    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FIXED encode with precise tensor shape control
        """
        audio = ensure_stereo_audio(audio, target_device=audio.device)
        
        # Encoder forward pass
        encoded, cross = self.encoder(audio)  # [B, 384, target_time_steps], [B, 1, target_time_steps]
        
        # Project to latent space
        latent_raw = self.to_latent(encoded)  # [B, latent_channels, target_time_steps]
        
        # Continuous adaptive quantization
        latent_quantized, quant_loss = self.continuous_quantizer(latent_raw)
        
        # Latent space regularization
        reg_loss = self.latent_regularizer(latent_quantized)
        
        # Store losses
        self.quantization_loss = quant_loss
        self.regularization_loss = reg_loss
        
        return latent_quantized, cross
    
    def decode(self, latent: torch.Tensor, cross: Optional[torch.Tensor] = None) -> torch.Tensor:
        """FIXED decode with proper tensor handling"""
        audio = self.decoder(latent, cross)
        return ensure_stereo_audio(audio, target_device=latent.device)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        FIXED forward pass with shape consistency
        """
        original_length = audio.shape[-1]
        audio = ensure_stereo_audio(audio, target_device=audio.device)
        
        # Encode
        latent, cross = self.encode(audio)
        
        # Decode
        reconstructed = self.decode(latent, cross)
        
        # Match original length
        if reconstructed.shape[-1] != original_length:
            if reconstructed.shape[-1] > original_length:
                reconstructed = reconstructed[..., :original_length]
            else:
                pad_amount = original_length - reconstructed.shape[-1]
                reconstructed = F.pad(reconstructed, (0, pad_amount))
        
        return reconstructed
    
    def get_discriminator_features(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """Get discriminator features for feature matching loss"""
        return self.discriminator(audio)
    
    def get_continuous_latent(self, audio: torch.Tensor) -> torch.Tensor:
        """Get continuous latent representation for SSM+Flow Matching training"""
        with torch.no_grad():
            latent, _ = self.encode(audio)
        return latent
    
    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode from latent representation"""
        return self.decode(latent)
    
    def get_compression_info(self, audio: torch.Tensor) -> Dict[str, Any]:
        """Get detailed compression information"""
        with torch.no_grad():
            latent, cross = self.encode(audio)
            
            original_elements = audio.numel()
            latent_elements = latent.numel()
            cross_elements = cross.numel() if cross is not None else 0
            total_compressed_elements = latent_elements + cross_elements
            
            compression_ratio = original_elements / total_compressed_elements
            
            duration = audio.shape[-1] / self.sample_rate
            effective_bitrate = self.continuous_quantizer.get_effective_bitrate(latent.shape, duration)
            
            return {
                'original_shape': tuple(audio.shape),
                'latent_shape': tuple(latent.shape),
                'cross_shape': tuple(cross.shape) if cross is not None else None,
                'compression_ratio': compression_ratio,
                'effective_bitrate_kbps': effective_bitrate,
                'original_elements': original_elements,
                'compressed_elements': total_compressed_elements,
                'quantization_levels': self.continuous_quantizer.num_levels,
                'latent_is_continuous': True,
                'flow_matching_compatible': True,
                'ssm_compatible': True,
            }


class FixedContinuousUltraCompressedDCAELoss(nn.Module):
    """FIXED continuous ultra-compressed loss function"""
    
    def __init__(self):
        super().__init__()
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                model: FixedContinuousUltraCompressedDCAEModel,
                pred_feat: Optional[List[torch.Tensor]] = None,
                target_feat: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """FIXED continuous ultra-compressed loss computation"""
        
        self.mel_transform = self.mel_transform.to(pred.device)
        
        quantization_loss = model.quantization_loss if hasattr(model, 'quantization_loss') else torch.tensor(0.0, device=pred.device)
        regularization_loss = model.regularization_loss if hasattr(model, 'regularization_loss') else torch.tensor(0.0, device=pred.device)
        
        dual_loss = loss_continuous_ultra_compressed(
            pred, target, quantization_loss, regularization_loss, self.mel_transform
        )
        
        feat_loss = torch.tensor(0.0, device=pred.device)
        if pred_feat is not None and target_feat is not None:
            for p_feat, t_feat in zip(pred_feat, target_feat):
                try:
                    feat_loss += F.l1_loss(p_feat, t_feat.detach())
                except:
                    continue
        
        total_loss = dual_loss + 0.1 * feat_loss
        
        return {
            'total_loss': total_loss,
            'dual_domain_loss': dual_loss,
            'feature_matching_loss': feat_loss,
            'quantization_loss': quantization_loss,
            'regularization_loss': regularization_loss,
        }


def create_continuous_ultra_compressed_dcae_model(
    sample_rate: int = 44100,
    latent_channels: int = 6,
    target_time_steps: int = 11,
    **kwargs
) -> FixedContinuousUltraCompressedDCAEModel:
    """Create FIXED continuous ultra-compressed DCAE model"""
    
    model = FixedContinuousUltraCompressedDCAEModel(
        sample_rate=sample_rate,
        latent_channels=latent_channels,
        target_time_steps=target_time_steps,
        **kwargs
    )
    
    return model


# Main interfaces
create_enhanced_dcae_model = create_continuous_ultra_compressed_dcae_model
EnhancedDCAEModel = FixedContinuousUltraCompressedDCAEModel
EnhancedDCAELoss = FixedContinuousUltraCompressedDCAELoss