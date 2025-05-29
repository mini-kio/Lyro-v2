# lyro/dcae/model.py
"""
Enhanced Lyro Music DCAE Implementation
Improvements: FIR Low-pass, Weight Norm, Extended STFT, Enhanced Skip Connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Tuple, Optional, List, Dict, Union
import math
from pathlib import Path


# ==================== Enhanced Advanced Convolution Blocks ====================

class AdvancedLayerNorm(nn.Module):
    """Advanced layer normalization supporting multiple data formats"""
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class ResidualBlock1D(nn.Module):
    """Advanced residual block for 1D convolutions with Weight Norm"""
    
    def __init__(
        self, 
        channels: int, 
        kernel_size: int = 3, 
        dilation: int = 1,
        dropout: float = 0.1,
        activation: str = "swish",
        use_weight_norm: bool = True  # T-2: Weight Normalization
    ):
        super().__init__()
        
        padding = (kernel_size * dilation - dilation) // 2
        
        # T-2: Apply Weight Normalization to Conv layers
        conv1 = nn.Conv1d(channels, channels, kernel_size, 
                         dilation=dilation, padding=padding)
        conv2 = nn.Conv1d(channels, channels, kernel_size,
                         dilation=dilation, padding=padding)
        
        if use_weight_norm:
            conv1 = nn.utils.weight_norm(conv1)
            conv2 = nn.utils.weight_norm(conv2)
        
        self.conv1 = conv1
        self.norm1 = nn.GroupNorm(min(32, channels//4), channels)
        
        self.conv2 = conv2
        self.norm2 = nn.GroupNorm(min(32, channels//4), channels)
        
        self.dropout = nn.Dropout(dropout)
        
        if activation == "swish":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        x = x + residual
        x = self.activation(x)
        
        return x


class MultiScaleConv1D(nn.Module):
    """Multi-scale convolution for better feature extraction with Weight Norm"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [3, 7, 11],
        stride: int = 1,
        padding_mode: str = "reflect",
        use_weight_norm: bool = True  # T-2: Weight Normalization
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        branch_channels = out_channels // len(kernel_sizes)
        
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            conv = nn.Conv1d(in_channels, branch_channels, kernel_size,
                           stride=stride, padding=padding, padding_mode=padding_mode)
            
            # T-2: Apply Weight Normalization
            if use_weight_norm:
                conv = nn.utils.weight_norm(conv)
            
            self.convs.append(nn.Sequential(
                conv,
                nn.GroupNorm(min(8, branch_channels//4), branch_channels),
                nn.SiLU()
            ))
        
        # Adjust final channel count
        total_branch_channels = branch_channels * len(kernel_sizes)
        channel_adjust = nn.Conv1d(total_branch_channels, out_channels, 1)
        
        if use_weight_norm:
            channel_adjust = nn.utils.weight_norm(channel_adjust)
        
        self.channel_adjust = channel_adjust if total_branch_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        branch_outputs = [conv(x) for conv in self.convs]
        combined = torch.cat(branch_outputs, dim=1)
        return self.channel_adjust(combined)


class AttentionBlock1D(nn.Module):
    """Self-attention block for 1D sequences"""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(min(32, channels//4), channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, x):
        B, C, T = x.shape
        residual = x
        
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, T, C)
        
        # Self-attention
        x, _ = self.attention(x, x, x)
        
        x = x.transpose(1, 2)  # (B, C, T)
        return x + residual


# ==================== Enhanced Encoder with Skip Connections ====================

class LyroEncoder(nn.Module):
    """Advanced encoder with multi-scale processing and skip connections"""
    
    def __init__(
        self,
        input_channels: int = 2,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4, 8, 16],
        kernel_sizes: List[int] = [7, 5, 5, 3, 3],
        strides: List[int] = [2, 2, 2, 2, 2],
        num_res_blocks: List[int] = [2, 2, 3, 3, 2],
        latent_channels: int = 8,
        use_attention: List[bool] = [False, False, True, True, False],
        dropout: float = 0.1,
        use_weight_norm: bool = True  # T-2: Weight Normalization
    ):
        super().__init__()
        
        assert len(channel_multipliers) == len(kernel_sizes) == len(strides) == len(num_res_blocks)
        
        self.num_stages = len(channel_multipliers)
        
        # Initial convolution with Weight Norm
        stem_conv = nn.Conv1d(input_channels, base_channels, 7, padding=3, padding_mode="reflect")
        if use_weight_norm:
            stem_conv = nn.utils.weight_norm(stem_conv)
        
        self.stem = nn.Sequential(
            stem_conv,
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        
        # Encoder stages
        self.stages = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(self.num_stages):
            out_channels = base_channels * channel_multipliers[i]
            
            # Downsampling layer
            if i == 0:
                downsample = nn.Identity()
            else:
                downsample = MultiScaleConv1D(
                    current_channels, out_channels,
                    kernel_sizes=[kernel_sizes[i]],
                    stride=strides[i],
                    use_weight_norm=use_weight_norm
                )
            
            # Residual blocks
            res_blocks = nn.ModuleList()
            for j in range(num_res_blocks[i]):
                res_blocks.append(ResidualBlock1D(
                    out_channels, 
                    kernel_size=3,
                    dilation=2**min(j, 3),  # Increasing dilation
                    dropout=dropout,
                    use_weight_norm=use_weight_norm
                ))
            
            # Attention block
            if use_attention[i] and out_channels >= 128:
                attention = AttentionBlock1D(out_channels)
            else:
                attention = nn.Identity()
            
            stage = nn.ModuleDict({
                'downsample': downsample,
                'res_blocks': res_blocks,
                'attention': attention
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
            nn.GroupNorm(8, latent_channels * 2),
            nn.SiLU(),
            final_conv2
        )
        
        self.apply(self._init_weights)
    
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
        Encode audio to latent representation with skip connections
        
        Args:
            x: (B, 2, T) stereo audio
        Returns:
            latent: (B, 8, T//32) compressed latent
            skip_features: List of skip connection features for decoder
        """
        x = self.stem(x)
        
        skip_features = []
        
        for stage in self.stages:
            x = stage['downsample'](x)
            
            for res_block in stage['res_blocks']:
                x = res_block(x)
            
            x = stage['attention'](x)
            
            # A-2: Store skip connection features
            skip_features.append(x.clone())
        
        latent = self.final_conv(x)
        
        return latent, skip_features


# ==================== Enhanced Upsampling with Anti-Aliasing ====================

class AdvancedUpsampling(nn.Module):
    """Advanced upsampling with anti-aliasing and Weight Norm"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        kernel_size: int = 8,
        use_pixel_shuffle: bool = False,
        use_weight_norm: bool = True,  # T-2: Weight Normalization
        sample_rate: int = 44100  # A-3: For anti-aliasing filter
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.sample_rate = sample_rate
        
        if use_pixel_shuffle and scale_factor == 2:
            # Sub-pixel convolution
            conv = nn.Conv1d(in_channels, out_channels * scale_factor, 3, padding=1)
            if use_weight_norm:
                conv = nn.utils.weight_norm(conv)
            
            self.upsample = nn.Sequential(
                conv,
                nn.GLU(dim=1),  # Gated activation
            )
            self.use_pixel_shuffle = True
        else:
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
            self.use_pixel_shuffle = False
        
        # A-3: Anti-aliasing filter for upsampling
        if scale_factor > 1:
            cutoff_freq = 0.9 * (sample_rate / 2) / scale_factor
            self.anti_alias_filter = torchaudio.transforms.LowpassBiquad(
                sample_rate=sample_rate, 
                cutoff_freq=cutoff_freq
            )
        else:
            self.anti_alias_filter = nn.Identity()
        
        self.norm = nn.GroupNorm(min(32, out_channels//4), out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        x = self.upsample(x)
        
        if self.use_pixel_shuffle:
            # Rearrange for sub-pixel convolution
            B, C, T = x.shape
            x = x.view(B, C // self.scale_factor, self.scale_factor, T)
            x = x.permute(0, 1, 3, 2).contiguous()
            x = x.view(B, C // self.scale_factor, T * self.scale_factor)
        
        # A-3: Apply anti-aliasing filter after upsampling
        if self.scale_factor > 1:
            # Apply anti-aliasing per channel
            x_filtered = []
            for i in range(x.shape[1]):
                x_ch = self.anti_alias_filter(x[:, i:i+1])
                x_filtered.append(x_ch)
            x = torch.cat(x_filtered, dim=1)
        
        x = self.norm(x)
        x = self.activation(x)
        
        return x


# ==================== Enhanced Decoder with Skip Connections ====================

class LyroDecoder(nn.Module):
    """Advanced decoder with high-quality upsampling and skip connections"""
    
    def __init__(
        self,
        latent_channels: int = 8,
        base_channels: int = 64,
        channel_multipliers: List[int] = [16, 8, 4, 2, 1],
        kernel_sizes: List[int] = [8, 8, 6, 6, 7],
        scale_factors: List[int] = [2, 2, 2, 2, 2],
        num_res_blocks: List[int] = [2, 3, 3, 2, 2],
        output_channels: int = 2,
        use_attention: List[bool] = [False, True, True, False, False],
        use_pixel_shuffle: List[bool] = [False, True, True, False, False],
        dropout: float = 0.1,
        use_weight_norm: bool = True,  # T-2: Weight Normalization
        sample_rate: int = 44100  # A-3: For anti-aliasing
    ):
        super().__init__()
        
        assert len(channel_multipliers) == len(kernel_sizes) == len(scale_factors) == len(num_res_blocks)
        
        self.num_stages = len(channel_multipliers)
        
        # Initial projection from latent space
        initial_channels = base_channels * channel_multipliers[0]
        initial_conv = nn.Conv1d(latent_channels, initial_channels, 3, padding=1)
        
        if use_weight_norm:
            initial_conv = nn.utils.weight_norm(initial_conv)
        
        self.initial_conv = nn.Sequential(
            initial_conv,
            nn.GroupNorm(min(32, initial_channels//4), initial_channels),
            nn.SiLU()
        )
        
        # A-2: Skip connection projection layers
        self.skip_projections = nn.ModuleList()
        
        # Decoder stages
        self.stages = nn.ModuleList()
        current_channels = initial_channels
        
        for i in range(self.num_stages):
            if i == self.num_stages - 1:
                out_channels = output_channels
            else:
                out_channels = base_channels * channel_multipliers[i + 1]
            
            # A-2: Skip connection projection (except for last stage)
            if i < self.num_stages - 1:
                skip_proj = nn.Conv1d(current_channels * 2, current_channels, 1)  # *2 for concatenation
                if use_weight_norm:
                    skip_proj = nn.utils.weight_norm(skip_proj)
                self.skip_projections.append(skip_proj)
            else:
                self.skip_projections.append(nn.Identity())
            
            # Upsampling layer
            upsample = AdvancedUpsampling(
                current_channels, out_channels,
                scale_factor=scale_factors[i],
                kernel_size=kernel_sizes[i],
                use_pixel_shuffle=use_pixel_shuffle[i],
                use_weight_norm=use_weight_norm,
                sample_rate=sample_rate
            )
            
            # Residual blocks (except for last stage)
            if i < self.num_stages - 1:
                res_blocks = nn.ModuleList()
                for j in range(num_res_blocks[i]):
                    res_blocks.append(ResidualBlock1D(
                        out_channels,
                        kernel_size=3,
                        dilation=2**min(j, 2),
                        dropout=dropout,
                        use_weight_norm=use_weight_norm
                    ))
                
                # Attention block
                if use_attention[i] and out_channels >= 128:
                    attention = AttentionBlock1D(out_channels)
                else:
                    attention = nn.Identity()
            else:
                res_blocks = nn.ModuleList()
                attention = nn.Identity()
            
            stage = nn.ModuleDict({
                'upsample': upsample,
                'res_blocks': res_blocks,
                'attention': attention
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
        Decode latent to audio with skip connections
        
        Args:
            latent: (B, 8, T//32) compressed latent
            skip_features: List of skip connection features from encoder
        Returns:
            audio: (B, 2, T) reconstructed stereo audio
        """
        x = self.initial_conv(latent)
        
        # Reverse skip features order (decoder goes from deep to shallow)
        skip_features = skip_features[::-1]
        
        for i, stage in enumerate(self.stages):
            # A-2: Apply skip connections (except for last stage)
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
            
            x = stage['upsample'](x)
            
            for res_block in stage['res_blocks']:
                x = res_block(x)
            
            x = stage['attention'](x)
        
        audio = self.final_conv(x)
        
        return audio


# ==================== Enhanced Multi-Resolution STFT Loss ====================

class MultiResolutionSTFTLoss(nn.Module):
    """Enhanced Multi-resolution STFT loss with extended frequency coverage"""
    
    def __init__(
        self,
        # A-5: Extended STFT resolutions for better frequency coverage
        fft_sizes: List[int] = [4096, 2048, 1024, 512, 256, 128, 64],
        hop_sizes: Optional[List[int]] = None,
        win_sizes: Optional[List[int]] = None,
        w_sc: float = 1.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.5,  # Increased weight for linear magnitude
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
            x_stft = torch.stack(x_stft_list, dim=1)  # (B, C, F, T)
            y_stft = torch.stack(y_stft_list, dim=1)
            
            # Magnitude spectra
            x_mag = torch.abs(x_stft)
            y_mag = torch.abs(y_stft)
            
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
            
            # A-5: Frequency-weighted loss for better high-frequency preservation
            if i < 3:  # Apply to higher resolution STFTs
                freq_weights = torch.linspace(0.5, 2.0, x_mag.shape[2], device=x_mag.device)
                freq_weights = freq_weights.view(1, 1, -1, 1)
                
                weighted_loss = F.l1_loss(x_mag * freq_weights, y_mag * freq_weights)
                total_loss += 0.1 * weighted_loss
        
        return total_loss / len(self.fft_sizes)


# ==================== Enhanced Complete DCAE Model ====================

class LyroMusicDCAE(nn.Module):
    """
    Enhanced Lyro Music DCAE with all improvements applied
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
        use_weight_norm: bool = True,  # T-2: Weight Normalization
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.latent_channels = latent_channels
        self.use_vq = use_vector_quantization
        self.dual_channel_processing = dual_channel_processing
        
        # Enhanced Encoder with skip connections
        self.encoder = LyroEncoder(
            input_channels=2,
            base_channels=encoder_base_channels,
            latent_channels=latent_channels,
            dropout=dropout,
            use_weight_norm=use_weight_norm
        )
        
        # Vector quantization (optional)
        if use_vector_quantization:
            from .enhanced_vq import VectorQuantizer  # Assume VQ is in separate file
            self.quantizer = VectorQuantizer(
                num_embeddings=vq_num_embeddings,
                embedding_dim=latent_channels,
                commitment_cost=vq_commitment_cost
            )
        
        # Enhanced Decoder with skip connections
        self.decoder = LyroDecoder(
            latent_channels=latent_channels,
            base_channels=decoder_base_channels,
            output_channels=2,
            dropout=dropout,
            use_weight_norm=use_weight_norm,
            sample_rate=sample_rate
        )
        
        # Dual-channel processing (learnable channel importance)
        if dual_channel_processing:
            self.vocal_weight = nn.Parameter(torch.ones(4))  # Channels 0-3
            self.inst_weight = nn.Parameter(torch.ones(4))   # Channels 4-7
        
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
        Encode audio to latent representation with skip features
        
        Args:
            audio: (B, 2, T) stereo audio at 44.1kHz
        Returns:
            latent: (B, 8, T//32) compressed latent
            skip_features: List of skip connection features
        """
        latent, skip_features = self.encoder(audio)
        
        # Apply dual-channel weighting
        if self.dual_channel_processing:
            vocal_channels = latent[:, :4] * self.vocal_weight.view(1, 4, 1)
            inst_channels = latent[:, 4:] * self.inst_weight.view(1, 4, 1)
            latent = torch.cat([vocal_channels, inst_channels], dim=1)
        
        return latent, skip_features
    
    def decode(self, latent: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode latent to audio with skip connections
        
        Args:
            latent: (B, 8, T//32) compressed latent
            skip_features: List of skip connection features from encoder
        Returns:
            audio: (B, 2, T) reconstructed stereo audio
        """
        return self.decoder(latent, skip_features)
    
    def forward(
        self,
        audio: torch.Tensor,
        return_loss: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Complete forward pass with all enhancements
        
        Args:
            audio: (B, 2, T) input stereo audio
            return_loss: whether to return loss components
        """
        # Encode with skip connections
        latent, skip_features = self.encode(audio)
        
        # Vector quantization (if enabled)
        vq_loss = torch.tensor(0.0, device=audio.device)
        if self.use_vq:
            latent, vq_loss, _ = self.quantizer(latent)
        
        # Decode with skip connections
        reconstructed = self.decode(latent, skip_features)
        
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
        vocal_latent = latent[:, :4]  # Channels 0-3
        inst_latent = latent[:, 4:]   # Channels 4-7
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
        return 32.0  # 8x time compression × 4x channel compression
    
    def estimate_latent_shape(self, audio_length: int) -> Tuple[int, int]:
        """Estimate latent shape for given audio length"""
        latent_length = audio_length // 32
        return (self.latent_channels, latent_length)


# ==================== Enhanced Model Factory ====================

def create_enhanced_lyro_dcae(
    model_size: str = "base",
    sample_rate: int = 44100,
    use_vq: bool = False,
    use_weight_norm: bool = True,
    **kwargs
) -> LyroMusicDCAE:
    """Create Enhanced Lyro DCAE model with all improvements"""
    
    if model_size == "small":
        config = {
            "encoder_base_channels": 48,
            "decoder_base_channels": 48,
            "latent_channels": 6,
        }
    elif model_size == "base":
        config = {
            "encoder_base_channels": 64,
            "decoder_base_channels": 64,
            "latent_channels": 8,
        }
    elif model_size == "large":
        config = {
            "encoder_base_channels": 96,
            "decoder_base_channels": 96,
            "latent_channels": 12,
        }
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    config.update(kwargs)
    
    return LyroMusicDCAE(
        sample_rate=sample_rate,
        use_vector_quantization=use_vq,
        use_weight_norm=use_weight_norm,
        **config
    )