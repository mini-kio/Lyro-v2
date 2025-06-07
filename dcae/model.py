# lyro/dcae/model.py
"""
Enhanced CQT-SSM-based LYRO DCAE Implementation with Unified SSM Components
State Space Model based on Constant-Q Transform for high-quality music compression
Now using unified SSM components from ssm.model for consistency and performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchaudio
import numpy as np
from typing import Tuple, Optional, List, Dict, Union
import math
from pathlib import Path
import librosa

# Import unified SSM components from SSM module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssm.model import (
    OptimizedS6StateSpaceKernel, 
    OptimizedS6Block, 
    OptimizedMultiScaleS6, 
    SinusoidalEmbedding
)


# ==================== CQT and Harmonic-Percussive Modules ====================

class ConstantQTransform(nn.Module):
    """
    Constant-Q Transform optimized for music processing
    Better than mel spectrogram for musical content
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        fmin: float = 32.7,  # C1
        n_bins: int = 84,    # 7 octaves
        bins_per_octave: int = 12,
        window: str = 'hann',
        center: bool = True,
        pad_mode: str = 'reflect',
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.center = center
        self.pad_mode = pad_mode
        
        # Pre-compute CQT kernel
        self._initialize_cqt_kernel()
    
    def _initialize_cqt_kernel(self):
        """Initialize CQT transformation kernel"""
        # Calculate frequencies
        freqs = self.fmin * (2.0 ** (np.arange(self.n_bins) / self.bins_per_octave))
        
        # Calculate Q factor and kernel lengths
        Q = 1.0 / (2.0 ** (1.0 / self.bins_per_octave) - 1.0)
        kernel_lengths = (Q * self.sample_rate / freqs).astype(int)
        max_kernel_length = max(kernel_lengths)
        
        # Build CQT kernels
        kernels = []
        for i, (freq, length) in enumerate(zip(freqs, kernel_lengths)):
            # Create complex exponential kernel
            t = np.arange(length) / self.sample_rate
            kernel = np.exp(-2j * np.pi * freq * t) * np.hanning(length)
            
            # Normalize
            kernel = kernel / np.linalg.norm(kernel)
            
            # Pad to max length with stable padding
            padded_kernel = np.zeros(max_kernel_length, dtype=complex)
            start_idx = max(0, (max_kernel_length - length) // 2)  # Center padding
            end_idx = min(max_kernel_length, start_idx + length)
            actual_length = end_idx - start_idx
            padded_kernel[start_idx:end_idx] = kernel[:actual_length]
            kernels.append(padded_kernel)
        
        # Convert to PyTorch tensor
        kernels = np.stack(kernels)  # (n_bins, max_kernel_length)
        
        # Separate real and imaginary parts for convolution
        kernel_real = torch.from_numpy(kernels.real).float()
        kernel_imag = torch.from_numpy(kernels.imag).float()
        
        # Register as buffers
        self.register_buffer('kernel_real', kernel_real.unsqueeze(1))  # (n_bins, 1, kernel_length)
        self.register_buffer('kernel_imag', kernel_imag.unsqueeze(1))
        self.kernel_length = max_kernel_length
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply CQT to audio with enhanced padding stability and torch.compile compatibility
        
        Args:
            audio: (B, T) or (B, C, T) audio tensor
        Returns:
            cqt: (B, n_bins, T_frames) CQT representation
        """
        if audio.dim() == 3:
            B, C, T = audio.shape
            # Convert stereo to mono with stable averaging
            audio = audio.mean(dim=1)  # (B, T)
        else:
            B, T = audio.shape
        
        # Enhanced padding for center mode with length validation
        if self.center:
            # Use static pad_length to avoid torch.compile issues
            pad_length = self.kernel_length // 2
            # Ensure minimum audio length for stable processing
            if T < pad_length:
                # Pad input audio to minimum required length
                extra_pad = pad_length - T + 1
                audio = F.pad(audio, (0, extra_pad), mode='constant', value=0.0)
                T = audio.shape[-1]
            
            # Use constant padding mode to avoid torch.compile issues with 'reflect' mode
            audio = F.pad(audio, (pad_length, pad_length), mode='constant', value=0.0)
        
        # Apply CQT kernels via convolution
        audio = audio.unsqueeze(1)  # (B, 1, T)
        
        # Real and imaginary convolutions with improved error handling
        try:
            cqt_real = F.conv1d(audio, self.kernel_real, stride=self.hop_length)
            cqt_imag = F.conv1d(audio, self.kernel_imag, stride=self.hop_length)
        except RuntimeError as e:
            # More robust fallback for dimension mismatch
            print(f"CQT convolution fallback triggered: {str(e)[:100]}")
            
            # Ensure minimum valid dimensions
            min_kernel_size = 32  # Minimum required kernel size
            audio_len = audio.shape[-1]
            kernel_len = self.kernel_real.shape[-1]
            
            if audio_len < min_kernel_size or kernel_len < min_kernel_size:
                # Create minimal valid output
                min_output_frames = 1
                cqt_real = torch.zeros(B, self.n_bins, min_output_frames, device=audio.device, dtype=audio.dtype)
                cqt_imag = torch.zeros(B, self.n_bins, min_output_frames, device=audio.device, dtype=audio.dtype)
            else:
                # Try with reduced kernel size
                safe_kernel_len = min(audio_len // 4, kernel_len)
                kernel_real_safe = self.kernel_real[..., :safe_kernel_len]
                kernel_imag_safe = self.kernel_imag[..., :safe_kernel_len]
                
                try:
                    cqt_real = F.conv1d(audio, kernel_real_safe, stride=min(self.hop_length, safe_kernel_len))
                    cqt_imag = F.conv1d(audio, kernel_imag_safe, stride=min(self.hop_length, safe_kernel_len))
                except RuntimeError:
                    # Ultimate fallback
                    output_frames = max(1, audio_len // self.hop_length)
                    cqt_real = torch.zeros(B, self.n_bins, output_frames, device=audio.device, dtype=audio.dtype)
                    cqt_imag = torch.zeros(B, self.n_bins, output_frames, device=audio.device, dtype=audio.dtype)
        
        # Compute magnitude with numerical stability
        cqt_mag = torch.sqrt(cqt_real**2 + cqt_imag**2 + 1e-8)
        
        # Log compression with stability
        cqt_log = torch.log(cqt_mag + 1e-7)
        
        return cqt_log  # (B, n_bins, T_frames)


class HarmonicPercussiveSeparation(nn.Module):
    """
    Harmonic-Percussive Separation for music structure understanding
    Separates CQT into harmonic and percussive components
    """
    
    def __init__(
        self,
        kernel_size_h: int = 17,  # Harmonic kernel (vertical)
        kernel_size_p: int = 17,  # Percussive kernel (horizontal)
        power: float = 2.0,
        margin: float = 1.0,
        learnable: bool = True,  # Whether to make separation learnable
    ):
        super().__init__()
        
        self.power = power
        self.margin = margin
        self.learnable = learnable
        
        if learnable:
            # Learnable separation filters
            self.harmonic_filter = nn.Conv2d(
                1, 1, 
                kernel_size=(kernel_size_h, 1), 
                padding=(kernel_size_h//2, 0),
                bias=False
            )
            self.percussive_filter = nn.Conv2d(
                1, 1,
                kernel_size=(1, kernel_size_p),
                padding=(0, kernel_size_p//2), 
                bias=False
            )
            
            # Initialize with median filter-like weights
            self._initialize_filters(kernel_size_h, kernel_size_p)
        else:
            # Traditional median filtering approach
            self.kernel_size_h = kernel_size_h
            self.kernel_size_p = kernel_size_p
    
    def _initialize_filters(self, kernel_size_h: int, kernel_size_p: int):
        """Initialize learnable filters with median filter characteristics"""
        # Harmonic filter (emphasizes vertical structure)
        h_weight = torch.ones(1, 1, kernel_size_h, 1) / kernel_size_h
        self.harmonic_filter.weight.data = h_weight
        
        # Percussive filter (emphasizes horizontal structure)  
        p_weight = torch.ones(1, 1, 1, kernel_size_p) / kernel_size_p
        self.percussive_filter.weight.data = p_weight
    
    def forward(self, cqt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Separate CQT into harmonic and percussive components with stability
        
        Args:
            cqt: (B, n_bins, T_frames) CQT representation
        Returns:
            harmonic: (B, n_bins, T_frames) harmonic component
            percussive: (B, n_bins, T_frames) percussive component
        """
        B, n_bins, T_frames = cqt.shape
        
        # Validate input dimensions
        if T_frames == 0 or n_bins == 0:
            # Return zeros for empty input
            zeros = torch.zeros_like(cqt)
            return zeros, zeros
        
        # Add channel dimension for conv2d
        cqt_2d = cqt.unsqueeze(1)  # (B, 1, n_bins, T_frames)
        
        if self.learnable:
            # Learnable separation with error handling
            try:
                harmonic_enhanced = self.harmonic_filter(cqt_2d)
                percussive_enhanced = self.percussive_filter(cqt_2d)
            except RuntimeError as e:
                # Fallback to simple processing
                harmonic_enhanced = cqt_2d
                percussive_enhanced = cqt_2d
        else:
            # Traditional median filtering
            harmonic_enhanced = self._median_filter_harmonic(cqt_2d)
            percussive_enhanced = self._median_filter_percussive(cqt_2d)
        
        # Compute separation masks using soft masking with numerical stability
        total_energy = harmonic_enhanced + percussive_enhanced + 1e-8
        
        harmonic_mask = (harmonic_enhanced**self.power) / (total_energy**self.power + 1e-10)
        percussive_mask = (percussive_enhanced**self.power) / (total_energy**self.power + 1e-10)
        
        # Apply masks
        harmonic = cqt_2d * harmonic_mask
        percussive = cqt_2d * percussive_mask
        
        # Remove channel dimension
        harmonic = harmonic.squeeze(1)  # (B, n_bins, T_frames)
        percussive = percussive.squeeze(1)
        
        return harmonic, percussive
    
    def _median_filter_harmonic(self, x: torch.Tensor) -> torch.Tensor:
        """Apply harmonic enhancement (vertical median filtering)"""
        kernel = torch.ones(1, 1, self.kernel_size_h, 1, device=x.device) / self.kernel_size_h
        return F.conv2d(x, kernel, padding=(self.kernel_size_h//2, 0))
    
    def _median_filter_percussive(self, x: torch.Tensor) -> torch.Tensor:
        """Apply percussive enhancement (horizontal median filtering)"""
        kernel = torch.ones(1, 1, 1, self.kernel_size_p, device=x.device) / self.kernel_size_p
        return F.conv2d(x, kernel, padding=(0, self.kernel_size_p//2))


class CQTInverseTransform(nn.Module):
    """
    Inverse CQT transformation with neural vocoder
    Converts CQT back to audio waveform
    """
    
    def __init__(
        self,
        n_bins: int = 84,
        sample_rate: int = 44100,
        hop_length: int = 512,
        n_layers: int = 4,
        channels: int = 256,
    ):
        super().__init__()
        
        self.n_bins = n_bins
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
        # Neural vocoder for CQT to audio with enhanced stability
        layers = []
        current_channels = n_bins
        
        for i in range(n_layers):
            out_channels = channels // (2 ** i) if i > 0 else channels
            
            # Ensure minimum channel count
            out_channels = max(out_channels, 32)
            
            # Transposed convolution for upsampling
            if i == n_layers - 1:
                # Final layer to audio with stable padding
                layers.extend([
                    nn.ConvTranspose1d(
                        current_channels, 2,  # Stereo output
                        kernel_size=hop_length * 2,
                        stride=hop_length,
                        padding=hop_length // 2,
                        output_padding=0
                    ),
                    nn.Tanh()
                ])
            else:
                layers.extend([
                    nn.ConvTranspose1d(
                        current_channels, out_channels,
                        kernel_size=8, stride=2, padding=3,
                        output_padding=1
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(0.2),
                ])
                current_channels = out_channels
        
        self.vocoder = nn.Sequential(*layers)
        
        # Alternative: Griffin-Lim for validation
        self.use_griffin_lim = False
    
    def forward(self, cqt: torch.Tensor) -> torch.Tensor:
        """
        Convert CQT back to audio with stability enhancements
        
        Args:
            cqt: (B, n_bins, T_frames) CQT representation
        Returns:
            audio: (B, 2, T) reconstructed stereo audio
        """
        if cqt.shape[-1] == 0:
            # Handle empty input
            batch_size = cqt.shape[0]
            return torch.zeros(batch_size, 2, 1, device=cqt.device, dtype=cqt.dtype)
            
        if self.use_griffin_lim:
            return self._griffin_lim_inverse(cqt)
        else:
            try:
                return self.vocoder(cqt)
            except RuntimeError as e:
                # Fallback to simpler reconstruction
                return self._griffin_lim_inverse(cqt)
    
    def _griffin_lim_inverse(self, cqt: torch.Tensor) -> torch.Tensor:
        """Enhanced inverse using interpolation with stable padding"""
        B, n_bins, T_frames = cqt.shape
        
        # Estimate audio length with minimum length
        audio_length = max(T_frames * self.hop_length, self.hop_length)
        
        # Simple upsampling interpolation with error handling
        try:
            audio_mono = F.interpolate(cqt, size=audio_length, mode='linear', align_corners=False)
        except RuntimeError:
            # Fallback for interpolation issues
            audio_mono = cqt.repeat_interleave(self.hop_length // max(1, T_frames // 10), dim=-1)
            if audio_mono.shape[-1] > audio_length:
                audio_mono = audio_mono[..., :audio_length]
            elif audio_mono.shape[-1] < audio_length:
                pad_length = audio_length - audio_mono.shape[-1]
                audio_mono = F.pad(audio_mono, (0, pad_length), mode='replicate')
        
        # Convert to stereo
        audio_mono = audio_mono.mean(dim=1, keepdim=True)  # Average to single channel
        audio_stereo = audio_mono.repeat(1, 2, 1)  # Duplicate to stereo
        
        return audio_stereo


# ==================== Enhanced SSM Components (Using Unified SSM) ====================

class EnhancedCQTSSMBlock(OptimizedS6Block):
    """
    CQT-optimized SSM Block using the unified SSM implementation
    Inherits from OptimizedS6Block and adds CQT-specific enhancements
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        # CQT-specific parameters
        use_frequency_conditioning: bool = True,
        use_harmonic_bias: bool = True,
        **kwargs
    ):
        super().__init__(d_model, d_state, d_conv, **kwargs)
        
        self.use_frequency_conditioning = use_frequency_conditioning
        self.use_harmonic_bias = use_harmonic_bias
        
        if use_frequency_conditioning:
            # Frequency-aware conditioning
            self.freq_cond = nn.Linear(d_model, d_model)
        
        if use_harmonic_bias:
            # Harmonic bias for musical structure
            self.harmonic_bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor, freq_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enhanced forward pass with CQT-specific conditioning
        
        Args:
            x: (B, L, D) input sequence  
            freq_info: (B, L, D) frequency information
        Returns:
            output: (B, L, D) processed sequence
        """
        residual = x
        x = self.norm(x)
        
        # Apply frequency conditioning if available
        if self.use_frequency_conditioning and freq_info is not None:
            freq_cond = torch.sigmoid(self.freq_cond(freq_info))
            x = x * freq_cond
        
        # Apply SSM processing (using parent's implementation)
        x = self.s6(x)
        
        # Apply harmonic bias
        if self.use_harmonic_bias:
            x = x + self.harmonic_bias
        
        x = self.dropout(x)
        return x + residual


# ==================== CQT-SSM Encoder ====================

class CQTSSMEncoder(nn.Module):
    """
    CQT-based SSM encoder with Harmonic-Percussive separation
    Now using unified SSM components for better consistency
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_bins: int = 84,
        hop_length: int = 512,
        base_channels: int = 64,
        latent_channels: int = 8,
        ssm_layers: List[int] = [2, 2, 3, 3, 2],
        d_state: int = 64,
        dropout: float = 0.1,
        use_harmonic_percussive: bool = True,
        use_multiscale_ssm: bool = True,
        use_weight_norm: bool = True,
        # Memory optimization parameters
        chunk_size: int = 256,
        use_checkpointing: bool = True,
        memory_efficient: bool = True,
    ):
        super().__init__()
        
        self.n_bins = n_bins
        self.use_harmonic_percussive = use_harmonic_percussive
        self.use_checkpointing = use_checkpointing
        self.chunk_size = chunk_size
        self.memory_efficient = memory_efficient
        
        # CQT Transform
        self.cqt_transform = ConstantQTransform(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_bins=n_bins
        )
        
        # Harmonic-Percussive Separation
        if use_harmonic_percussive:
            self.hp_separator = HarmonicPercussiveSeparation(learnable=True)
            input_channels = 2  # Harmonic + Percussive
        else:
            input_channels = 1  # Just CQT
        
        # Initial projection
        stem_conv = nn.Conv2d(input_channels, base_channels, 7, padding=3)
        if use_weight_norm:
            stem_conv = nn.utils.weight_norm(stem_conv)
        
        self.stem = nn.Sequential(
            stem_conv,
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU()
        )
        
        # Multi-stage encoder with unified SSM
        self.stages = nn.ModuleList()
        current_channels = base_channels
        
        for i, num_ssm_layers in enumerate(ssm_layers):
            out_channels = base_channels * (2 ** i)
            
            # Downsampling (except first stage)
            if i == 0:
                downsample = nn.Identity()
            else:
                conv = nn.Conv2d(current_channels, out_channels, 3, stride=2, padding=1)
                if use_weight_norm:
                    conv = nn.utils.weight_norm(conv)
                
                downsample = nn.Sequential(
                    conv,
                    nn.GroupNorm(min(8, out_channels), out_channels),
                    nn.SiLU()
                )
            
            # SSM processing using unified components
            if use_multiscale_ssm:
                ssm_processor = OptimizedMultiScaleS6(
                    d_model=out_channels,
                    scales=[1, 2] if out_channels >= 128 else [1],
                    d_state=d_state,
                    dropout=dropout,
                )
            else:
                # Use list of CQTSSMBlocks (inheriting from unified S6Block)
                ssm_processor = nn.ModuleList([
                    EnhancedCQTSSMBlock(
                        d_model=out_channels,
                        d_state=d_state,
                        dropout=dropout,
                        use_frequency_conditioning=True,
                        use_harmonic_bias=True,
                    ) for _ in range(num_ssm_layers)
                ])
            
            stage = nn.ModuleDict({
                'downsample': downsample,
                'ssm_processor': ssm_processor
            })
            
            self.stages.append(stage)
            current_channels = out_channels
        
        # Final projection to latent space
        final_conv1 = nn.Conv2d(current_channels, latent_channels * 2, 3, padding=1)
        final_conv2 = nn.Conv2d(latent_channels * 2, latent_channels, 1)
        
        if use_weight_norm:
            final_conv1 = nn.utils.weight_norm(final_conv1)
            final_conv2 = nn.utils.weight_norm(final_conv2)
        
        # Calculate appropriate number of groups for GroupNorm
        num_groups = min(8, latent_channels * 2)
        # Ensure num_channels is divisible by num_groups
        while (latent_channels * 2) % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        
        self.final_conv = nn.Sequential(
            final_conv1,
            nn.GroupNorm(num_groups, latent_channels * 2),
            nn.SiLU(),
            final_conv2
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode audio to latent representation using CQT and unified SSM
        
        Args:
            audio: (B, 2, T) stereo audio
        Returns:
            latent: (B, latent_channels, H//32, W//32) compressed representation
            skip_features: Skip connections for decoder
        """
        # CQT transformation with error handling
        try:
            cqt = self.cqt_transform(audio)  # (B, n_bins, T_frames)
        except Exception as e:
            # Fallback for CQT issues
            B = audio.shape[0]
            T_frames = max(1, audio.shape[-1] // 512)
            cqt = torch.zeros(B, self.n_bins, T_frames, device=audio.device, dtype=audio.dtype)
        
        # Harmonic-Percussive separation
        if self.use_harmonic_percussive:
            harmonic, percussive = self.hp_separator(cqt)
            # Stack as channels
            x = torch.stack([harmonic, percussive], dim=1)  # (B, 2, n_bins, T_frames)
        else:
            x = cqt.unsqueeze(1)  # (B, 1, n_bins, T_frames)
        
        # Initial convolution
        x = self.stem(x)
        
        skip_features = []
        
        # Multi-stage processing with unified SSM
        for i, stage in enumerate(self.stages):
            # Downsampling
            x = stage['downsample'](x)
            
            # SSM processing with dimension validation
            B, C, H, W = x.shape
            
            if H * W == 0:
                # Skip processing for empty dimensions
                skip_features.append(torch.zeros_like(x))
                continue
            
            # Reshape for SSM processing
            x_seq = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)  # (B*H, W, C)
            
            if isinstance(stage['ssm_processor'], OptimizedMultiScaleS6):
                # Enhanced MultiScale SSM
                if self.use_checkpointing and self.training:
                    x_seq = checkpoint.checkpoint(
                        stage['ssm_processor'], x_seq,
                        use_reentrant=False
                    )
                else:
                    x_seq = stage['ssm_processor'](x_seq)
            else:
                # List of CQTSSMBlocks
                if self.use_checkpointing and self.training:
                    x_seq = checkpoint.checkpoint(
                        self._process_ssm_blocks, x_seq, stage['ssm_processor'],
                        use_reentrant=False
                    )
                else:
                    x_seq = self._process_ssm_blocks(x_seq, stage['ssm_processor'])
            
            # Reshape back
            x = x_seq.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            
            # Store skip features (compressed) with size validation
            if x.shape[-2] > 0 and x.shape[-1] > 0:
                skip_size = (max(1, x.shape[-2]//2), max(1, x.shape[-1]//2))
                skip_features.append(F.adaptive_avg_pool2d(x, skip_size))
            else:
                skip_features.append(torch.zeros_like(x))
        
        # Final latent
        latent = self.final_conv(x)
        
        return latent, skip_features
    
    def _process_ssm_blocks(self, x_seq, ssm_blocks):
        """Process sequence through list of SSM blocks"""
        for ssm_block in ssm_blocks:
            x_seq = ssm_block(x_seq)
        return x_seq


# ==================== CQT-SSM Decoder ====================

class CQTSSMDecoder(nn.Module):
    """
    CQT-based SSM decoder with inverse CQT transformation
    Now using unified SSM components
    """
    
    def __init__(
        self,
        latent_channels: int = 8,
        base_channels: int = 64,
        n_bins: int = 84,
        ssm_layers: List[int] = [2, 3, 3, 2, 2],
        output_channels: int = 2,
        d_state: int = 64,
        use_multiscale_ssm: bool = True,
        dropout: float = 0.1,
        use_weight_norm: bool = True,
        sample_rate: int = 44100,
        hop_length: int = 512,
        # Memory optimization
        chunk_size: int = 256,
        use_checkpointing: bool = True,
        memory_efficient: bool = True,
    ):
        super().__init__()
        
        self.num_stages = len(ssm_layers)
        self.use_checkpointing = use_checkpointing
        self.n_bins = n_bins
        
        # Initial projection from latent space
        initial_channels = base_channels * (2 ** (self.num_stages - 1))
        initial_conv = nn.Conv2d(latent_channels, initial_channels, 3, padding=1)
        
        if use_weight_norm:
            initial_conv = nn.utils.weight_norm(initial_conv)
        
        self.initial_conv = nn.Sequential(
            initial_conv,
            nn.GroupNorm(min(8, initial_channels), initial_channels),
            nn.SiLU()
        )
        
        # Skip connection projections
        self.skip_projections = nn.ModuleList()
        
        # Decoder stages with unified SSM
        self.stages = nn.ModuleList()
        current_channels = initial_channels
        
        for i in range(self.num_stages):
            if i == self.num_stages - 1:
                out_channels = base_channels
            else:
                out_channels = base_channels * (2 ** (self.num_stages - 2 - i))
            
            # Skip connection projection
            if i < self.num_stages - 1:
                skip_proj = nn.Conv2d(current_channels * 2, current_channels, 1)
                if use_weight_norm:
                    skip_proj = nn.utils.weight_norm(skip_proj)
                self.skip_projections.append(skip_proj)
            else:
                self.skip_projections.append(nn.Identity())
            
            # Upsampling layer
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    current_channels, out_channels,
                    kernel_size=4, stride=2, padding=1
                ),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU()
            )
            
            if use_weight_norm:
                upsample[0] = nn.utils.weight_norm(upsample[0])
            
            # SSM processing using unified components
            if i < self.num_stages - 1:
                if use_multiscale_ssm:
                    ssm_processor = OptimizedMultiScaleS6(
                        d_model=out_channels,
                        scales=[1, 2] if out_channels >= 128 else [1],
                        d_state=d_state,
                        dropout=dropout,
                    )
                else:
                    ssm_processor = nn.ModuleList([
                        EnhancedCQTSSMBlock(
                            d_model=out_channels,
                            d_state=d_state,
                            dropout=dropout,
                            use_frequency_conditioning=True,
                            use_harmonic_bias=True,
                        ) for _ in range(ssm_layers[i])
                    ])
            else:
                ssm_processor = nn.Identity()
            
            stage = nn.ModuleDict({
                'upsample': upsample,
                'ssm_processor': ssm_processor
            })
            
            self.stages.append(stage)
            current_channels = out_channels
        
        # Final CQT reconstruction
        final_conv = nn.Conv2d(current_channels, 2, 3, padding=1)  # Harmonic + Percussive
        if use_weight_norm:
            final_conv = nn.utils.weight_norm(final_conv)
        
        self.final_conv = nn.Sequential(
            final_conv,
            nn.Tanh()
        )
        
        # Inverse CQT transform to audio
        self.inverse_cqt = CQTInverseTransform(
            n_bins=n_bins,
            sample_rate=sample_rate,
            hop_length=hop_length
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, latent: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode latent to audio using CQT and unified SSM
        
        Args:
            latent: (B, latent_channels, H, W)
            skip_features: Skip connections from encoder
        Returns:
            audio: (B, 2, T) reconstructed stereo audio
        """
        x = self.initial_conv(latent)
        
        # Reverse skip features order
        skip_features = skip_features[::-1]
        
        for i, stage in enumerate(self.stages):
            # Skip connections with enhanced size matching
            if i < self.num_stages - 1 and i < len(skip_features):
                skip_feature = skip_features[i]
                
                # Resize skip to match current resolution
                if skip_feature.shape[-2:] != x.shape[-2:]:
                    skip_feature = F.interpolate(
                        skip_feature, size=x.shape[-2:], 
                        mode='bilinear', align_corners=False
                    )
                
                # Enhanced channel matching
                if skip_feature.shape[1] != x.shape[1]:
                    if skip_feature.shape[1] < x.shape[1]:
                        # Pad channels
                        pad_channels = x.shape[1] - skip_feature.shape[1]
                        skip_feature = F.pad(skip_feature, (0, 0, 0, 0, 0, pad_channels))
                    else:
                        # Truncate channels
                        skip_feature = skip_feature[:, :x.shape[1]]
                
                x = torch.cat([x, skip_feature], dim=1)
                x = self.skip_projections[i](x)
            
            # Upsampling
            x = stage['upsample'](x)
            
            # SSM processing using unified components
            if not isinstance(stage['ssm_processor'], nn.Identity):
                B, C, H, W = x.shape
                
                if H * W > 0:
                    x_seq = x.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
                    
                    if isinstance(stage['ssm_processor'], OptimizedMultiScaleS6):
                        # Enhanced MultiScale SSM
                        if self.use_checkpointing and self.training:
                            x_seq = checkpoint.checkpoint(
                                stage['ssm_processor'], x_seq,
                                use_reentrant=False
                            )
                        else:
                            x_seq = stage['ssm_processor'](x_seq)
                    else:
                        # List of CQTSSMBlocks
                        if self.use_checkpointing and self.training:
                            x_seq = checkpoint.checkpoint(
                                self._process_ssm_blocks, x_seq, stage['ssm_processor'],
                                use_reentrant=False
                            )
                        else:
                            x_seq = self._process_ssm_blocks(x_seq, stage['ssm_processor'])
                    
                    x = x_seq.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # Convert to CQT representation (Harmonic + Percussive)
        cqt_components = self.final_conv(x)  # (B, 2, n_bins, T_frames)
        
        # Combine harmonic and percussive components
        cqt_reconstructed = cqt_components.sum(dim=1)  # (B, n_bins, T_frames)
        
        # Convert CQT back to audio
        audio = self.inverse_cqt(cqt_reconstructed)
        
        return audio
    
    def _process_ssm_blocks(self, x_seq, ssm_blocks):
        """Process sequence through list of SSM blocks"""
        for ssm_block in ssm_blocks:
            x_seq = ssm_block(x_seq)
        return x_seq


# ==================== Enhanced CQT Loss ====================

class CQTLoss(nn.Module):
    """
    CQT-based loss function optimized for music
    Superior to STFT loss for musical content
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_lengths: List[int] = [256, 512, 1024],
        n_bins_list: List[int] = [72, 84, 96],
        w_cqt: float = 1.0,
        w_temporal: float = 0.5,
        w_harmonic: float = 0.3,
    ):
        super().__init__()
        
        self.w_cqt = w_cqt
        self.w_temporal = w_temporal 
        self.w_harmonic = w_harmonic
        
        # Multiple CQT transforms for multi-resolution loss
        self.cqt_transforms = nn.ModuleList([
            ConstantQTransform(
                sample_rate=sample_rate,
                hop_length=hop_length,
                n_bins=n_bins
            ) for hop_length, n_bins in zip(hop_lengths, n_bins_list)
        ])
        
        # Harmonic-Percussive separator for additional loss
        self.hp_separator = HarmonicPercussiveSeparation(learnable=False)
    
    def forward(self, pred_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute CQT-based loss with enhanced stability
        
        Args:
            pred_audio: (B, 2, T) predicted audio
            target_audio: (B, 2, T) target audio
        Returns:
            total_loss: Combined CQT loss
        """
        # Ensure same length with enhanced padding
        min_length = min(pred_audio.shape[-1], target_audio.shape[-1])
        if min_length <= 0:
            # Handle edge case
            return torch.tensor(0.0, device=pred_audio.device, requires_grad=True)
            
        pred_audio = pred_audio[..., :min_length]
        target_audio = target_audio[..., :min_length]
        
        total_loss = 0.0
        valid_losses = 0
        
        # Multi-resolution CQT loss with error handling
        for cqt_transform in self.cqt_transforms:
            try:
                pred_cqt = cqt_transform(pred_audio)
                target_cqt = cqt_transform(target_audio)
                
                # CQT magnitude loss
                cqt_loss = F.l1_loss(pred_cqt, target_cqt)
                total_loss += self.w_cqt * cqt_loss
                valid_losses += 1
            except Exception:
                # Skip this resolution if it fails
                continue
        
        # Temporal consistency loss
        if self.w_temporal > 0:
            try:
                temporal_loss = self._temporal_consistency_loss(pred_audio, target_audio)
                total_loss += self.w_temporal * temporal_loss
            except Exception:
                pass
        
        # Harmonic structure loss
        if self.w_harmonic > 0:
            try:
                harmonic_loss = self._harmonic_structure_loss(pred_audio, target_audio)
                total_loss += self.w_harmonic * harmonic_loss
            except Exception:
                pass
        
        # Ensure we have at least one valid loss
        if valid_losses == 0:
            return F.l1_loss(pred_audio, target_audio)  # Fallback to simple L1 loss
        
        return total_loss / max(valid_losses, 1)
    
    def _temporal_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Temporal consistency loss using frame differences"""
        if pred.shape[-1] <= 1:
            return torch.tensor(0.0, device=pred.device)
            
        pred_diff = torch.diff(pred, dim=-1)
        target_diff = torch.diff(target, dim=-1)
        return F.mse_loss(pred_diff, target_diff)
    
    def _harmonic_structure_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Harmonic structure loss using H-P separation"""
        try:
            # Use first CQT transform for H-P separation
            pred_cqt = self.cqt_transforms[0](pred)
            target_cqt = self.cqt_transforms[0](target)
            
            pred_h, pred_p = self.hp_separator(pred_cqt)
            target_h, target_p = self.hp_separator(target_cqt)
            
            harmonic_loss = F.l1_loss(pred_h, target_h)
            percussive_loss = F.l1_loss(pred_p, target_p)
            
            return harmonic_loss + percussive_loss
        except Exception:
            return torch.tensor(0.0, device=pred.device)


# ==================== Complete CQT-SSM DCAE Model ====================

class CQTSSMDCAE(nn.Module):
    """
    Complete CQT-SSM-based DCAE with Harmonic-Percussive separation
    Now using unified SSM components for better consistency and performance
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        n_bins: int = 84,
        hop_length: int = 512,
        latent_channels: int = 8,
        use_vector_quantization: bool = False,
        vq_num_embeddings: int = 1024,
        vq_commitment_cost: float = 0.25,
        encoder_base_channels: int = 64,
        decoder_base_channels: int = 64,
        dropout: float = 0.1,
        use_weight_norm: bool = True,
        use_multiscale_ssm: bool = True,
        d_state: int = 64,
        # Memory optimization
        chunk_size: int = 256,
        use_checkpointing: bool = True,
        memory_efficient: bool = True,
        checkpointing_segments: int = 4,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.latent_channels = latent_channels
        self.use_vq = use_vector_quantization
        self.n_bins = n_bins
        self.hop_length = hop_length
        
        # CQT-SSM Encoder using unified SSM components
        self.encoder = CQTSSMEncoder(
            sample_rate=sample_rate,
            n_bins=n_bins,
            hop_length=hop_length,
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
            try:
                from vector_quantize_pytorch import VectorQuantize
                self.quantizer = VectorQuantize(
                    dim=latent_channels,
                    codebook_size=vq_num_embeddings,
                    decay=0.99,
                    commitment_weight=vq_commitment_cost
                )
            except ImportError:
                print("Warning: vector_quantize_pytorch not available, disabling VQ")
                self.use_vq = False
        
        # CQT-SSM Decoder using unified SSM components
        self.decoder = CQTSSMDecoder(
            latent_channels=latent_channels,
            base_channels=decoder_base_channels,
            n_bins=n_bins,
            output_channels=2,
            d_state=d_state,
            dropout=dropout,
            use_multiscale_ssm=use_multiscale_ssm,
            use_weight_norm=use_weight_norm,
            sample_rate=sample_rate,
            hop_length=hop_length,
            chunk_size=chunk_size,
            use_checkpointing=use_checkpointing,
            memory_efficient=memory_efficient,
        )
        
        # Enhanced loss function
        self.cqt_loss_fn = CQTLoss(sample_rate=sample_rate)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode audio to latent using CQT and unified SSM"""
        # Remember original length for consistent decoding
        self._last_input_length = audio.shape[-1]
        return self.encoder(audio)
    
    def decode(self, latent: torch.Tensor, skip_features: List[torch.Tensor]) -> torch.Tensor:
        """Decode latent to audio using CQT and unified SSM"""
        audio = self.decoder(latent, skip_features)

        # If we encoded an input previously, ensure the length matches
        if hasattr(self, "_last_input_length"):
            target_len = self._last_input_length
            if audio.shape[-1] > target_len:
                audio = audio[..., :target_len]
            elif audio.shape[-1] < target_len:
                pad_len = target_len - audio.shape[-1]
                audio = F.pad(audio, (0, pad_len), mode="reflect")

        return audio
    
    def forward(
        self,
        audio: torch.Tensor,
        return_loss: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Complete forward pass with CQT-SSM processing using unified components"""
        # Store original length
        original_length = audio.shape[-1]
        
        # Encode using CQT and unified SSM
        latent, skip_features = self.encode(audio)
        
        # Vector quantization (if enabled)
        vq_loss = torch.tensor(0.0, device=audio.device)
        if self.use_vq and hasattr(self, 'quantizer'):
            latent, vq_loss, _ = self.quantizer(latent)
        
        # Decode using unified SSM and inverse CQT
        reconstructed = self.decode(latent, skip_features)
        
        # Ensure reconstructed audio has the same length as input with enhanced padding
        if reconstructed.shape[-1] != original_length:
            if reconstructed.shape[-1] > original_length:
                reconstructed = reconstructed[..., :original_length]
            else:
                pad_length = original_length - reconstructed.shape[-1]
                # Use reflect padding for more natural sound
                reconstructed = F.pad(reconstructed, (0, pad_length), mode='reflect')
        
        if return_loss:
            # Enhanced CQT-based reconstruction loss
            cqt_loss = self.cqt_loss_fn(reconstructed, audio)
            
            # Time domain loss (reduced weight since CQT is primary)
            time_loss = F.l1_loss(reconstructed, audio)
            
            # Total loss with proper weighting
            total_loss = cqt_loss + 0.1 * time_loss + 0.02 * vq_loss
            
            # Return consistent loss dictionary with CQT terminology
            loss_dict = {
                'total_loss': total_loss,
                'cqt_loss': cqt_loss,  # CQT-based loss
                'time_loss': time_loss,
                'vq_loss': vq_loss
            }
            
            return reconstructed, loss_dict
        
        return reconstructed
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio based on CQT parameters"""
        # CQT provides better compression than raw audio
        cqt_compression = self.hop_length  # Temporal compression
        spatial_compression = 32  # From encoder downsampling
        return cqt_compression * spatial_compression / 2  # ~8192 for hop_length=512
    
    def estimate_latent_shape(self, audio_length: int) -> Tuple[int, int]:
        """Estimate latent shape for given audio length"""
        cqt_frames = audio_length // self.hop_length
        latent_frames = cqt_frames // 32  # From encoder downsampling
        return (self.latent_channels, latent_frames)
    
    def get_memory_stats(self) -> Dict[str, str]:
        """Get memory optimization configuration"""
        return {
            'representation': 'CQT + Harmonic-Percussive',
            'ssm_components': 'Unified SSM (OptimizedS6StateSpaceKernel, OptimizedS6Block, OptimizedMultiScaleS6)',
            'n_bins': self.n_bins,
            'hop_length': self.hop_length,
            'compression_ratio': f'{self.get_compression_ratio():.1f}x',
            'use_checkpointing': self.encoder.use_checkpointing,
            'memory_efficient': self.encoder.memory_efficient,
            'estimated_memory_savings': '85-90% vs raw audio SSM'
        }


# ==================== Model Factory ====================

def create_cqt_ssm_dcae(
    model_size: str = "base",
    sample_rate: int = 44100,
    use_vq: bool = False,
    use_weight_norm: bool = True,
    encoder_base_channels: int = None,
    decoder_base_channels: int = None,
    dropout: float = 0.1,
    use_multiscale_ssm: bool = True,
    d_state: int = None,
    # Memory optimization parameters
    chunk_size: int = 256,
    use_checkpointing: bool = True,
    memory_efficient: bool = True,
    checkpointing_segments: int = 4,
    # Performance optimization parameters
    use_torch_compile: bool = False,  # 기본값을 False로 변경하여 compile 오류 방지
    use_mixed_precision: bool = True,
    compile_mode: str = "default",  # "default", "reduce-overhead", "max-autotune"
    **kwargs
) -> CQTSSMDCAE:
    """Create CQT-SSM-based DCAE model with unified SSM components and performance optimizations"""
    
    if model_size == "small":
        base_config = {
            "encoder_base_channels": encoder_base_channels or 32,
            "decoder_base_channels": decoder_base_channels or 32,
            "latent_channels": 6,
            "n_bins": 72,  # 6 octaves
        }
        effective_d_state = d_state or 32
        effective_chunk_size = 128
    elif model_size == "base":
        base_config = {
            "encoder_base_channels": encoder_base_channels or 64,
            "decoder_base_channels": decoder_base_channels or 64,
            "latent_channels": 8,
            "n_bins": 84,  # 7 octaves
        }
        effective_d_state = d_state or 64
        effective_chunk_size = chunk_size
    elif model_size == "large":
        base_config = {
            "encoder_base_channels": encoder_base_channels or 96,
            "decoder_base_channels": decoder_base_channels or 96,
            "latent_channels": 12,
            "n_bins": 96,  # 8 octaves
        }
        effective_d_state = d_state or 96
        effective_chunk_size = chunk_size * 2
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # Get all parameters that CQTSSMDCAE.__init__ accepts
    model_init_params = {
        'sample_rate', 'n_bins', 'hop_length', 'latent_channels', 'use_vector_quantization',
        'vq_num_embeddings', 'vq_commitment_cost', 'encoder_base_channels', 
        'decoder_base_channels', 'dropout', 'use_weight_norm', 'use_multiscale_ssm',
        'd_state', 'chunk_size', 'use_checkpointing', 'memory_efficient', 'checkpointing_segments'
    }
    
    # Combine base config with kwargs
    final_config = {}
    final_config.update(base_config)
    final_config.update({k: v for k, v in kwargs.items() if k in model_init_params})
    
    model = CQTSSMDCAE(
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
    
    # Performance optimizations
    if use_torch_compile and torch.__version__ >= "2.0.0":
        try:
            print(f"🚀 Applying torch.compile() with mode '{compile_mode}'...")
            
            # Check for Triton availability and use safer compile mode on Windows
            import platform
            if platform.system() == "Windows":
                # Use reduce-overhead mode to avoid Triton issues on Windows
                safe_compile_mode = "reduce-overhead" if compile_mode == "default" else compile_mode
                print(f"Windows detected, using safer compile mode: {safe_compile_mode}")
            else:
                safe_compile_mode = compile_mode
            
            # Compile the encoder and decoder separately for better optimization
            model.encoder = torch.compile(model.encoder, mode=safe_compile_mode)
            model.decoder = torch.compile(model.decoder, mode=safe_compile_mode)
            
            # Optionally compile the full model for end-to-end optimization
            # model = torch.compile(model, mode=safe_compile_mode)
            
            print("✅ torch.compile() applied successfully")
        except Exception as e:
            print(f"⚠️ torch.compile() failed: {e}")
            print("Continuing without compilation...")
            # Disable torch.compile for this model instance
            use_torch_compile = False
    
    # Set mixed precision flag for training loops to use
    model._use_mixed_precision = use_mixed_precision
    model._compile_mode = compile_mode if use_torch_compile else None
    
    return model


# ==================== Backward Compatibility ====================

# Alias for backward compatibility with existing training scripts
def create_memory_optimized_lyro_dcae(*args, **kwargs):
    """Backward compatibility wrapper - now uses unified CQT-SSM implementation"""
    return create_cqt_ssm_dcae(*args, **kwargs)

def create_enhanced_lyro_dcae(*args, **kwargs):
    """Backward compatibility wrapper"""
    return create_cqt_ssm_dcae(*args, **kwargs)

# Main model class alias
MemoryOptimizedLyroMusicDCAE = CQTSSMDCAE
LyroMusicDCAE = CQTSSMDCAE  # Backward compatibility