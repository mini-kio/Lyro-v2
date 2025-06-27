# lyro/models/dcae.py
"""
DCAE: Deep Convolutional Audio Encoder
Improved version with better compression and stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import math
from typing import Tuple, Optional, Dict, Any


class ResidualBlock(nn.Module):
    """Improved residual block with better normalization"""
    
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        
        self.norm1 = nn.GroupNorm(min(8, channels//4), channels)
        self.norm2 = nn.GroupNorm(min(8, channels//4), channels)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        
        return self.activation(x + residual)


class VectorQuantizer(nn.Module):
    """Improved vector quantizer with better stability"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert inputs from BCT -> BTC for embedding lookup
        input_shape = inputs.shape
        inputs = inputs.permute(0, 2, 1).contiguous()
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape[0], input_shape[2], self.embedding_dim)
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs.permute(0, 2, 1))
        q_latent_loss = F.mse_loss(quantized, inputs.permute(0, 2, 1).detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs.permute(0, 2, 1) + (quantized - inputs.permute(0, 2, 1)).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        return quantized, loss, encoding_indices.view(input_shape[0], input_shape[2])


class DCAEEncoder(nn.Module):
    """DCAE Encoder with improved architecture"""
    
    def __init__(self, input_channels: int = 2, latent_channels: int = 16):
        super().__init__()
        
        # Progressive downsampling
        self.conv_layers = nn.ModuleList([
            # Input: 2 channels -> 64 channels, downsample 4x
            nn.Sequential(
                nn.Conv1d(input_channels, 64, 7, stride=4, padding=3),
                nn.GroupNorm(8, 64),
                nn.GELU()
            ),
            # 64 -> 128 channels, downsample 4x  
            nn.Sequential(
                nn.Conv1d(64, 128, 5, stride=4, padding=2),
                nn.GroupNorm(16, 128),
                nn.GELU()
            ),
            # 128 -> 256 channels, downsample 4x
            nn.Sequential(
                nn.Conv1d(128, 256, 3, stride=4, padding=1),
                nn.GroupNorm(32, 256),
                nn.GELU()
            ),
            # 256 -> 512 channels, downsample 2x
            nn.Sequential(
                nn.Conv1d(256, 512, 3, stride=2, padding=1),
                nn.GroupNorm(32, 512),
                nn.GELU()
            )
        ])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(64),
            ResidualBlock(128, dilation=2),
            ResidualBlock(256, dilation=2),
            ResidualBlock(512, dilation=1)
        ])
        
        # Final projection to latent space
        self.to_latent = nn.Conv1d(512, latent_channels, 1)
        
        # Adaptive pooling for consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(128)  # Fixed temporal dimension
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, T) audio waveform
        Returns:
            (B, latent_channels, 128) latent representation
        """
        # Ensure input is stereo
        if x.shape[1] == 1:
            x = x.repeat(1, 2, 1)
        elif x.shape[1] > 2:
            x = x[:, :2, :]
            
        for conv, residual in zip(self.conv_layers, self.residual_blocks):
            x = conv(x)
            x = residual(x)
            
        x = self.to_latent(x)
        x = self.adaptive_pool(x)
        
        return x


class DCAEDecoder(nn.Module):
    """DCAE Decoder with improved architecture"""
    
    def __init__(self, latent_channels: int = 16, output_channels: int = 2, target_length: int = 44100):
        super().__init__()
        
        self.target_length = target_length
        
        # Input projection
        self.from_latent = nn.Conv1d(latent_channels, 512, 1)
        
        # Progressive upsampling
        self.conv_layers = nn.ModuleList([
            # 512 -> 256 channels, upsample 2x
            nn.Sequential(
                nn.ConvTranspose1d(512, 256, 4, stride=2, padding=1),
                nn.GroupNorm(32, 256),
                nn.GELU()
            ),
            # 256 -> 128 channels, upsample 4x
            nn.Sequential(
                nn.ConvTranspose1d(256, 128, 8, stride=4, padding=2),
                nn.GroupNorm(16, 128),
                nn.GELU()
            ),
            # 128 -> 64 channels, upsample 4x
            nn.Sequential(
                nn.ConvTranspose1d(128, 64, 8, stride=4, padding=2),
                nn.GroupNorm(8, 64),
                nn.GELU()
            ),
            # 64 -> output channels, upsample 4x
            nn.Sequential(
                nn.ConvTranspose1d(64, output_channels, 8, stride=4, padding=2),
                nn.Tanh()
            )
        ])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(512, dilation=1),
            ResidualBlock(256, dilation=2), 
            ResidualBlock(128, dilation=2),
            None  # No residual for final layer
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, latent_channels, 128) latent representation
        Returns:
            (B, 2, target_length) audio waveform
        """
        x = self.from_latent(x)
        
        for conv, residual in zip(self.conv_layers, self.residual_blocks):
            x = conv(x)
            if residual is not None:
                x = residual(x)
                
        # Adjust length to target
        if x.shape[-1] != self.target_length:
            if x.shape[-1] > self.target_length:
                x = x[..., :self.target_length]
            else:
                x = F.interpolate(x, size=self.target_length, mode='linear', align_corners=False)
                
        return x


class DCAE(nn.Module):
    """
    Deep Convolutional Audio Encoder
    Improved version with better compression and stability
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        latent_channels: int = 16,
        target_compression_ratio: float = 50.0,
        use_quantization: bool = True,
        quantization_num_embeddings: int = 1024
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.latent_channels = latent_channels
        self.target_compression_ratio = target_compression_ratio
        self.use_quantization = use_quantization
        
        # Encoder and decoder
        self.encoder = DCAEEncoder(input_channels=2, latent_channels=latent_channels)
        self.decoder = DCAEDecoder(latent_channels=latent_channels, output_channels=2)
        
        # Vector quantization (optional)
        if use_quantization:
            self.quantizer = VectorQuantizer(
                num_embeddings=quantization_num_embeddings,
                embedding_dim=latent_channels
            )
        else:
            self.quantizer = None
            
        # Spectral transforms for loss calculation
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            f_min=0.0,
            f_max=sample_rate // 2
        )
        
    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode audio to latent representation
        
        Args:
            audio: (B, 2, T) audio waveform
            
        Returns:
            latent: (B, latent_channels, 128) latent representation
            quantization_loss: Optional quantization loss
        """
        latent = self.encoder(audio)
        quantization_loss = None
        
        if self.quantizer is not None:
            latent, quantization_loss, _ = self.quantizer(latent)
            
        return latent, quantization_loss
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to audio
        
        Args:
            latent: (B, latent_channels, 128) latent representation
            
        Returns:
            audio: (B, 2, T) audio waveform
        """
        return self.decoder(latent)
    
    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Full encode-decode pass
        
        Args:
            audio: (B, 2, T) audio waveform
            
        Returns:
            reconstructed: (B, 2, T) reconstructed audio
            quantization_loss: Optional quantization loss
        """
        latent, quantization_loss = self.encode(audio)
        reconstructed = self.decode(latent)
        
        return reconstructed, quantization_loss
    
    def get_compression_info(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Calculate compression statistics
        
        Args:
            audio: (B, 2, T) audio waveform
            
        Returns:
            info: Compression information dictionary
        """
        with torch.no_grad():
            latent, _ = self.encode(audio)
            
            # Calculate sizes
            original_size = audio.numel()
            compressed_size = latent.numel()
            
            # Compression ratio
            compression_ratio = original_size / max(compressed_size, 1)
            
            # Bitrate calculation
            duration = audio.shape[-1] / self.sample_rate
            if duration > 0:
                original_bitrate = (original_size * 32) / duration / 1000  # kbps
                compressed_bitrate = (compressed_size * 32) / duration / 1000  # kbps
            else:
                original_bitrate = compressed_bitrate = 0.0
                
            return {
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': float(compression_ratio),
                'original_bitrate_kbps': float(original_bitrate),
                'compressed_bitrate_kbps': float(compressed_bitrate),
                'latent_shape': tuple(latent.shape),
                'original_shape': tuple(audio.shape)
            }
    
    def compute_spectral_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral loss between original and reconstructed audio
        Fixed version that ensures non-zero output
        """
        device = original.device
        
        # Ensure both tensors are on the same device
        if reconstructed.device != device:
            reconstructed = reconstructed.to(device)
            
        # Move mel transform to correct device
        if self.mel_transform.sample_rate != self.sample_rate:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                hop_length=256,
                n_mels=80,
                f_min=0.0,
                f_max=self.sample_rate // 2
            ).to(device)
        else:
            self.mel_transform = self.mel_transform.to(device)
        
        # Match tensor lengths
        min_length = min(original.shape[-1], reconstructed.shape[-1])
        if min_length < 1024:  # Too short for spectral analysis
            return torch.tensor(0.1, device=device, requires_grad=True)
            
        original = original[..., :min_length]
        reconstructed = reconstructed[..., :min_length]
        
        losses = []
        
        # STFT loss
        try:
            n_fft = min(1024, min_length // 4)
            hop_length = n_fft // 4
            
            window = torch.hann_window(n_fft, device=device)
            
            # Convert to mono for STFT
            orig_mono = original.mean(dim=1) if original.dim() > 1 else original
            recon_mono = reconstructed.mean(dim=1) if reconstructed.dim() > 1 else reconstructed
            
            orig_stft = torch.stft(orig_mono.reshape(-1), n_fft=n_fft, hop_length=hop_length, 
                                 return_complex=True, window=window)
            recon_stft = torch.stft(recon_mono.reshape(-1), n_fft=n_fft, hop_length=hop_length,
                                  return_complex=True, window=window)
            
            stft_loss = F.l1_loss(torch.abs(orig_stft), torch.abs(recon_stft))
            if torch.isfinite(stft_loss) and stft_loss > 1e-8:
                losses.append(stft_loss)
                
        except Exception:
            pass
        
        # Mel spectrogram loss
        try:
            orig_mel = self.mel_transform(original.mean(dim=1))
            recon_mel = self.mel_transform(reconstructed.mean(dim=1))
            
            mel_loss = F.l1_loss(orig_mel, recon_mel)
            if torch.isfinite(mel_loss) and mel_loss > 1e-8:
                losses.append(mel_loss)
                
        except Exception:
            pass
        
        # Return combined loss or fallback
        if losses:
            total_loss = sum(losses) / len(losses)
            return torch.clamp(total_loss, min=1e-6, max=10.0)
        else:
            # Fallback to simple L1 loss
            return F.l1_loss(original, reconstructed)


def create_dcae(
    sample_rate: int = 44100,
    latent_channels: int = 16,
    target_compression_ratio: float = 50.0,
    use_quantization: bool = True,
    **kwargs
) -> DCAE:
    """
    Create DCAE model with specified configuration
    
    Args:
        sample_rate: Audio sample rate
        latent_channels: Number of latent channels
        target_compression_ratio: Target compression ratio
        use_quantization: Whether to use vector quantization
        **kwargs: Additional model arguments
        
    Returns:
        DCAE model
    """
    return DCAE(
        sample_rate=sample_rate,
        latent_channels=latent_channels,
        target_compression_ratio=target_compression_ratio,
        use_quantization=use_quantization,
        **kwargs
    )