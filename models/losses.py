# lyro/models/losses.py
"""
Loss Functions for LYRO
Includes Flow Matching, REPA (HuBERT), Reconstruction, and Perceptual losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, Optional, Tuple, List
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import warnings

warnings.filterwarnings("ignore")


class FlowMatchingLoss(nn.Module):
    """Flow Matching Loss for continuous generation"""
    
    def __init__(
        self,
        beta_schedule: str = "cosine",
        num_timesteps: int = 1000,
        sigma: float = 1e-4
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.sigma = sigma
        
        # Create beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif beta_schedule == "cosine":
            steps = torch.arange(num_timesteps + 1) / num_timesteps
            alphas_cumprod = torch.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = torch.clamp(betas, 0.0001, 0.02)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
            
        self.register_buffer('betas', betas)
        
    def forward(
        self,
        predicted_v: torch.Tensor,
        target_v: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute flow matching loss
        
        Args:
            predicted_v: (B, C, L) predicted velocity
            target_v: (B, C, L) target velocity  
            t: (B,) time steps
            mask: (B, C, L) optional mask
            
        Returns:
            Flow matching loss
        """
        # Basic MSE loss
        loss = F.mse_loss(predicted_v, target_v, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
            
        # Time-dependent weighting
        if t is not None:
            # Weight loss based on time step
            weight = 1.0 + 0.5 * torch.sin(np.pi * t.mean())
            loss = loss * weight
            
        return loss


class REPALoss(nn.Module):
    """
    REPA Loss using frozen HuBERT features
    Based on ZhenYe234/hubert_base_general_audio
    """
    
    def __init__(
        self,
        hubert_model: str = "ZhenYe234/hubert_base_general_audio",
        layer_weights: Optional[List[float]] = None,
        sample_rate: int = 44100,
        target_sample_rate: int = 16000
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        
        try:
            # Load HuBERT model and feature extractor
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model)
            self.hubert = HubertModel.from_pretrained(hubert_model)
            
            # Freeze HuBERT parameters
            for param in self.hubert.parameters():
                param.requires_grad = False
                
            self.hubert.eval()
            
            # Layer weights for multi-layer features
            if layer_weights is None:
                # Default: use middle layers more
                num_layers = len(self.hubert.encoder.layers)
                layer_weights = [0.1] * 4 + [1.0] * (num_layers - 8) + [0.1] * 4
                layer_weights = layer_weights[:num_layers]
                
            self.layer_weights = nn.Parameter(
                torch.tensor(layer_weights, dtype=torch.float32),
                requires_grad=False
            )
            
            print(f"Loaded HuBERT model with {num_layers} layers")
            
        except Exception as e:
            print(f"Warning: Failed to load HuBERT model: {e}")
            print("Using fallback REPA loss")
            
            # Fallback: simple convolutional feature extractor
            self.hubert = None
            self.feature_extractor = None
            
            self.fallback_features = nn.Sequential(
                nn.Conv1d(1, 64, 7, stride=2, padding=3),
                nn.ReLU(),
                nn.Conv1d(64, 128, 5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(128, 256, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(256, 512, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(100)
            )
            
            # Freeze fallback features
            for param in self.fallback_features.parameters():
                param.requires_grad = False
                
    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Preprocess audio for HuBERT"""
        # Convert to mono
        if audio.dim() > 1 and audio.shape[-2] > 1:
            audio = audio.mean(dim=-2, keepdim=True)
            
        # Resample if needed
        if self.sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                self.sample_rate, self.target_sample_rate
            ).to(audio.device)
            audio = resampler(audio)
            
        return audio
    
    def _extract_hubert_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract multi-layer HuBERT features"""
        batch_size = audio.shape[0]
        device = audio.device
        
        features = []
        
        for i in range(batch_size):
            # Process each sample individually
            sample = audio[i].squeeze().cpu().numpy()
            
            try:
                # Extract features using feature extractor
                inputs = self.feature_extractor(
                    sample,
                    sampling_rate=self.target_sample_rate,
                    return_tensors="pt"
                )
                
                # Move to correct device
                input_values = inputs.input_values.to(device)
                
                # Get HuBERT features
                with torch.no_grad():
                    outputs = self.hubert(input_values, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    
                # Weighted combination of layers
                weighted_features = torch.zeros_like(hidden_states[0])
                total_weight = 0
                
                for layer_idx, weight in enumerate(self.layer_weights):
                    if layer_idx < len(hidden_states):
                        weighted_features += weight * hidden_states[layer_idx]
                        total_weight += weight
                        
                if total_weight > 0:
                    weighted_features = weighted_features / total_weight
                    
                features.append(weighted_features.squeeze(0))
                
            except Exception as e:
                print(f"Warning: HuBERT feature extraction failed for sample {i}: {e}")
                # Fallback to zero features
                fallback_shape = (500, 768)  # Typical HuBERT output shape
                fallback_features = torch.zeros(fallback_shape, device=device)
                features.append(fallback_features)
                
        # Stack features
        try:
            # Pad to same length
            max_len = max(f.shape[0] for f in features)
            padded_features = []
            
            for f in features:
                if f.shape[0] < max_len:
                    pad_len = max_len - f.shape[0]
                    f = F.pad(f, (0, 0, 0, pad_len))
                padded_features.append(f)
                
            return torch.stack(padded_features, dim=0)
            
        except Exception as e:
            print(f"Warning: Feature stacking failed: {e}")
            # Return zero tensor
            return torch.zeros(batch_size, 500, 768, device=device)
    
    def _extract_fallback_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features using fallback CNN"""
        # Convert to mono if needed
        if audio.dim() > 2:
            audio = audio.mean(dim=1, keepdim=True)
        elif audio.dim() == 2 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True).unsqueeze(0)
            
        return self.fallback_features(audio)
    
    def forward(
        self,
        predicted_audio: torch.Tensor,
        target_audio: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute REPA loss between predicted and target audio
        
        Args:
            predicted_audio: (B, 2, T) predicted audio
            target_audio: (B, 2, T) target audio
            
        Returns:
            REPA loss
        """
        # Preprocess audio
        pred_processed = self._preprocess_audio(predicted_audio)
        target_processed = self._preprocess_audio(target_audio)
        
        # Extract features
        if self.hubert is not None:
            pred_features = self._extract_hubert_features(pred_processed)
            target_features = self._extract_hubert_features(target_processed)
        else:
            pred_features = self._extract_fallback_features(pred_processed)
            target_features = self._extract_fallback_features(target_processed)
            
        # Compute feature matching loss
        feature_loss = F.l1_loss(pred_features, target_features)
        
        # Additional perceptual losses
        cosine_loss = 1 - F.cosine_similarity(
            pred_features.flatten(1), 
            target_features.flatten(1),
            dim=1
        ).mean()
        
        # Combine losses
        total_loss = feature_loss + 0.1 * cosine_loss
        
        return total_loss


class ReconstructionLoss(nn.Module):
    """Multi-scale reconstruction loss"""
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 0.1,
        use_multi_scale: bool = True
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.use_multi_scale = use_multi_scale
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss
        
        Args:
            predicted: (B, 2, T) predicted audio
            target: (B, 2, T) target audio
            
        Returns:
            Reconstruction loss
        """
        # Match lengths
        min_length = min(predicted.shape[-1], target.shape[-1])
        predicted = predicted[..., :min_length]
        target = target[..., :min_length]
        
        # Basic L1 and L2 losses
        l1_loss = F.l1_loss(predicted, target)
        l2_loss = F.mse_loss(predicted, target)
        
        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss
        
        # Multi-scale loss
        if self.use_multi_scale:
            scales = [2, 4, 8, 16]
            
            for scale in scales:
                if min_length // scale > 10:
                    # Downsample
                    pred_ds = F.avg_pool1d(predicted, scale, scale)
                    target_ds = F.avg_pool1d(target, scale, scale)
                    
                    # Add multi-scale loss
                    ms_l1 = F.l1_loss(pred_ds, target_ds)
                    total_loss += 0.1 * ms_l1
                    
        return total_loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using spectral features"""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        mel_weight: float = 1.0,
        stft_weight: float = 0.5
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.mel_weight = mel_weight
        self.stft_weight = stft_weight
        
        # Mel spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            f_min=0.0,
            f_max=sample_rate // 2
        )
        
        # Multiple STFT scales
        self.stft_scales = [512, 1024, 2048]
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss
        
        Args:
            predicted: (B, 2, T) predicted audio
            target: (B, 2, T) target audio
            
        Returns:
            Perceptual loss
        """
        device = predicted.device
        
        # Move mel transform to correct device
        self.mel_transform = self.mel_transform.to(device)
        
        # Match lengths
        min_length = min(predicted.shape[-1], target.shape[-1])
        if min_length < 1024:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        predicted = predicted[..., :min_length]
        target = target[..., :min_length]
        
        total_loss = 0.0
        
        # Convert to mono for spectral analysis
        pred_mono = predicted.mean(dim=-2) if predicted.dim() > 1 else predicted
        target_mono = target.mean(dim=-2) if target.dim() > 1 else target
        
        # Mel spectrogram loss
        try:
            pred_mel = self.mel_transform(pred_mono)
            target_mel = self.mel_transform(target_mono)
            mel_loss = F.l1_loss(pred_mel, target_mel)
            total_loss += self.mel_weight * mel_loss
        except Exception:
            pass
            
        # Multi-scale STFT loss
        for n_fft in self.stft_scales:
            if min_length >= n_fft:
                hop_length = n_fft // 4
                
                try:
                    window = torch.hann_window(n_fft, device=device)
                    
                    pred_stft = torch.stft(
                        pred_mono.reshape(-1),
                        n_fft=n_fft,
                        hop_length=hop_length,
                        return_complex=True,
                        window=window
                    )
                    
                    target_stft = torch.stft(
                        target_mono.reshape(-1),
                        n_fft=n_fft,
                        hop_length=hop_length,
                        return_complex=True,
                        window=window
                    )
                    
                    # Magnitude loss
                    stft_loss = F.l1_loss(torch.abs(pred_stft), torch.abs(target_stft))
                    total_loss += self.stft_weight * stft_loss / len(self.stft_scales)
                    
                except Exception:
                    continue
                    
        return total_loss if total_loss > 0 else torch.tensor(0.1, device=device, requires_grad=True)


class AdversarialLoss(nn.Module):
    """Adversarial loss for improved quality"""
    
    def __init__(self):
        super().__init__()
        
        # Simple discriminator
        self.discriminator = nn.Sequential(
            nn.Conv1d(2, 32, 15, stride=1, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 41, stride=4, padding=20, groups=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 41, stride=4, padding=20, groups=64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adversarial loss
        
        Args:
            predicted: (B, 2, T) predicted audio
            target: (B, 2, T) target audio
            
        Returns:
            generator_loss, discriminator_loss
        """
        # Discriminator scores
        real_score = self.discriminator(target.detach())
        fake_score = self.discriminator(predicted)
        
        # Generator loss (fool discriminator)
        gen_loss = F.binary_cross_entropy_with_logits(
            fake_score, 
            torch.ones_like(fake_score)
        )
        
        # Discriminator loss
        real_loss = F.binary_cross_entropy_with_logits(
            real_score,
            torch.ones_like(real_score)
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_score.detach(),
            torch.zeros_like(fake_score)
        )
        disc_loss = (real_loss + fake_loss) / 2
        
        return gen_loss, disc_loss


class CombinedLoss(nn.Module):
    """Combined loss function for LYRO training"""
    
    def __init__(
        self,
        flow_loss: FlowMatchingLoss,
        repa_loss: REPALoss,
        recon_loss: ReconstructionLoss,
        perceptual_loss: PerceptualLoss,
        weights: Dict[str, float] = None
    ):
        super().__init__()
        
        self.flow_loss = flow_loss
        self.repa_loss = repa_loss
        self.recon_loss = recon_loss
        self.perceptual_loss = perceptual_loss
        
        # Default weights
        if weights is None:
            weights = {
                'flow': 1.0,
                'repa': 0.1,
                'recon': 0.5,
                'perceptual': 0.2
            }
        self.weights = weights
        
        # Adversarial loss (optional)
        self.adversarial_loss = AdversarialLoss()
        self.use_adversarial = False
        
    def forward(
        self,
        predicted_v: torch.Tensor,
        target_v: torch.Tensor,
        predicted_audio: torch.Tensor,
        target_audio: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            predicted_v: (B, C, L) predicted velocity
            target_v: (B, C, L) target velocity
            predicted_audio: (B, 2, T) predicted audio
            target_audio: (B, 2, T) target audio
            t: (B,) time steps
            mask: (B, C, L) optional mask
            
        Returns:
            Dictionary of individual and total losses
        """
        losses = {}
        
        # Flow matching loss
        flow_loss = self.flow_loss(predicted_v, target_v, t, mask)
        losses['flow'] = flow_loss
        
        # REPA loss
        repa_loss = self.repa_loss(predicted_audio, target_audio)
        losses['repa'] = repa_loss
        
        # Reconstruction loss
        recon_loss = self.recon_loss(predicted_audio, target_audio)
        losses['recon'] = recon_loss
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(predicted_audio, target_audio)
        losses['perceptual'] = perceptual_loss
        
        # Adversarial loss (if enabled)
        if self.use_adversarial:
            gen_loss, disc_loss = self.adversarial_loss(predicted_audio, target_audio)
            losses['adversarial_gen'] = gen_loss
            losses['adversarial_disc'] = disc_loss
        
        # Compute total loss
        total_loss = (
            self.weights.get('flow', 1.0) * flow_loss +
            self.weights.get('repa', 0.1) * repa_loss +
            self.weights.get('recon', 0.5) * recon_loss +
            self.weights.get('perceptual', 0.2) * perceptual_loss
        )
        
        if self.use_adversarial:
            total_loss += self.weights.get('adversarial', 0.01) * gen_loss
            
        losses['total'] = total_loss
        
        return losses
    
    def enable_adversarial(self, enable: bool = True):
        """Enable/disable adversarial training"""
        self.use_adversarial = enable
        
    def update_weights(self, new_weights: Dict[str, float]):
        """Update loss weights"""
        self.weights.update(new_weights)