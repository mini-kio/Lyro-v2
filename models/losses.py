import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
import warnings

warnings.filterwarnings("ignore")


class AlignmentLoss(nn.Module):
    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        
        if weights is None:
            weights = {
                'alignment': 1.0,
                'segment': 0.5,
                'tts': 0.3,
                'consistency': 0.2
            }
        
        self.weights = weights
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        total_loss = 0.0
        
        if 'alignment_targets' in targets:
            align_loss = self.mse_loss(
                predictions['alignment_scores'],
                targets['alignment_targets']
            )
            losses['alignment'] = align_loss
            total_loss += self.weights['alignment'] * align_loss
        
        if 'segment_targets' in targets:
            segment_loss = self.ce_loss(
                predictions['segment_probs'].view(-1, 3),
                targets['segment_targets'].view(-1).long()
            )
            losses['segment'] = segment_loss
            total_loss += self.weights['segment'] * segment_loss
        
        if 'tts_targets' in targets and 'tts_conditions' in predictions:
            tts_loss = self.mse_loss(
                predictions['tts_conditions'],
                targets['tts_targets']
            )
            losses['tts'] = tts_loss
            total_loss += self.weights['tts'] * tts_loss
        
        if 'audio_features' in predictions and 'speech_features' in predictions:
            audio_norm = F.normalize(predictions['audio_features'], dim=-1)
            speech_norm = F.normalize(predictions['speech_features'], dim=-1)
            
            consistency_loss = 1 - F.cosine_similarity(
                audio_norm, speech_norm, dim=-1
            ).mean()
            
            losses['consistency'] = consistency_loss
            total_loss += self.weights['consistency'] * consistency_loss
        
        losses['total'] = total_loss
        return losses


class FlowMatchingLoss(nn.Module):
    """Flow Matching Loss for continuous generation"""
    
    def __init__(
        self,
        beta_schedule: str = "cosine",
        num_timesteps: int = None,
        sigma: float = 1e-4
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.sigma = sigma
        self.beta_schedule = beta_schedule
        
        if num_timesteps is not None:
            self._create_beta_schedule(num_timesteps)
        
    def _create_beta_schedule(self, num_timesteps: int):
        """Create beta schedule dynamically based on timesteps"""
        self.num_timesteps = num_timesteps
        
        if self.beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, num_timesteps)
        elif self.beta_schedule == "cosine":
            steps = torch.arange(num_timesteps + 1) / num_timesteps
            alphas_cumprod = torch.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = torch.clamp(betas, 0.0001, 0.02)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
            
        self.register_buffer('betas', betas)
    
    def sync_with_sampler_steps(self, sampler_steps: int):
        """Synchronize beta schedule with sampler steps"""
        if self.num_timesteps != sampler_steps:
            self._create_beta_schedule(sampler_steps)
        
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
        if self.num_timesteps is None and hasattr(self, 'betas') and self.betas is not None:
            self.num_timesteps = len(self.betas)
        
        # Basic MSE loss
        loss = F.mse_loss(predicted_v, target_v, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
            
        # Time-dependent weighting using beta schedule
        if t is not None and hasattr(self, 'betas') and self.betas is not None:
            # Map continuous t [0,1] to discrete timestep indices
            t_indices = (t * (self.num_timesteps - 1)).long().clamp(0, self.num_timesteps - 1)
            beta_weights = self.betas[t_indices]
            weight = 1.0 + beta_weights.mean()
            loss = loss * weight
            
        return loss


class LatentConsistencyLoss(nn.Module):
    """
    Latent Consistency Loss for maintaining coherent latent representations
    """
    
    def __init__(
        self,
        consistency_weight: float = 1.0,
        temporal_weight: float = 0.5,
        channel_weight: float = 0.3
    ):
        super().__init__()
        
        self.consistency_weight = consistency_weight
        self.temporal_weight = temporal_weight
        self.channel_weight = channel_weight
        
    def forward(
        self,
        predicted_latents: torch.Tensor,
        target_latents: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute latent consistency loss
        
        Args:
            predicted_latents: (B, C, T) predicted latent vectors
            target_latents: (B, C, T) target latent vectors
            
        Returns:
            Latent consistency loss
        """
        # Basic reconstruction loss
        recon_loss = F.mse_loss(predicted_latents, target_latents)
        
        # Temporal consistency (neighboring time steps should be similar)
        pred_temporal_diff = torch.diff(predicted_latents, dim=-1)
        target_temporal_diff = torch.diff(target_latents, dim=-1)
        temporal_loss = F.mse_loss(pred_temporal_diff, target_temporal_diff)
        
        # Channel correlation (channels should maintain relative relationships)
        pred_channel_corr = self._compute_channel_correlation(predicted_latents)
        target_channel_corr = self._compute_channel_correlation(target_latents)
        channel_loss = F.mse_loss(pred_channel_corr, target_channel_corr)
        
        # Combined loss
        total_loss = (
            self.consistency_weight * recon_loss +
            self.temporal_weight * temporal_loss +
            self.channel_weight * channel_loss
        )
        
        return total_loss
    
    def _compute_channel_correlation(self, latents: torch.Tensor) -> torch.Tensor:
        """Compute correlation matrix between channels"""
        B, C, T = latents.shape
        
        # Reshape to (B, C, T) -> (B*T, C)
        latents_flat = latents.permute(0, 2, 1).reshape(-1, C)
        
        # Compute correlation matrix
        latents_centered = latents_flat - latents_flat.mean(dim=0, keepdim=True)
        cov_matrix = torch.mm(latents_centered.T, latents_centered) / (latents_centered.shape[0] - 1)
        
        # Normalize to correlation
        std_dev = torch.sqrt(torch.diag(cov_matrix)).unsqueeze(0)
        corr_matrix = cov_matrix / (std_dev.T @ std_dev + 1e-8)
        
        return corr_matrix


class LatentReconstructionLoss(nn.Module):
    """Multi-scale reconstruction loss for latent vectors"""
    
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
        Compute reconstruction loss for latent vectors
        
        Args:
            predicted: (B, C, T) predicted latent vectors
            target: (B, C, T) target latent vectors
            
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
        
        # Multi-scale loss for latent vectors
        if self.use_multi_scale:
            scales = [2, 4, 8]
            
            for scale in scales:
                if min_length // scale > 4:
                    # Downsample latents
                    pred_ds = F.avg_pool1d(predicted, scale, scale)
                    target_ds = F.avg_pool1d(target, scale, scale)
                    
                    # Add multi-scale loss
                    ms_l1 = F.l1_loss(pred_ds, target_ds)
                    total_loss += 0.1 * ms_l1
                    
        return total_loss


class LatentPerceptualLoss(nn.Module):
    """Perceptual loss for latent vectors using statistical features"""
    
    def __init__(
        self,
        stat_weight: float = 1.0,
        spectral_weight: float = 0.5
    ):
        super().__init__()
        
        self.stat_weight = stat_weight
        self.spectral_weight = spectral_weight
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss for latent vectors
        
        Args:
            predicted: (B, C, T) predicted latent vectors
            target: (B, C, T) target latent vectors
            
        Returns:
            Perceptual loss
        """
        device = predicted.device
        
        # Match lengths
        min_length = min(predicted.shape[-1], target.shape[-1])
        
        # Enhanced skip logic for short sequences
        if min_length < 16:  # Skip if too short for meaningful perceptual features
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Progressive scaling for medium-length sequences
        length_scale = 1.0
        if min_length < 32:
            length_scale = 0.5  # Reduce loss weight for medium sequences
        elif min_length < 64:
            length_scale = 0.75
            
        predicted = predicted[..., :min_length]
        target = target[..., :min_length]
        
        total_loss = 0.0
        
        # Statistical features loss (more robust for short sequences)
        try:
            stat_loss = self._compute_statistical_loss(predicted, target)
            total_loss += self.stat_weight * stat_loss * length_scale
        except Exception as e:
            print(f"Statistical loss computation failed: {e}")
        
        # Spectral features loss (skip for very short sequences)
        if min_length >= 32:  # Only compute spectral loss for longer sequences
            try:
                spectral_loss = self._compute_spectral_loss(predicted, target)
                total_loss += self.spectral_weight * spectral_loss * length_scale
            except Exception as e:
                print(f"Spectral loss computation failed: {e}")
                    
        return total_loss if total_loss > 0 else torch.tensor(0.01, device=device, requires_grad=True)
    
    def _compute_statistical_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss based on statistical moments"""
        # Mean
        pred_mean = torch.mean(predicted, dim=-1)
        target_mean = torch.mean(target, dim=-1)
        mean_loss = F.mse_loss(pred_mean, target_mean)
        
        # Variance
        pred_var = torch.var(predicted, dim=-1)
        target_var = torch.var(target, dim=-1)
        var_loss = F.mse_loss(pred_var, target_var)
        
        # Skewness (third moment)
        pred_centered = predicted - pred_mean.unsqueeze(-1)
        target_centered = target - target_mean.unsqueeze(-1)
        
        pred_skew = torch.mean(pred_centered ** 3, dim=-1)
        target_skew = torch.mean(target_centered ** 3, dim=-1)
        skew_loss = F.mse_loss(pred_skew, target_skew)
        
        return mean_loss + 0.5 * var_loss + 0.2 * skew_loss
    
    def _compute_spectral_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss based on frequency domain features"""
        try:
            # FFT of latent vectors
            pred_fft = torch.fft.fft(predicted, dim=-1)
            target_fft = torch.fft.fft(target, dim=-1)
            
            # Magnitude spectrum
            pred_mag = torch.abs(pred_fft)
            target_mag = torch.abs(target_fft)
            
            # Spectral loss
            spectral_loss = F.l1_loss(pred_mag, target_mag)
            
            return spectral_loss
            
        except Exception:
            return torch.tensor(0.0, device=predicted.device)


class LatentRegularizationLoss(nn.Module):
    """Regularization loss for latent vectors to encourage smooth and realistic distributions"""
    
    def __init__(
        self,
        smoothness_weight: float = 0.1,
        sparsity_weight: float = 0.05,
        norm_weight: float = 0.1
    ):
        super().__init__()
        
        self.smoothness_weight = smoothness_weight
        self.sparsity_weight = sparsity_weight
        self.norm_weight = norm_weight
        
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization loss for latent vectors
        
        Args:
            latents: (B, C, T) latent vectors
            
        Returns:
            Regularization loss
        """
        total_loss = 0.0
        
        # Smoothness loss (encourage temporal smoothness)
        if self.smoothness_weight > 0:
            temporal_diff = torch.diff(latents, dim=-1)
            smoothness_loss = torch.mean(temporal_diff ** 2)
            total_loss += self.smoothness_weight * smoothness_loss
        
        # Sparsity loss (encourage some channels to be inactive)
        if self.sparsity_weight > 0:
            l1_norm = torch.mean(torch.abs(latents))
            total_loss += self.sparsity_weight * l1_norm
        
        # Norm regularization (prevent explosion)
        if self.norm_weight > 0:
            l2_norm = torch.mean(latents ** 2)
            total_loss += self.norm_weight * l2_norm
        
        return total_loss


class CombinedLoss(nn.Module):
    """Combined loss function for LYRO Generator training (수정됨 - Latent 전용)"""
    
    def __init__(
        self,
        flow_loss: FlowMatchingLoss,
        consistency_loss: LatentConsistencyLoss = None,
        recon_loss: LatentReconstructionLoss = None,
        perceptual_loss: LatentPerceptualLoss = None,
        regularization_loss: LatentRegularizationLoss = None,
        weights: Dict[str, float] = None
    ):
        super().__init__()
        
        self.flow_loss = flow_loss
        self.consistency_loss = consistency_loss or LatentConsistencyLoss()
        self.recon_loss = recon_loss or LatentReconstructionLoss()
        self.perceptual_loss = perceptual_loss or LatentPerceptualLoss()
        self.regularization_loss = regularization_loss or LatentRegularizationLoss()
        
        # Default weights
        if weights is None:
            weights = {
                'flow': 1.0,
                'consistency': 0.3,
                'recon': 0.5,
                'perceptual': 0.2,
                'regularization': 0.1
            }
        self.weights = weights
        
    def forward(
        self,
        predicted_v: torch.Tensor,
        target_v: torch.Tensor,
        predicted_latents: torch.Tensor,
        target_latents: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for latent vector generation
        
        Args:
            predicted_v: (B, C, L) predicted velocity
            target_v: (B, C, L) target velocity
            predicted_latents: (B, C, L) predicted latent vectors
            target_latents: (B, C, L) target latent vectors
            t: (B,) time steps
            mask: (B, C, L) optional mask
            
        Returns:
            Dictionary of individual and total losses
        """
        losses = {}
        
        # Flow matching loss
        flow_loss = self.flow_loss(predicted_v, target_v, t, mask)
        losses['flow'] = flow_loss
        
        # Latent consistency loss
        consistency_loss = self.consistency_loss(predicted_latents, target_latents)
        losses['consistency'] = consistency_loss
        
        # Reconstruction loss
        recon_loss = self.recon_loss(predicted_latents, target_latents)
        losses['recon'] = recon_loss
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(predicted_latents, target_latents)
        losses['perceptual'] = perceptual_loss
        
        # Regularization loss
        regularization_loss = self.regularization_loss(predicted_latents)
        losses['regularization'] = regularization_loss
        
        # Compute total loss
        total_loss = (
            self.weights.get('flow', 1.0) * flow_loss +
            self.weights.get('consistency', 0.3) * consistency_loss +
            self.weights.get('recon', 0.5) * recon_loss +
            self.weights.get('perceptual', 0.2) * perceptual_loss +
            self.weights.get('regularization', 0.1) * regularization_loss
        )
            
        losses['total'] = total_loss
        
        return losses
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update loss weights"""
        self.weights.update(new_weights)


# Legacy classes for backward compatibility (simplified)
class REPALoss(nn.Module):
    """Simplified REPA Loss for backward compatibility"""
    
    def __init__(self, hubert_model: str = None, layer_weights: List[float] = None):
        super().__init__()
        print("Warning: REPALoss simplified for latent-only training")
        
    def forward(self, predicted_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        # Return dummy loss for compatibility
        return torch.tensor(0.0, device=predicted_audio.device, requires_grad=True)


class ReconstructionLoss(LatentReconstructionLoss):
    """Alias for LatentReconstructionLoss"""
    pass


class PerceptualLoss(LatentPerceptualLoss):
    """Alias for LatentPerceptualLoss"""
    pass