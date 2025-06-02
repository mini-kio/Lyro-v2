# lyro/dcae/training_utils.py
"""
Enhanced Training Utilities for CQT-SSM DCAE
T-3: EMA (Exponential Moving Average)
T-1: Mix-scale Augmentation
Updated for CQT-based music processing
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
import random
from typing import Dict, Optional, Any, Union, List
from collections import defaultdict
import copy
import librosa


# ==================== T-3: Exponential Moving Average ====================

class EMAWrapper:
    """
    Exponential Moving Average for model parameters
    Improves inference quality by maintaining shadow parameters
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
        update_after: int = 100,  # Start EMA after this many steps
        update_every: int = 10,   # Update EMA every N steps
    ):
        """
        Args:
            model: Model to apply EMA to
            decay: EMA decay rate (0.999 = very slow decay)
            device: Device to store shadow parameters
            update_after: Start EMA updates after this many steps
            update_every: Update EMA every N steps
        """
        self.model = model
        self.decay = decay
        self.device = device or next(model.parameters()).device
        self.update_after = update_after
        self.update_every = update_every
        
        self.step_count = 0
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        self.initialize_shadow_params()
        
    def initialize_shadow_params(self):
        """Initialize shadow parameters with model's current parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach().to(self.device)
    
    def update(self):
        """Update EMA parameters"""
        self.step_count += 1
        
        # Start updating after specified steps
        if self.step_count <= self.update_after:
            return
            
        # Update every N steps
        if self.step_count % self.update_every != 0:
            return
            
        # Update shadow parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.to(self.device), alpha=1.0 - self.decay
                )
    
    def apply_shadow(self):
        """Apply shadow parameters to model (for inference)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore_original(self):
        """Restore original parameters (after inference)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()
    
    def state_dict(self):
        """Get EMA state dict for saving"""
        return {
            'shadow': self.shadow,
            'step_count': self.step_count,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict"""
        self.shadow = state_dict['shadow']
        self.step_count = state_dict['step_count']
        self.decay = state_dict.get('decay', self.decay)
    
    def copy_to(self, other_model):
        """Copy EMA parameters to another model"""
        for (name, param), (other_name, other_param) in zip(
            self.model.named_parameters(), other_model.named_parameters()
        ):
            if param.requires_grad and name in self.shadow:
                other_param.data.copy_(self.shadow[name])


class EMAContext:
    """Context manager for EMA inference"""
    
    def __init__(self, ema_wrapper: EMAWrapper):
        self.ema_wrapper = ema_wrapper
    
    def __enter__(self):
        self.ema_wrapper.apply_shadow()
        return self.ema_wrapper.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ema_wrapper.restore_original()


# ==================== T-1: Advanced Audio Augmentation ====================

class MixScaleAugmentation:
    """
    Advanced mix-scale audio augmentation for better generalization
    Includes gain, filtering, pitch shift, and other audio effects
    Optimized for CQT-based music processing
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        augmentation_prob: float = 0.8,
        gain_range: tuple = (-3.0, 3.0),  # dB range
        pitch_range: tuple = (-0.5, 0.5),  # semitones
        tempo_range: tuple = (0.9, 1.1),   # tempo multiplier
        noise_level: float = 0.005,         # background noise level
        reverb_prob: float = 0.3,           # probability of reverb
        eq_prob: float = 0.4,               # probability of EQ
        # CQT-specific parameters
        preserve_musical_structure: bool = True,  # Preserve harmonic content for CQT
        harmonic_distortion_prob: float = 0.2,    # Probability of harmonic distortion
    ):
        self.sample_rate = sample_rate
        self.augmentation_prob = augmentation_prob
        self.gain_range = gain_range
        self.pitch_range = pitch_range
        self.tempo_range = tempo_range
        self.noise_level = noise_level
        self.reverb_prob = reverb_prob
        self.eq_prob = eq_prob
        self.preserve_musical_structure = preserve_musical_structure
        self.harmonic_distortion_prob = harmonic_distortion_prob
        
        # Pre-built transforms optimized for musical content
        # Use Biquad filters instead of deprecated HighpassBiquad/LowpassBiquad
        self.highpass_filters = [
            T.Biquad(sample_rate, b0=1, b1=-1, b2=0, a0=1, a1=-0.9, a2=0)  # Simple highpass
            for freq in [60, 80, 100, 120]  # Lower frequencies to preserve musical content
        ]
        self.lowpass_filters = [
            T.Biquad(sample_rate, b0=0.5, b1=0.5, b2=0, a0=1, a1=-0.5, a2=0)  # Simple lowpass
            for freq in [8000, 12000, 16000, 18000]
        ]
        
        self.eq_filters = self._create_musical_eq_filters()
    
    def _create_musical_eq_filters(self):
        """Create parametric EQ filters optimized for musical content"""
        eq_filters = []
        
        # Use simple Biquad filters instead of deprecated BandpassBiquad
        # These coefficients approximate bandpass filters for different frequency bands
        filter_configs = [
            # freq, b0, b1, b2, a0, a1, a2 (simplified bandpass approximations)
            (40, 0.1, 0, -0.1, 1, -1.8, 0.85),    # Sub-bass
            (80, 0.15, 0, -0.15, 1, -1.7, 0.8),   # Bass low
            (160, 0.2, 0, -0.2, 1, -1.6, 0.75),   # Bass high
            (350, 0.25, 0, -0.25, 1, -1.4, 0.7),  # Low-mids
            (800, 0.3, 0, -0.3, 1, -1.2, 0.65),   # Mids low
            (1600, 0.35, 0, -0.35, 1, -1.0, 0.6), # Mids high
            (3000, 0.4, 0, -0.4, 1, -0.8, 0.55),  # Upper-mids
            (6000, 0.3, 0, -0.3, 1, -0.6, 0.5),   # Highs
            (12000, 0.2, 0, -0.2, 1, -0.4, 0.45)  # Air
        ]
        
        for freq, b0, b1, b2, a0, a1, a2 in filter_configs:
            eq_filters.append(T.Biquad(self.sample_rate, b0=b0, b1=b1, b2=b2, a0=a0, a1=a1, a2=a2))
        
        return eq_filters
    
    def apply_gain_augmentation(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random gain adjustment"""
        if random.random() > 0.7:  # 70% probability
            gain_db = random.uniform(*self.gain_range)
            gain_linear = 10 ** (gain_db / 20)
            audio = audio * gain_linear
        return audio
    
    def apply_pitch_shift(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random pitch shifting (conservative for musical content)"""
        if random.random() > 0.85:  # 15% probability (expensive operation)
            n_steps = random.uniform(*self.pitch_range)
            if abs(n_steps) > 0.1:  # Only apply if significant shift
                # Apply to each channel separately
                audio_shifted = []
                for channel in range(audio.shape[0]):
                    try:
                        shifted = F.pitch_shift(
                            audio[channel:channel+1], 
                            self.sample_rate, 
                            n_steps=n_steps
                        )
                        audio_shifted.append(shifted)
                    except:
                        # Fallback if pitch shift fails
                        audio_shifted.append(audio[channel:channel+1])
                audio = torch.cat(audio_shifted, dim=0)
        return audio
    
    def apply_musical_filtering(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply musical filtering effects optimized for CQT"""
        # High-pass filtering (conservative to preserve bass content)
        if random.random() < 0.25:  # 25% probability
            hp_filter = random.choice(self.highpass_filters)
            audio = hp_filter(audio)
        
        # Low-pass filtering (conservative to preserve high-frequency content)
        if random.random() < 0.15:  # 15% probability
            lp_filter = random.choice(self.lowpass_filters)
            audio = lp_filter(audio)
        
        # Musical parametric EQ
        if random.random() < self.eq_prob:
            eq_filter = random.choice(self.eq_filters)
            # Apply with random gain (more conservative for musical content)
            eq_gain = random.uniform(0.7, 1.3)
            audio_eq = eq_filter(audio)
            audio = audio + eq_gain * (audio_eq - audio) * 0.5  # Reduced intensity
        
        return audio
    
    def apply_harmonic_distortion(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply subtle harmonic distortion for CQT robustness"""
        if random.random() < self.harmonic_distortion_prob:
            # Subtle harmonic distortion
            distortion_amount = random.uniform(0.1, 0.3)
            
            # Soft clipping with musical character
            audio = torch.tanh(audio * (1 + distortion_amount)) / (1 + distortion_amount)
            
            # Add subtle odd harmonics
            if random.random() < 0.5:
                harmonics = torch.sin(audio * 3) * 0.02  # 3rd harmonic
                audio = audio + harmonics
        
        return audio
    
    def apply_musical_reverb(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply reverb simulation optimized for musical content"""
        if random.random() < self.reverb_prob:
            # Multiple delay lines for richer reverb
            num_delays = random.randint(2, 4)
            reverb_audio = audio.clone()
            
            for _ in range(num_delays):
                delay_samples = random.randint(800, 3000)  # 18-68ms at 44.1kHz
                decay = random.uniform(0.05, 0.2)  # Lighter reverb
                
                if delay_samples < audio.shape[-1]:
                    delayed = torch.zeros_like(audio)
                    delayed[..., delay_samples:] = audio[..., :-delay_samples] * decay
                    reverb_audio = reverb_audio + delayed
            
            # Mix with dry signal
            wet_amount = random.uniform(0.1, 0.3)
            audio = audio * (1 - wet_amount) + reverb_audio * wet_amount
        
        return audio
    
    def apply_musical_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Add musical noise for robustness"""
        if random.random() < 0.3:  # 30% probability
            noise_type = random.choice(['white', 'pink', 'brown'])
            noise_level = random.uniform(0, self.noise_level)
            
            if noise_type == 'white':
                noise = torch.randn_like(audio) * noise_level
            elif noise_type == 'pink':
                # Approximate pink noise (1/f)
                noise = torch.randn_like(audio) * noise_level
                # Simple low-pass to approximate pink noise
                if audio.shape[-1] > 100:
                    kernel = torch.ones(1, 1, 5, device=audio.device) / 5
                    noise = F.conv1d(noise.unsqueeze(1), kernel, padding=2).squeeze(1)
            else:  # brown noise
                # Approximate brown noise (1/f^2)
                noise = torch.randn_like(audio) * noise_level * 0.5
                if audio.shape[-1] > 100:
                    kernel = torch.ones(1, 1, 10, device=audio.device) / 10
                    noise = F.conv1d(noise.unsqueeze(1), kernel, padding=5).squeeze(1)
            
            audio = audio + noise
        
        return audio
    
    def apply_stereo_effects(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply stereo effects for spatial augmentation"""
        if audio.shape[0] == 2 and random.random() < 0.4:
            effect_type = random.choice(['width', 'pan', 'phase'])
            
            if effect_type == 'width':
                # Stereo width adjustment
                width = random.uniform(0.8, 1.2)
                mid = (audio[0] + audio[1]) / 2
                side = (audio[0] - audio[1]) / 2 * width
                audio[0] = mid + side
                audio[1] = mid - side
            elif effect_type == 'pan':
                # Subtle panning
                pan_amount = random.uniform(-0.2, 0.2)
                gain_l = 1 - max(0, pan_amount)
                gain_r = 1 + min(0, pan_amount)
                audio[0] *= gain_l
                audio[1] *= gain_r
            elif effect_type == 'phase':
                # Subtle phase adjustment
                if random.random() < 0.3:
                    phase_shift = random.randint(1, 10)
                    if phase_shift < audio.shape[-1]:
                        audio[1] = torch.roll(audio[1], phase_shift)
        
        return audio
    
    def apply_dynamics_processing(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply dynamic range effects optimized for music"""
        # Gentle compression simulation
        if random.random() < 0.4:
            threshold = random.uniform(0.6, 0.9)
            ratio = random.uniform(1.5, 3.0)
            
            audio_abs = torch.abs(audio)
            over_threshold = audio_abs > threshold
            
            if over_threshold.any():
                compression_factor = 1 + (ratio - 1) * (audio_abs - threshold) / (1 - threshold)
                compression_factor = torch.clamp(compression_factor, 1.0, ratio)
                audio = torch.where(over_threshold, audio / compression_factor, audio)
        
        return audio
    
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply CQT-optimized mix-scale augmentation pipeline
        
        Args:
            audio: (C, T) audio tensor
        Returns:
            augmented_audio: (C, T) augmented audio
        """
        if random.random() > self.augmentation_prob:
            return audio
        
        # Ensure audio is in correct format
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Apply augmentations in sequence (optimized order for musical content)
        audio = self.apply_gain_augmentation(audio)
        audio = self.apply_musical_filtering(audio)
        audio = self.apply_musical_reverb(audio)
        audio = self.apply_musical_noise(audio)
        audio = self.apply_stereo_effects(audio)
        
        # Apply potentially destructive effects last and sparingly
        if self.preserve_musical_structure:
            if random.random() < 0.7:  # Only apply these 70% of the time
                audio = self.apply_harmonic_distortion(audio)
            if random.random() < 0.8:  # Pitch shift even less frequently
                audio = self.apply_pitch_shift(audio)
        else:
            audio = self.apply_harmonic_distortion(audio)
            audio = self.apply_pitch_shift(audio)
        
        audio = self.apply_dynamics_processing(audio)
        
        # Normalize to prevent clipping
        max_val = torch.abs(audio).max()
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)
        
        return audio


# ==================== Enhanced Training Configuration ====================

class EnhancedDCAEConfig:
    """Enhanced configuration with EMA and augmentation settings for CQT-SSM"""
    
    def __init__(self):
        # Model architecture (CQT-based)
        self.latent_channels = 8
        self.encoder_base_channels = 64
        self.decoder_base_channels = 64
        self.use_vector_quantization = False
        self.dual_channel_processing = True
        self.use_weight_norm = True  # T-2: Weight Normalization
        
        # CQT Configuration
        self.sample_rate = 44100
        self.n_bins = 84  # 7 octaves
        self.hop_length = 512
        self.bins_per_octave = 12
        self.fmin = 32.7  # C1
        
        # Training
        self.learning_rate = 3e-4
        self.weight_decay = 1e-2
        self.batch_size = 8
        self.epochs = 150
        self.grad_clip = 1.0
        
        # T-3: EMA Configuration
        self.use_ema = True
        self.ema_decay = 0.999
        self.ema_update_after = 100
        self.ema_update_every = 10
        
        # T-1: Augmentation Configuration (CQT-optimized)
        self.use_augmentation = True
        self.augmentation_prob = 0.8
        self.gain_range = (-3.0, 3.0)
        self.pitch_range = (-0.5, 0.5)
        self.tempo_range = (0.9, 1.1)
        self.noise_level = 0.005
        self.reverb_prob = 0.3
        self.eq_prob = 0.4
        self.preserve_musical_structure = True
        self.harmonic_distortion_prob = 0.2
        
        # Loss weights (A-5: Enhanced CQT loss)
        self.cqt_weight = 1.0  # Changed from stft_weight
        self.time_weight = 0.1
        self.vq_weight = 0.02
        self.adv_weight = 0.1
        
        # CQT Loss Configuration
        self.cqt_hop_lengths = [256, 512, 1024]  # Multi-resolution CQT
        self.cqt_n_bins_list = [72, 84, 96]      # Different octave ranges
        self.w_cqt = 1.0
        self.w_temporal = 0.5
        self.w_harmonic = 0.3
        
        # Scheduler
        self.scheduler_type = "cosine"
        self.min_lr = 1e-6
        self.warmup_epochs = 10
        
        # Data
        self.max_length = 44100 * 10  # 10 seconds max
    
    def create_augmentation(self) -> Optional[MixScaleAugmentation]:
        """Create CQT-optimized augmentation pipeline"""
        if not self.use_augmentation:
            return None
            
        return MixScaleAugmentation(
            sample_rate=self.sample_rate,
            augmentation_prob=self.augmentation_prob,
            gain_range=self.gain_range,
            pitch_range=self.pitch_range,
            tempo_range=self.tempo_range,
            noise_level=self.noise_level,
            reverb_prob=self.reverb_prob,
            eq_prob=self.eq_prob,
            preserve_musical_structure=self.preserve_musical_structure,
            harmonic_distortion_prob=self.harmonic_distortion_prob
        )
    
    def create_ema_wrapper(self, model: nn.Module) -> Optional[EMAWrapper]:
        """Create EMA wrapper for model"""
        if not self.use_ema:
            return None
            
        return EMAWrapper(
            model=model,
            decay=self.ema_decay,
            update_after=self.ema_update_after,
            update_every=self.ema_update_every
        )
    
    def get_optimizer(self, model):
        """Get optimizer"""
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
    
    def get_scheduler(self, optimizer, total_steps):
        """Get scheduler"""
        if self.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=self.min_lr
            )
        elif self.scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
            )
        else:
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=total_steps//3, gamma=0.5
            )


# ==================== Training State Manager ====================

class TrainingStateManager:
    """Manages training state including EMA, augmentation, and metrics for CQT-SSM"""
    
    def __init__(self, config: EnhancedDCAEConfig):
        self.config = config
        self.metrics_history = defaultdict(list)
        self.best_metrics = {}
        self.current_epoch = 0
        self.global_step = 0
        
        # CQT-optimized augmentation pipeline
        self.augmentation = config.create_augmentation()
        
        # EMA will be initialized when model is provided
        self.ema_wrapper = None
    
    def setup_ema(self, model: nn.Module):
        """Setup EMA wrapper for model"""
        self.ema_wrapper = self.config.create_ema_wrapper(model)
    
    def update_ema(self):
        """Update EMA if enabled"""
        if self.ema_wrapper is not None:
            self.ema_wrapper.update()
    
    def apply_augmentation(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply CQT-optimized augmentation if enabled"""
        if self.augmentation is not None:
            return self.augmentation(audio)
        return audio
    
    def record_metrics(self, metrics: Dict[str, float]):
        """Record training metrics"""
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
            
            # Update best metrics
            if key not in self.best_metrics:
                self.best_metrics[key] = float('inf') if 'loss' in key else float('-inf')
            
            if 'loss' in key and value < self.best_metrics[key]:
                self.best_metrics[key] = value
            elif 'loss' not in key and value > self.best_metrics[key]:
                self.best_metrics[key] = value
    
    def get_ema_context(self) -> Optional[EMAContext]:
        """Get EMA context for inference"""
        if self.ema_wrapper is not None:
            return EMAContext(self.ema_wrapper)
        return None
    
    def state_dict(self) -> Dict[str, Any]:
        """Get training state dict"""
        state = {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'metrics_history': dict(self.metrics_history),
            'best_metrics': self.best_metrics
        }
        
        if self.ema_wrapper is not None:
            state['ema'] = self.ema_wrapper.state_dict()
            
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load training state dict"""
        self.current_epoch = state_dict.get('current_epoch', 0)
        self.global_step = state_dict.get('global_step', 0)
        self.metrics_history = defaultdict(list, state_dict.get('metrics_history', {}))
        self.best_metrics = state_dict.get('best_metrics', {})
        
        if 'ema' in state_dict and self.ema_wrapper is not None:
            self.ema_wrapper.load_state_dict(state_dict['ema'])


# ==================== Utility Functions ====================

def compute_snr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio in dB"""
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    
    if noise_power > 0:
        snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        return float(snr_db)
    else:
        return float('inf')


def compute_si_sdr(reference: torch.Tensor, estimation: torch.Tensor) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio"""
    # Zero-mean
    reference = reference - torch.mean(reference)
    estimation = estimation - torch.mean(estimation)
    
    # Handle zero-energy signals
    if torch.sum(reference ** 2) < 1e-8:
        return 0.0
    
    # Scale invariant target
    alpha = torch.sum(estimation * reference) / (torch.sum(reference ** 2) + 1e-8)
    target = alpha * reference
    
    # SI-SDR with numerical stability
    target_power = torch.sum(target ** 2)
    noise_power = torch.sum((estimation - target) ** 2)
    
    if noise_power > 0 and target_power > 0:
        si_sdr = 10 * torch.log10(target_power / (noise_power + 1e-8))
        return float(si_sdr)
    else:
        return 0.0


def analyze_frequency_response(
    original: torch.Tensor, 
    reconstructed: torch.Tensor,
    sample_rate: int = 44100,
    use_cqt: bool = True
) -> Dict[str, float]:
    """
    Analyze frequency response quality using CQT or STFT
    Enhanced for CQT-based models
    """
    if use_cqt:
        return analyze_cqt_response(original, reconstructed, sample_rate)
    else:
        return analyze_stft_response(original, reconstructed, sample_rate)


def analyze_cqt_response(
    original: torch.Tensor,
    reconstructed: torch.Tensor, 
    sample_rate: int = 44100
) -> Dict[str, float]:
    """Analyze frequency response using CQT (optimized for music)"""
    try:
        # Convert to numpy for librosa
        orig_np = original.mean(0).detach().cpu().numpy()
        recon_np = reconstructed.mean(0).detach().cpu().numpy()
        
        # CQT analysis
        orig_cqt = np.abs(librosa.cqt(orig_np, sr=sample_rate, hop_length=512, n_bins=84))
        recon_cqt = np.abs(librosa.cqt(recon_np, sr=sample_rate, hop_length=512, n_bins=84))
        
        # Musical frequency bands analysis
        n_bins = orig_cqt.shape[0]
        
        # Low notes (C1-C3): bins 0-24
        low_end = min(24, n_bins)
        # Mid notes (C3-C6): bins 24-60  
        mid_end = min(60, n_bins)
        # High notes (C6+): bins 60+
        
        low_error = np.mean(np.abs(recon_cqt[:low_end] - orig_cqt[:low_end]))
        mid_error = np.mean(np.abs(recon_cqt[low_end:mid_end] - orig_cqt[low_end:mid_end]))
        high_error = np.mean(np.abs(recon_cqt[mid_end:] - orig_cqt[mid_end:]))
        
        # Overall spectral similarity
        total_error = np.mean(np.abs(recon_cqt - orig_cqt))
        
        # Musical correlation
        correlation = np.corrcoef(orig_cqt.flatten(), recon_cqt.flatten())[0, 1]
        correlation = correlation if not np.isnan(correlation) else 0.0
        
        return {
            'low_freq_error': float(low_error),
            'mid_freq_error': float(mid_error), 
            'high_freq_error': float(high_error),
            'total_spectral_error': float(total_error),
            'spectral_correlation': float(correlation),
            'analysis_type': 'CQT'
        }
        
    except Exception as e:
        # Fallback to basic metrics
        return {
            'low_freq_error': 0.0,
            'mid_freq_error': 0.0,
            'high_freq_error': 0.0,
            'total_spectral_error': 0.0,
            'spectral_correlation': 0.0,
            'analysis_type': 'CQT_FALLBACK'
        }


def analyze_stft_response(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    sample_rate: int = 44100
) -> Dict[str, float]:
    """Analyze frequency response using STFT (fallback method)"""
    try:
        # Compute STFT
        n_fft = 2048
        original_stft = torch.stft(original.mean(0), n_fft, return_complex=True)
        reconstructed_stft = torch.stft(reconstructed.mean(0), n_fft, return_complex=True)
        
        # Magnitude spectra
        original_mag = torch.abs(original_stft)
        reconstructed_mag = torch.abs(reconstructed_stft)
        
        # Frequency bands analysis
        freq_bins = original_mag.shape[0]
        
        # Low (0-250 Hz), Mid (250-4000 Hz), High (4000+ Hz)
        low_end = int(250 * freq_bins / (sample_rate / 2))
        mid_end = int(4000 * freq_bins / (sample_rate / 2))
        
        low_error = F.l1_loss(reconstructed_mag[:low_end], original_mag[:low_end])
        mid_error = F.l1_loss(reconstructed_mag[low_end:mid_end], original_mag[low_end:mid_end])
        high_error = F.l1_loss(reconstructed_mag[mid_end:], original_mag[mid_end:])
        
        return {
            'low_freq_error': float(low_error),
            'mid_freq_error': float(mid_error), 
            'high_freq_error': float(high_error),
            'total_spectral_error': float(F.l1_loss(reconstructed_mag, original_mag)),
            'analysis_type': 'STFT'
        }
        
    except Exception:
        return {
            'low_freq_error': 0.0,
            'mid_freq_error': 0.0,
            'high_freq_error': 0.0,
            'total_spectral_error': 0.0,
            'analysis_type': 'STFT_FALLBACK'
        }


def compute_musical_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    sample_rate: int = 44100
) -> Dict[str, float]:
    """
    Compute music-specific quality metrics for CQT-based models
    """
    metrics = {}
    
    try:
        # Convert to numpy for librosa analysis
        orig_np = original.mean(0).detach().cpu().numpy()
        recon_np = reconstructed.mean(0).detach().cpu().numpy()
        
        # Ensure same length
        min_len = min(len(orig_np), len(recon_np))
        orig_np = orig_np[:min_len]
        recon_np = recon_np[:min_len]
        
        # Tempo consistency
        try:
            orig_tempo = librosa.beat.tempo(y=orig_np, sr=sample_rate)[0]
            recon_tempo = librosa.beat.tempo(y=recon_np, sr=sample_rate)[0]
            metrics['tempo_error'] = abs(orig_tempo - recon_tempo)
        except:
            metrics['tempo_error'] = 0.0
        
        # Harmonic-percussive separation quality
        try:
            orig_harmonic, orig_percussive = librosa.effects.hpss(orig_np)
            recon_harmonic, recon_percussive = librosa.effects.hpss(recon_np)
            
            # Harmonic preservation
            h_corr = np.corrcoef(orig_harmonic, recon_harmonic)[0, 1]
            metrics['harmonic_preservation'] = h_corr if not np.isnan(h_corr) else 0.0
            
            # Percussive preservation
            p_corr = np.corrcoef(orig_percussive, recon_percussive)[0, 1]
            metrics['percussive_preservation'] = p_corr if not np.isnan(p_corr) else 0.0
        except:
            metrics['harmonic_preservation'] = 0.0
            metrics['percussive_preservation'] = 0.0
        
        # Chroma similarity (harmonic content)
        try:
            orig_chroma = librosa.feature.chroma_cqt(y=orig_np, sr=sample_rate)
            recon_chroma = librosa.feature.chroma_cqt(y=recon_np, sr=sample_rate)
            
            min_frames = min(orig_chroma.shape[1], recon_chroma.shape[1])
            if min_frames > 0:
                orig_chroma = orig_chroma[:, :min_frames]
                recon_chroma = recon_chroma[:, :min_frames]
                
                chroma_sim = np.mean([
                    np.corrcoef(orig_chroma[i], recon_chroma[i])[0, 1]
                    for i in range(12) if not np.all(orig_chroma[i] == 0)
                ])
                metrics['chroma_similarity'] = chroma_sim if not np.isnan(chroma_sim) else 0.0
            else:
                metrics['chroma_similarity'] = 0.0
        except:
            metrics['chroma_similarity'] = 0.0
        
        # Spectral centroid (brightness)
        try:
            orig_centroid = librosa.feature.spectral_centroid(y=orig_np, sr=sample_rate)[0]
            recon_centroid = librosa.feature.spectral_centroid(y=recon_np, sr=sample_rate)[0]
            metrics['spectral_centroid_error'] = np.mean(np.abs(orig_centroid - recon_centroid))
        except:
            metrics['spectral_centroid_error'] = 0.0
            
    except Exception:
        # Fallback metrics
        metrics = {
            'tempo_error': 0.0,
            'harmonic_preservation': 0.0,
            'percussive_preservation': 0.0,
            'chroma_similarity': 0.0,
            'spectral_centroid_error': 0.0
        }
    
    return metrics