# lyro/dcae/training_utils.py
"""
Enhanced Training Utilities
T-3: EMA (Exponential Moving Average)
T-1: Mix-scale Augmentation
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
import random
from typing import Dict, Optional, Any, Union
from collections import defaultdict
import copy


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
    ):
        self.sample_rate = sample_rate
        self.augmentation_prob = augmentation_prob
        self.gain_range = gain_range
        self.pitch_range = pitch_range
        self.tempo_range = tempo_range
        self.noise_level = noise_level
        self.reverb_prob = reverb_prob
        self.eq_prob = eq_prob
        
        # Pre-built transforms
        self.highpass_filters = [
            T.HighpassBiquad(sample_rate, cutoff_freq=freq)
            for freq in [80, 100, 120, 150]
        ]
        
        self.lowpass_filters = [
            T.LowpassBiquad(sample_rate, cutoff_freq=freq) 
            for freq in [8000, 12000, 16000, 18000]
        ]
        
        self.eq_filters = self._create_eq_filters()
    
    def _create_eq_filters(self):
        """Create parametric EQ filters for different frequency bands"""
        eq_filters = []
        
        # Bass boost/cut
        eq_filters.extend([
            T.BandpassBiquad(self.sample_rate, central_freq=80, Q=0.7),
            T.BandpassBiquad(self.sample_rate, central_freq=200, Q=1.0),
        ])
        
        # Mid-range
        eq_filters.extend([
            T.BandpassBiquad(self.sample_rate, central_freq=1000, Q=1.0),
            T.BandpassBiquad(self.sample_rate, central_freq=3000, Q=1.0),
        ])
        
        # High frequencies
        eq_filters.extend([
            T.BandpassBiquad(self.sample_rate, central_freq=8000, Q=0.7),
            T.BandpassBiquad(self.sample_rate, central_freq=12000, Q=0.7),
        ])
        
        return eq_filters
    
    def apply_gain_augmentation(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random gain adjustment"""
        if random.random() > 0.7:  # 70% probability
            gain_db = random.uniform(*self.gain_range)
            gain_linear = 10 ** (gain_db / 20)
            audio = audio * gain_linear
        return audio
    
    def apply_pitch_shift(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random pitch shifting"""
        if random.random() > 0.8:  # 20% probability (expensive operation)
            n_steps = random.uniform(*self.pitch_range)
            if abs(n_steps) > 0.1:  # Only apply if significant shift
                # Apply to each channel separately
                audio_shifted = []
                for channel in range(audio.shape[0]):
                    shifted = F.pitch_shift(
                        audio[channel:channel+1], 
                        self.sample_rate, 
                        n_steps=n_steps
                    )
                    audio_shifted.append(shifted)
                audio = torch.cat(audio_shifted, dim=0)
        return audio
    
    def apply_filtering(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random filtering effects"""
        # High-pass filtering
        if random.random() < 0.3:  # 30% probability
            hp_filter = random.choice(self.highpass_filters)
            audio = hp_filter(audio)
        
        # Low-pass filtering  
        if random.random() < 0.2:  # 20% probability
            lp_filter = random.choice(self.lowpass_filters)
            audio = lp_filter(audio)
        
        # Parametric EQ
        if random.random() < self.eq_prob:
            eq_filter = random.choice(self.eq_filters)
            # Apply with random gain
            eq_gain = random.uniform(0.5, 1.5)
            audio_eq = eq_filter(audio)
            audio = audio + eq_gain * (audio_eq - audio)
        
        return audio
    
    def apply_time_effects(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply time-based effects (tempo, reverb, etc.)"""
        # Simple reverb simulation (early reflections)
        if random.random() < self.reverb_prob:
            delay_samples = random.randint(1000, 4000)  # 20-90ms at 44.1kHz
            decay = random.uniform(0.1, 0.3)
            
            if delay_samples < audio.shape[-1]:
                delayed = torch.zeros_like(audio)
                delayed[..., delay_samples:] = audio[..., :-delay_samples] * decay
                audio = audio + delayed
        
        return audio
    
    def apply_noise_augmentation(self, audio: torch.Tensor) -> torch.Tensor:
        """Add background noise for robustness"""
        if random.random() < 0.4:  # 40% probability
            noise_level = random.uniform(0, self.noise_level)
            noise = torch.randn_like(audio) * noise_level
            audio = audio + noise
        return audio
    
    def apply_dynamics(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply dynamic range effects"""
        # Simple soft clipping/saturation
        if random.random() < 0.3:
            saturation = random.uniform(1.0, 2.0)
            audio = torch.tanh(audio * saturation) / saturation
        
        # Random stereo width adjustment
        if audio.shape[0] == 2 and random.random() < 0.4:
            width = random.uniform(0.7, 1.3)
            mid = (audio[0] + audio[1]) / 2
            side = (audio[0] - audio[1]) / 2 * width
            audio[0] = mid + side
            audio[1] = mid - side
        
        return audio
    
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply mix-scale augmentation pipeline
        
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
        
        # Apply augmentations in sequence
        audio = self.apply_gain_augmentation(audio)
        audio = self.apply_filtering(audio)
        audio = self.apply_time_effects(audio)
        audio = self.apply_noise_augmentation(audio)
        audio = self.apply_pitch_shift(audio)  # Expensive, do last
        audio = self.apply_dynamics(audio)
        
        # Normalize to prevent clipping
        max_val = torch.abs(audio).max()
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)
        
        return audio


# ==================== Enhanced Training Configuration ====================

class EnhancedDCAEConfig:
    """Enhanced configuration with EMA and augmentation settings"""
    
    def __init__(self):
        # Model architecture (same as before)
        self.latent_channels = 8
        self.encoder_base_channels = 64
        self.decoder_base_channels = 64
        self.use_vector_quantization = False
        self.dual_channel_processing = True
        self.use_weight_norm = True  # T-2: Weight Normalization
        
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
        
        # T-1: Augmentation Configuration
        self.use_augmentation = True
        self.augmentation_prob = 0.8
        self.gain_range = (-3.0, 3.0)
        self.pitch_range = (-0.5, 0.5)
        self.tempo_range = (0.9, 1.1)
        self.noise_level = 0.005
        self.reverb_prob = 0.3
        self.eq_prob = 0.4
        
        # Loss weights (A-5: Enhanced STFT loss)
        self.stft_weight = 1.0
        self.time_weight = 0.1
        self.vq_weight = 0.02
        self.adv_weight = 0.1
        
        # Scheduler
        self.scheduler_type = "cosine"
        self.min_lr = 1e-6
        self.warmup_epochs = 10
        
        # Data
        self.sample_rate = 44100
        self.max_length = 44100 * 10  # 10 seconds max
    
    def create_augmentation(self) -> Optional[MixScaleAugmentation]:
        """Create augmentation pipeline"""
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
            eq_prob=self.eq_prob
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
    """Manages training state including EMA, augmentation, and metrics"""
    
    def __init__(self, config: EnhancedDCAEConfig):
        self.config = config
        self.metrics_history = defaultdict(list)
        self.best_metrics = {}
        self.current_epoch = 0
        self.global_step = 0
        
        # Augmentation pipeline
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
        """Apply augmentation if enabled"""
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
        snr_db = 10 * torch.log10(signal_power / noise_power)
        return float(snr_db)
    else:
        return float('inf')


def compute_si_sdr(reference: torch.Tensor, estimation: torch.Tensor) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio"""
    # Zero-mean
    reference = reference - torch.mean(reference)
    estimation = estimation - torch.mean(estimation)
    
    # Scale invariant target
    alpha = torch.sum(estimation * reference) / torch.sum(reference ** 2)
    target = alpha * reference
    
    # SI-SDR
    si_sdr = 10 * torch.log10(
        torch.sum(target ** 2) / torch.sum((estimation - target) ** 2)
    )
    
    return float(si_sdr)


def analyze_frequency_response(
    original: torch.Tensor, 
    reconstructed: torch.Tensor,
    sample_rate: int = 44100
) -> Dict[str, float]:
    """Analyze frequency response quality"""
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
        'total_spectral_error': float(F.l1_loss(reconstructed_mag, original_mag))
    }