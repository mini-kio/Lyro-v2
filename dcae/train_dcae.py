# lyro/dcae/train_dcae.py - Optimized CNN Model Training
"""
DCAE Training - Optimized CNN Model + Multi-GPU + Mixed Precision
Focus: Simplified training, efficient CNN architecture, enhanced metrics
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import time
import librosa
import torchaudio
from typing import Dict, Optional, List, Any, Union, Tuple
import math
import gc
import warnings
from tqdm import tqdm

# Accelerate for multi-GPU
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed, gather_object

# W&B handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Add project path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DCAE imports
from dcae.model import create_optimized_dcae_model
from dcae.training_utils import (
    ensure_stereo_audio, compute_snr, compute_si_sdr, 
    memory_cleanup, get_memory_stats
)
from dataset.dcae_dataset import DCAEDataset, DCAECollator, create_s6_ssm_compression_datasets

warnings.filterwarnings("ignore")


# ==================== Enhanced Loss Functions ====================

class OptimizedDCAELoss(nn.Module):
    """Optimized loss function for CNN-based DCAE"""
    
    def __init__(self):
        super().__init__()
        
        # Multi-scale STFT configurations
        self.stft_configs = [
            {"n_fft": 1024, "hop_length": 256, "weight": 1.0},
            {"n_fft": 512, "hop_length": 128, "weight": 0.7},
            {"n_fft": 2048, "hop_length": 512, "weight": 0.5},
        ]
        
        # Mel-spectrogram loss
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0
        )
        
        # Loss weights
        self.l1_weight = 1.0
        self.spectral_weight = 0.5
        self.mel_weight = 0.3
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute optimized loss with multiple components"""
        # Ensure stereo format
        pred = ensure_stereo_audio(pred, target_device=pred.device)
        target = ensure_stereo_audio(target, target_device=target.device)
        
        # Match lengths
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]
        
        # Primary L1 loss
        l1_loss = F.l1_loss(pred, target)
        
        # Multi-scale spectral loss
        spectral_loss = 0.0
        try:
            for config in self.stft_configs:
                # Process stereo channels
                pred_flat = pred.reshape(-1, pred.shape[-1])
                target_flat = target.reshape(-1, target.shape[-1])
                
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
                
                # Magnitude loss
                pred_mag = torch.abs(pred_stft)
                target_mag = torch.abs(target_stft)
                mag_loss = F.l1_loss(pred_mag, target_mag)
                
                spectral_loss += mag_loss * config["weight"]
        except Exception:
            spectral_loss = 0.0
        
        # Mel-spectrogram loss
        mel_loss = 0.0
        try:
            # Average over stereo channels
            pred_mono = pred.mean(dim=1)
            target_mono = target.mean(dim=1)
            
            pred_mel = self.mel_transform(pred_mono)
            target_mel = self.mel_transform(target_mono)
            
            mel_loss = F.l1_loss(pred_mel, target_mel)
        except Exception:
            mel_loss = 0.0
        
        # Combine losses
        total_loss = (
            self.l1_weight * l1_loss +
            self.spectral_weight * spectral_loss +
            self.mel_weight * mel_loss
        )
        
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'spectral_loss': spectral_loss,
            'mel_loss': mel_loss
        }


# ==================== Training Utilities ====================

def calculate_model_metrics(model, sample_rate: int = 44100, duration: float = 1.0, device=None):
    """Calculate model compression ratio and size"""
    if device is None:
        device = next(model.parameters()).device
    
    # Input audio size
    input_samples = int(sample_rate * duration)
    input_channels = 2  # Stereo
    input_elements = input_samples * input_channels
    input_bytes = input_elements * 4  # FP32 = 4 bytes per element
    
    # Create dummy input to get latent size
    dummy_audio = torch.randn(1, input_channels, input_samples, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        latent = model.encode(dummy_audio)
        latent_elements = latent.numel()
        latent_bytes = latent_elements * 4  # FP32 = 4 bytes per element
    
    # Calculate compression ratio
    compression_ratio = input_elements / latent_elements
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # FP32 parameters
    
    return {
        'input_elements': input_elements,
        'input_bytes': input_bytes,
        'latent_elements': latent_elements,
        'latent_bytes': latent_bytes,
        'compression_ratio': compression_ratio,
        'latent_shape': tuple(latent.shape),
        'total_params': total_params,
        'model_size_mb': model_size_mb
    }


def safe_wandb_log(value: Any) -> float:
    """Safe wandb value conversion"""
    if value is None:
        return 0.0
    
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            val = value.item()
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return float(val)
        else:
            return float(value.mean().item())
    
    return 0.0


# ==================== Main Trainer ====================

class OptimizedDCAETrainer:
    """Optimized DCAE Trainer for CNN model"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with="wandb" if args.use_wandb and WANDB_AVAILABLE else None,
            project_dir=args.checkpoint_dir,
        )
        
        # Set seed for reproducibility
        set_seed(42)
        
        self.device = self.accelerator.device
        self.is_main_process = self.accelerator.is_main_process
        
        if self.is_main_process:
            print(f"🚀 Optimized DCAE Training Started")
            print(f"⚡ Available GPUs: {torch.cuda.device_count()}")
            print(f"🎯 Mixed Precision: {args.mixed_precision}")
            print(f"🔄 Gradient Accumulation: {args.gradient_accumulation_steps}")
            print(f"📊 Processes: {self.accelerator.num_processes}")
        
        # Initialize model
        self._initialize_model()
        
        # Calculate and display model metrics
        if self.is_main_process:
            self._display_model_metrics()
        
        # Setup data
        self._setup_data()
        
        # Setup optimization
        self._setup_optimization()
        
        # Setup loss function
        self.loss_fn = OptimizedDCAELoss()
        
        # Prepare with accelerator
        self._prepare_with_accelerator()
        
        # Setup checkpoint dir
        self.checkpoint_dir = Path(args.checkpoint_dir)
        if self.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup wandb
        if args.use_wandb and WANDB_AVAILABLE and self.is_main_process:
            self._setup_wandb()
        
        # Best metrics tracking
        self.best_metrics = {
            'val_loss': float('inf'),
            'train_loss': float('inf'),
            'val_snr': -float('inf'),
        }
        
        if self.is_main_process:
            print("✅ Optimized DCAE Trainer Initialized")
            print(f"📊 Model Parameters: {sum(p.numel() for p in self.raw_model.parameters()):,}")
            print(f"💾 Multi-component Loss Function")
            print(f"⚡ CNN-based Architecture")
    
    def _initialize_model(self):
        """Initialize optimized CNN model"""
        self.raw_model = create_optimized_dcae_model(
            sample_rate=self.args.sample_rate,
            latent_channels=8,  # Optimized compression
            encoder_depths=[2, 2, 6, 2],
            encoder_dims=[96, 192, 384, 768],
        )
        
        if self.is_main_process:
            print(f"🧠 Optimized CNN DCAE Model Created")
            print(f"   - Architecture: ConvNeXt + HiFiGAN")
            print(f"   - Latent Channels: 8")
            print(f"   - No SSM components")
    
    def _display_model_metrics(self):
        """Display model compression ratio and size metrics"""
        try:
            metrics = calculate_model_metrics(
                self.raw_model, 
                sample_rate=self.args.sample_rate,
                duration=self.args.audio_duration,
                device=self.device
            )
            
            print("\n" + "="*60)
            print("📈 OPTIMIZED MODEL ANALYSIS")
            print("="*60)
            print(f"🎵 Input Audio:")
            print(f"   - Duration: {self.args.audio_duration}s")
            print(f"   - Sample Rate: {self.args.sample_rate:,} Hz")
            print(f"   - Channels: 2 (Stereo)")
            print(f"   - Total Samples: {metrics['input_elements']:,}")
            print(f"   - Input Size: {metrics['input_bytes']:,} bytes ({metrics['input_bytes']/1024/1024:.2f} MB)")
            
            print(f"\n🗜️ Latent Representation:")
            print(f"   - Latent Shape: {metrics['latent_shape']}")
            print(f"   - Latent Elements: {metrics['latent_elements']:,}")
            print(f"   - Latent Size: {metrics['latent_bytes']:,} bytes ({metrics['latent_bytes']/1024:.2f} KB)")
            
            print(f"\n⚡ Compression Metrics:")
            print(f"   - Compression Ratio: {metrics['compression_ratio']:.1f}:1")
            print(f"   - Size Reduction: {(1 - metrics['latent_bytes']/metrics['input_bytes'])*100:.1f}%")
            
            print(f"\n🧠 Model Size:")
            print(f"   - Total Parameters: {metrics['total_params']:,}")
            print(f"   - Model Size (FP32): {metrics['model_size_mb']:.1f} MB")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"⚠️ Could not calculate model metrics: {e}")
    
    def _setup_data(self):
        """Setup datasets"""
        if self.is_main_process:
            print("📚 Setting up datasets...")
        
        # Calculate effective batch size
        effective_batch_size = self.args.batch_size * self.args.gradient_accumulation_steps * self.accelerator.num_processes
        if self.is_main_process:
            print(f"🔄 Global Effective Batch Size: {effective_batch_size}")
        
        # Use existing dataset creation function
        try:
            self.train_dataset, self.val_dataset, self.test_dataset = create_s6_ssm_compression_datasets(
                data_root=self.args.dataset_root,
                sample_rate=self.args.sample_rate,
                max_duration=self.args.audio_duration,
                target_length=int(self.args.sample_rate * self.args.audio_duration),
                fast_mode=True,
                skip_validation=True,
                use_cached_list=True,
                min_duration=0.5,
                augmentation=True,
                cache_audio=True,
                skip_corrupted=True,
                auto_delete_corrupted=True,
                train_split=0.85,
                ensure_stereo=True,
                stereo_processing_mode='independent'
            )
        except Exception as e:
            if self.is_main_process:
                print(f"⚠️ Dataset creation failed: {e}")
            raise
        
        # Create dataloaders
        collator = DCAECollator(
            max_length=int(self.args.sample_rate * self.args.audio_duration),
            min_length=int(self.args.sample_rate * 0.5),
            pad_to_multiple=256,
            filter_corrupted=True,
            ensure_stereo=True
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collator,
            drop_last=True,
            persistent_workers=True if self.args.num_workers > 0 else False,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=max(1, self.args.num_workers // 2),
            pin_memory=True,
            collate_fn=collator,
            drop_last=False,
        )
        
        if self.is_main_process:
            print(f"📊 Dataset: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")
    
    def _setup_optimization(self):
        """Setup optimized training components"""
        # AdamW optimizer
        self.optimizer = optim.AdamW(
            self.raw_model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay,
            eps=1e-8,
        )
        
        # Cosine annealing scheduler
        steps_per_epoch = len(self.train_loader) // max(self.args.gradient_accumulation_steps, 1)
        total_steps = steps_per_epoch * self.args.epochs
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.args.learning_rate * 0.01
        )
        
        if self.is_main_process:
            print("⚙️ Optimization setup completed")
            print(f"   - Steps per epoch: {steps_per_epoch:,}")
            print(f"   - Total training steps: {total_steps:,}")
    
    def _prepare_with_accelerator(self):
        """Prepare model, optimizer, and dataloaders with accelerator"""
        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler, self.loss_fn = self.accelerator.prepare(
            self.raw_model, self.optimizer, self.train_loader, self.val_loader, self.scheduler, self.loss_fn
        )
        
        if self.is_main_process:
            print("✅ Model and components prepared with accelerator")
    
    def _setup_wandb(self):
        """Setup wandb logging"""
        try:
            self.accelerator.init_trackers(
                project_name="optimized-dcae-training",
                config={
                    'model_type': 'Optimized CNN DCAE',
                    'architecture': 'ConvNeXt + HiFiGAN',
                    'latent_channels': 8,
                    'batch_size': self.args.batch_size,
                    'gradient_accumulation_steps': self.args.gradient_accumulation_steps,
                    'effective_batch_size': self.args.batch_size * self.args.gradient_accumulation_steps * self.accelerator.num_processes,
                    'learning_rate': self.args.learning_rate,
                    'epochs': self.args.epochs,
                    'mixed_precision': self.args.mixed_precision,
                    'num_processes': self.accelerator.num_processes,
                    'optimized_cnn': True,
                }
            )
            print("📊 Wandb initialized")
        except Exception as e:
            print(f"⚠️ Wandb initialization failed: {e}")
    
    def _validate_batch(self, batch) -> Tuple[torch.Tensor, bool]:
        """Validate and process batch"""
        try:
            audio = batch['audio'] if isinstance(batch, dict) else batch
            
            if audio is None or audio.numel() == 0:
                return self._create_dummy_audio(), False
            
            # Basic validation
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                return self._create_dummy_audio(), False
            
            if audio.shape[-1] < 1000:  # Too short
                return self._create_dummy_audio(), False
            
            # Ensure stereo format
            audio = ensure_stereo_audio(audio, target_device=audio.device)
            
            return audio, True
            
        except Exception:
            return self._create_dummy_audio(), False
    
    def _create_dummy_audio(self) -> torch.Tensor:
        """Create dummy audio for corrupted batches"""
        T = int(self.args.sample_rate * self.args.audio_duration)
        return torch.randn(1, 2, T, device=self.device, dtype=torch.float32) * 0.001
    
    def _compute_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """Compute audio quality metrics"""
        try:
            snr = compute_snr(original, reconstructed)
            si_sdr = compute_si_sdr(original, reconstructed)
            
            return {
                'snr_db': float(snr),
                'si_sdr_db': float(si_sdr)
            }
        except Exception:
            return {
                'snr_db': 0.0,
                'si_sdr_db': 0.0
            }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Training epoch with optimized loss"""
        self.model.train()
        
        total_loss = 0.0
        total_snr = 0.0
        total_si_sdr = 0.0
        successful_batches = 0
        batch_times = []
        
        # Component losses
        total_l1_loss = 0.0
        total_spectral_loss = 0.0
        total_mel_loss = 0.0
        
        # Progress bar
        if self.is_main_process:
            pbar = tqdm(
                self.train_loader, 
                desc=f'Training Epoch {epoch}',
                dynamic_ncols=True,
                leave=False
            )
        else:
            pbar = self.train_loader
        
        for batch_idx, batch in enumerate(pbar):
            batch_start = time.time()
            
            try:
                # Memory cleanup periodically
                if batch_idx % 50 == 0:
                    memory_cleanup()
                
                # Validate batch
                audio, is_valid = self._validate_batch(batch)
                
                # Forward pass with gradient accumulation
                with self.accelerator.accumulate(self.model):
                    # Model forward pass
                    reconstructed = self.model(audio)
                    
                    # Compute loss
                    loss_dict = self.loss_fn(reconstructed, audio)
                    loss = loss_dict['total_loss']
                    
                    # Reduce loss weight for invalid batches
                    if not is_valid:
                        loss = loss * 0.1
                    
                    # Skip unreasonable losses
                    if loss.item() > 100.0 or loss.item() < 0.0:
                        continue
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping and optimizer step
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), 
                            self.args.grad_clip
                        )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Metrics tracking
                if self.accelerator.sync_gradients and is_valid:
                    loss_val = loss.item()
                    total_loss += loss_val
                    
                    # Component losses
                    total_l1_loss += loss_dict['l1_loss'].item()
                    total_spectral_loss += loss_dict['spectral_loss'].item()
                    total_mel_loss += loss_dict['mel_loss'].item()
                    
                    # Audio quality metrics
                    metrics = self._compute_metrics(audio, reconstructed)
                    total_snr += metrics['snr_db']
                    total_si_sdr += metrics['si_sdr_db']
                    
                    successful_batches += 1
                    batch_times.append(time.time() - batch_start)
                    
                    # Update progress bar
                    if self.is_main_process:
                        avg_loss = total_loss / successful_batches
                        avg_snr = total_snr / successful_batches
                        avg_time = np.mean(batch_times[-10:])
                        
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'snr': f'{avg_snr:.1f}dB',
                            'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                            'time': f'{avg_time:.2f}s'
                        })
                        
                        # Wandb logging
                        if (self.args.use_wandb and WANDB_AVAILABLE and 
                            successful_batches % self.args.log_interval == 0):
                            
                            self.accelerator.log({
                                'train/loss': safe_wandb_log(avg_loss),
                                'train/l1_loss': safe_wandb_log(total_l1_loss / successful_batches),
                                'train/spectral_loss': safe_wandb_log(total_spectral_loss / successful_batches),
                                'train/mel_loss': safe_wandb_log(total_mel_loss / successful_batches),
                                'train/snr_db': safe_wandb_log(avg_snr),
                                'train/lr': safe_wandb_log(self.optimizer.param_groups[0]['lr']),
                                'step': epoch * len(self.train_loader) + batch_idx
                            })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if self.is_main_process:
                        print(f"💥 OOM at batch {batch_idx}, clearing cache...")
                    memory_cleanup()
                    self.optimizer.zero_grad()
                    continue
                else:
                    if self.is_main_process:
                        print(f"❌ Runtime error: {e}")
                    continue
            except Exception as e:
                if self.is_main_process:
                    print(f"❌ Batch {batch_idx} failed: {e}")
                continue
        
        # Gather metrics from all processes
        avg_loss = total_loss / max(successful_batches, 1)
        avg_snr = total_snr / max(successful_batches, 1)
        avg_si_sdr = total_si_sdr / max(successful_batches, 1)
        avg_l1_loss = total_l1_loss / max(successful_batches, 1)
        avg_spectral_loss = total_spectral_loss / max(successful_batches, 1)
        avg_mel_loss = total_mel_loss / max(successful_batches, 1)
        
        # Epoch summary
        if self.is_main_process:
            print(f"📈 Training Epoch {epoch} Summary:")
            print(f"   - Total Loss: {avg_loss:.4f}")
            print(f"   - L1 Loss: {avg_l1_loss:.4f}")
            print(f"   - Spectral Loss: {avg_spectral_loss:.4f}")
            print(f"   - Mel Loss: {avg_mel_loss:.4f}")
            print(f"   - SNR: {avg_snr:.1f} dB")
            print(f"   - SI-SDR: {avg_si_sdr:.1f} dB")
            print(f"   - Successful Batches: {successful_batches}")
        
        return {
            'loss': avg_loss,
            'l1_loss': avg_l1_loss,
            'spectral_loss': avg_spectral_loss,
            'mel_loss': avg_mel_loss,
            'snr_db': avg_snr,
            'si_sdr_db': avg_si_sdr,
            'successful_batches': successful_batches,
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validation epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_snr = 0.0
        total_si_sdr = 0.0
        batch_count = 0
        
        with torch.no_grad():
            if self.is_main_process:
                val_pbar = tqdm(
                    self.val_loader, 
                    desc='Validation',
                    dynamic_ncols=True,
                    leave=False
                )
            else:
                val_pbar = self.val_loader
            
            for batch_idx, batch in enumerate(val_pbar):
                if batch_idx >= 20:  # Limit validation batches
                    break
                
                try:
                    audio, is_valid = self._validate_batch(batch)
                    
                    if not is_valid:
                        continue
                    
                    # Forward pass
                    reconstructed = self.model(audio)
                    loss_dict = self.loss_fn(reconstructed, audio)
                    loss = loss_dict['total_loss']
                    
                    total_loss += loss.item()
                    
                    # Audio quality metrics
                    metrics = self._compute_metrics(audio, reconstructed)
                    total_snr += metrics['snr_db']
                    total_si_sdr += metrics['si_sdr_db']
                    
                    batch_count += 1
                    
                    # Update progress bar
                    if self.is_main_process:
                        avg_snr = total_snr / batch_count
                        avg_si_sdr = total_si_sdr / batch_count
                        
                        val_pbar.set_postfix({
                            'val_loss': f'{total_loss / batch_count:.4f}',
                            'val_snr': f'{avg_snr:.1f}dB',
                            'val_si_sdr': f'{avg_si_sdr:.1f}dB'
                        })
                
                except Exception as e:
                    if self.is_main_process:
                        print(f"⚠️ Val batch {batch_idx} failed: {e}")
                    continue
        
        avg_val_loss = total_loss / max(batch_count, 1)
        avg_val_snr = total_snr / max(batch_count, 1)
        avg_val_si_sdr = total_si_sdr / max(batch_count, 1)
        
        if self.is_main_process:
            print(f"✅ Validation: Loss={avg_val_loss:.4f}, SNR={avg_val_snr:.1f}dB, SI-SDR={avg_val_si_sdr:.1f}dB")
        
        return {
            'loss': avg_val_loss,
            'snr_db': avg_val_snr,
            'si_sdr_db': avg_val_si_sdr,
            'batch_count': batch_count
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any], is_best: bool = False):
        """Save checkpoint"""
        if not self.is_main_process:
            return
        
        # Save using accelerator's save_state
        checkpoint_dir = self.checkpoint_dir / f'epoch_{epoch}'
        self.accelerator.save_state(checkpoint_dir)
        
        # Save additional metadata
        metadata = {
            'epoch': epoch,
            'args': vars(self.args),
            'metrics': metrics,
            'model_config': {
                'type': 'Optimized CNN DCAE',
                'architecture': 'ConvNeXt + HiFiGAN',
                'latent_channels': 8,
                'mixed_precision': self.args.mixed_precision,
                'optimized_cnn': True,
            }
        }
        
        metadata_path = checkpoint_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Best model
        if is_best:
            best_dir = self.checkpoint_dir / 'best'
            if best_dir.exists():
                import shutil
                shutil.rmtree(best_dir)
            
            import shutil
            shutil.copytree(checkpoint_dir, best_dir)
            print(f"🏆 New best model saved! SNR: {metrics.get('snr_db', 0):.1f}dB")
        
        print(f"💾 Checkpoint saved: epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint"""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load state using accelerator
        self.accelerator.load_state(checkpoint_dir)
        
        # Load metadata
        metadata_path = checkpoint_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata['epoch']
        
        return 0
    
    def train(self):
        """Main training loop"""
        start_epoch = 0
        
        # Resume from checkpoint
        if self.args.resume:
            start_epoch = self.load_checkpoint(self.args.resume)
            if self.is_main_process:
                print(f"🔄 Resumed from epoch {start_epoch}")
        
        if self.is_main_process:
            print("\n" + "="*60)
            print("🚀 OPTIMIZED DCAE CNN TRAINING STARTED")
            print("="*60)
            print(f"🎯 Mixed Precision: {self.args.mixed_precision}")
            print(f"🔄 Gradient Accumulation: {self.args.gradient_accumulation_steps}")
            print(f"⚡ Multi-GPU: {self.accelerator.num_processes} processes")
            print(f"🧠 Architecture: ConvNeXt + HiFiGAN")
            print(f"📊 Multi-component Loss Function")
            print("="*60 + "\n")
        
        # Training loop
        for epoch in range(start_epoch, self.args.epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate(epoch)
            
            # Check if best model
            is_best = False
            if self.is_main_process:
                is_best_loss = val_metrics['loss'] < self.best_metrics['val_loss']
                is_best_snr = val_metrics['snr_db'] > self.best_metrics['val_snr']
                is_best = is_best_loss or is_best_snr
                
                if is_best:
                    if is_best_loss:
                        self.best_metrics['val_loss'] = val_metrics['loss']
                    if is_best_snr:
                        self.best_metrics['val_snr'] = val_metrics['snr_db']
                    self.best_metrics['train_loss'] = train_metrics['loss']
            
            # Wandb logging
            if self.args.use_wandb and WANDB_AVAILABLE and self.is_main_process:
                self.accelerator.log({
                    'epoch/train_loss': safe_wandb_log(train_metrics['loss']),
                    'epoch/train_l1_loss': safe_wandb_log(train_metrics['l1_loss']),
                    'epoch/train_spectral_loss': safe_wandb_log(train_metrics['spectral_loss']),
                    'epoch/train_mel_loss': safe_wandb_log(train_metrics['mel_loss']),
                    'epoch/train_snr_db': safe_wandb_log(train_metrics['snr_db']),
                    'epoch/train_si_sdr_db': safe_wandb_log(train_metrics['si_sdr_db']),
                    'epoch/val_loss': safe_wandb_log(val_metrics['loss']),
                    'epoch/val_snr_db': safe_wandb_log(val_metrics['snr_db']),
                    'epoch/val_si_sdr_db': safe_wandb_log(val_metrics['si_sdr_db']),
                    'epoch': epoch
                })
            
            # Save checkpoint
            if self.is_main_process and (epoch % self.args.save_interval == 0 or epoch == self.args.epochs - 1):
                all_metrics = {**train_metrics, **val_metrics}
                self.save_checkpoint(epoch, all_metrics, is_best)
        
        if self.is_main_process:
            print("\n" + "="*60)
            print("🎉 OPTIMIZED DCAE CNN TRAINING COMPLETED!")
            print("="*60)
            print(f"🏆 Best Val Loss: {self.best_metrics['val_loss']:.4f}")
            print(f"🎵 Best Val SNR: {self.best_metrics['val_snr']:.1f} dB")
            print(f"📊 Optimized CNN architecture successfully trained!")
            print("="*60)
        
        # Clean up
        self.accelerator.end_training()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Optimized CNN DCAE Training')
    
    # Data
    parser.add_argument('--dataset_root', type=str, default='dataset-dcae/datasets/raw')
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--audio_duration', type=float, default=1.0)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Mixed precision
    parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['fp16', 'bf16', 'no'])
    
    # Data loading
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Logging and saving
    parser.add_argument('--checkpoint_dir', type=str, default='dcae/checkpoints_optimized')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # CUDA check
    if not torch.cuda.is_available():
        print("❌ CUDA required for optimized training!")
        return
    
    try:
        trainer = OptimizedDCAETrainer(args)
        trainer.train()
        print("🎉 Optimized training completed successfully!")
        print("🎯 Key improvements:")
        print("   - ❌ SSM removed for efficiency")
        print("   - ✅ CNN-based architecture")
        print("   - 📈 Multi-component loss function")
        print("   - ⚡ Enhanced compression ratio")
        print("   - 🎵 High-quality 44.1kHz stereo output")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()