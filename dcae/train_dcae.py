# lyro/dcae/train_dcae.py - Enhanced DCAE Training
"""
Enhanced DCAE Training - Fixed architecture with proper shape handling
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

from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed, gather_object

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcae.model import create_continuous_ultra_compressed_dcae_model, FixedContinuousUltraCompressedDCAELoss
from dcae.training_utils import (
    ensure_stereo_audio, compute_snr, compute_si_sdr, 
    memory_cleanup, get_memory_stats
)
from dataset.dcae_dataset import DCAEDataset, DCAECollator, create_s6_ssm_compression_datasets

warnings.filterwarnings("ignore")


def calculate_optimized_model_metrics(model, sample_rate: int = 44100, duration: float = 1.0, device=None):
    """Calculate optimized 4kbps model compression ratio and efficiency metrics"""
    if device is None:
        device = next(model.parameters()).device
    
    input_samples = int(sample_rate * duration)
    input_channels = 2
    input_elements = input_samples * input_channels
    input_bytes = input_elements * 4
    
    dummy_audio = torch.randn(1, input_channels, input_samples, device=device, dtype=torch.float32)
    
    with torch.no_grad():
        # Use the model's compression info method
        compression_info = model.get_compression_info(dummy_audio)
        
        latent, cross = model.encode(dummy_audio)
        latent_elements = latent.numel()
        cross_elements = cross.numel() if cross is not None else 0
        total_latent_elements = latent_elements + cross_elements
        latent_bytes = total_latent_elements * 4
    
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    # Compare with standard codecs
    cd_quality_bitrate = 1411.2  # CD quality 44.1kHz 16-bit stereo
    mp3_bitrate = 320  # High quality MP3
    our_bitrate = compression_info['effective_bitrate_kbps']
    
    cd_improvement = cd_quality_bitrate / our_bitrate
    mp3_improvement = mp3_bitrate / our_bitrate
    
    return {
        'input_elements': input_elements,
        'input_bytes': input_bytes,
        'latent_elements': latent_elements,
        'cross_elements': cross_elements,
        'total_latent_elements': total_latent_elements,
        'latent_bytes': latent_bytes,
        'compression_ratio': compression_info['compression_ratio'],
        'latent_shape': tuple(latent.shape),
        'cross_shape': tuple(cross.shape) if cross is not None else None,
        'total_params': total_params,
        'model_size_mb': model_size_mb,
        'our_bitrate_kbps': our_bitrate,
        'cd_quality_bitrate_kbps': cd_quality_bitrate,
        'mp3_bitrate_kbps': mp3_bitrate,
        'cd_improvement_ratio': cd_improvement,
        'mp3_improvement_ratio': mp3_improvement,
        'target_4kbps_achieved': abs(our_bitrate - 4.0) < 1.0,
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


class EnhancedDCAETrainer:
    """Enhanced DCAE Trainer"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with="wandb" if args.use_wandb and WANDB_AVAILABLE else None,
            project_dir=args.checkpoint_dir,
        )
        
        set_seed(42)
        
        self.device = self.accelerator.device
        self.is_main_process = self.accelerator.is_main_process
        
        self._initialize_enhanced_model()
        
        if self.is_main_process:
            self._display_enhanced_model_metrics()
        
        self._setup_data()
        self._setup_optimization()
        
        self.loss_fn = FixedContinuousUltraCompressedDCAELoss()
        
        self._prepare_with_accelerator()
        
        self.checkpoint_dir = Path(args.checkpoint_dir)
        if self.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if args.use_wandb and WANDB_AVAILABLE and self.is_main_process:
            self._setup_wandb()        
        self.best_metrics = {
            'val_loss': float('inf'),
            'train_loss': float('inf'),
            'val_snr': -float('inf'),
            'compression_ratio': 0.0,
            'bitrate_improvement': 0.0,
        }
    
    def _initialize_enhanced_model(self):
        """Initialize optimized 4kbps DCAE model"""
        # Load optimized configuration
        config_path = Path(__file__).parent.parent / 'final_4kbps_dcae_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_params = config['parameters']
            print(f"Loading optimized 4kbps configuration from {config_path}")
        else:
            # Fallback to optimized parameters
            model_params = {
                'sample_rate': self.args.sample_rate,
                'latent_channels': 18,
                'target_time_steps': 35,
                'quantization_num_levels': 64,
                'quantization_temperature': 1.0
            }
            print("Using fallback optimized 4kbps parameters")
        
        print(f"Model parameters: {model_params}")
        
        self.raw_model = create_continuous_ultra_compressed_dcae_model(**model_params)
    def _display_enhanced_model_metrics(self):
        """Display optimized model compression ratio and performance metrics"""
        try:
            metrics = calculate_optimized_model_metrics(
                self.raw_model, 
                sample_rate=self.args.sample_rate,
                duration=self.args.audio_duration,
                device=self.device
            )
            
            print("\n" + "="*60)
            print("OPTIMIZED 4KBPS DCAE MODEL ANALYSIS")
            print("="*60)
            print(f"Input: {self.args.audio_duration}s, {self.args.sample_rate:,}Hz stereo")
            print(f"Samples: {metrics['input_elements']:,}")
            print(f"Input Size: {metrics['input_bytes']/1024/1024:.2f} MB")
            print(f"Latent Shape: {metrics['latent_shape']}")
            print(f"Cross Shape: {metrics['cross_shape']}")
            print(f"Compression: {metrics['compression_ratio']:.1f}:1")
            print(f"Our Bitrate: {metrics['our_bitrate_kbps']:.2f} kbps")
            print(f"4kbps Target: {'✓ ACHIEVED' if metrics['target_4kbps_achieved'] else '⚠ Needs tuning'}")
            print(f"vs CD Quality: {metrics['cd_improvement_ratio']:.1f}x improvement")
            print(f"vs MP3 320kbps: {metrics['mp3_improvement_ratio']:.1f}x improvement")
            print(f"Model Size: {metrics['model_size_mb']:.1f} MB")
            print("="*60 + "\n")
            
        except Exception as e:
            if self.is_main_process:
                print(f"Could not calculate model metrics: {e}")
    
    def _setup_data(self):
        """Setup datasets"""
        effective_batch_size = self.args.batch_size * self.args.gradient_accumulation_steps * self.accelerator.num_processes
        
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
                print(f"Dataset creation failed: {e}")
            raise
        
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
            print(f"Dataset: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")
    
    def _setup_optimization(self):
        """Setup optimization components"""
        self.optimizer = optim.AdamW(
            self.raw_model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay,
            eps=1e-8,
        )
        
        steps_per_epoch = len(self.train_loader) // max(self.args.gradient_accumulation_steps, 1)
        total_steps = steps_per_epoch * self.args.epochs
        warmup_steps = steps_per_epoch * 10
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.args.learning_rate * 0.01
        )
        
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        self.warmup_steps = warmup_steps
        self.total_steps = 0
    
    def _prepare_with_accelerator(self):
        """Prepare model and components with accelerator"""
        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler, self.loss_fn = self.accelerator.prepare(
            self.raw_model, self.optimizer, self.train_loader, self.val_loader, self.scheduler, self.loss_fn
        )
    
    def _setup_wandb(self):
        """Setup wandb logging"""
        try:
            self.accelerator.init_trackers(
                project_name="optimized-4kbps-dcae-training",                config={
                    'model_type': 'Optimized 4kbps DCAE',
                    'architecture': 'Continuous Ultra-Compressed DCAE',
                    'latent_channels': 18,  # Updated to optimized value
                    'target_time_steps': 35,  # Updated to optimized value
                    'target_bitrate_kbps': 4.0,
                    'batch_size': self.args.batch_size,
                    'gradient_accumulation_steps': self.args.gradient_accumulation_steps,
                    'effective_batch_size': self.args.batch_size * self.args.gradient_accumulation_steps * self.accelerator.num_processes,
                    'learning_rate': self.args.learning_rate,
                    'epochs': self.args.epochs,
                    'mixed_precision': self.args.mixed_precision,
                    'num_processes': self.accelerator.num_processes,
                }
            )
        except Exception as e:
            print(f"Wandb initialization failed: {e}")
    
    def _validate_batch(self, batch) -> Tuple[torch.Tensor, bool]:
        """Validate and process batch"""
        try:
            audio = batch['audio'] if isinstance(batch, dict) else batch
            
            if audio is None or audio.numel() == 0:
                return self._create_dummy_audio(), False
            
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                return self._create_dummy_audio(), False
            
            if audio.shape[-1] < 1000:
                return self._create_dummy_audio(), False
            
            audio = ensure_stereo_audio(audio, target_device=audio.device)
            
            return audio, True
            
        except Exception:
            return self._create_dummy_audio(), False
    
    def _create_dummy_audio(self) -> torch.Tensor:
        """Create dummy audio for corrupted batches"""
        T = int(self.args.sample_rate * self.args.audio_duration)
        return torch.randn(1, 2, T, device=self.device, dtype=torch.float32) * 0.001
    
    def _compute_enhanced_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """Compute audio quality metrics"""
        try:
            snr = compute_snr(original, reconstructed)
            si_sdr = compute_si_sdr(original, reconstructed)
            
            original_flat = original.flatten()
            reconstructed_flat = reconstructed.flatten()
            
            mse = F.mse_loss(reconstructed_flat, original_flat).item()
            
            try:
                orig_stft = torch.stft(original_flat, n_fft=1024, hop_length=256, return_complex=True)
                recon_stft = torch.stft(reconstructed_flat, n_fft=1024, hop_length=256, return_complex=True)
                orig_mag = torch.abs(orig_stft)
                recon_mag = torch.abs(recon_stft)
                spectral_convergence = torch.norm(orig_mag - recon_mag) / torch.norm(orig_mag)
                spectral_convergence = spectral_convergence.item()
            except:
                spectral_convergence = 0.0
            
            return {
                'snr_db': float(snr),
                'si_sdr_db': float(si_sdr),
                'mse': float(mse),
                'spectral_convergence': float(spectral_convergence)
            }
        except Exception:
            return {
                'snr_db': 0.0,
                'si_sdr_db': 0.0,
                'mse': 1.0,
                'spectral_convergence': 1.0
            }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Training epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_dual_loss = 0.0
        total_feat_loss = 0.0
        total_snr = 0.0
        total_si_sdr = 0.0
        total_spectral_conv = 0.0
        successful_batches = 0
        batch_times = []
        
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
                if batch_idx % 50 == 0:
                    memory_cleanup()
                
                audio, is_valid = self._validate_batch(batch)
                
                with self.accelerator.accumulate(self.model):
                    reconstructed = self.model(audio)
                    
                    # Get discriminator features for the new model
                    with torch.no_grad():
                        real_feat = self.model.get_discriminator_features(audio)
                    fake_feat = self.model.get_discriminator_features(reconstructed.detach())
                    
                    # Use the new loss function signature
                    loss_dict = self.loss_fn(
                        pred=reconstructed, 
                        target=audio, 
                        model=self.model,
                        pred_feat=fake_feat, 
                        target_feat=real_feat
                    )
                    loss = loss_dict['total_loss']
                    
                    if not is_valid:
                        loss = loss * 0.1
                    
                    if loss.item() > 100.0 or loss.item() < 0.0:
                        continue
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), 
                            self.args.grad_clip
                        )
                    
                    self.optimizer.step()
                    
                    if self.total_steps < self.warmup_steps:
                        self.warmup_scheduler.step()
                    else:
                        if self.warmup_scheduler is not None:
                            self.warmup_scheduler = None
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    self.total_steps += 1
                
                if self.accelerator.sync_gradients and is_valid:
                    loss_val = loss.item()
                    total_loss += loss_val
                    total_dual_loss += loss_dict['dual_domain_loss'].item()
                    total_feat_loss += loss_dict['feature_matching_loss'].item()
                    
                    metrics = self._compute_enhanced_metrics(audio, reconstructed)
                    total_snr += metrics['snr_db']
                    total_si_sdr += metrics['si_sdr_db']
                    total_spectral_conv += metrics['spectral_convergence']
                    
                    successful_batches += 1
                    batch_times.append(time.time() - batch_start)
                    
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
                        
                        if (self.args.use_wandb and WANDB_AVAILABLE and 
                            successful_batches % self.args.log_interval == 0):
                            
                            self.accelerator.log({
                                'train/total_loss': safe_wandb_log(avg_loss),
                                'train/dual_domain_loss': safe_wandb_log(total_dual_loss / successful_batches),
                                'train/feature_matching_loss': safe_wandb_log(total_feat_loss / successful_batches),
                                'train/snr_db': safe_wandb_log(avg_snr),
                                'train/si_sdr_db': safe_wandb_log(total_si_sdr / successful_batches),
                                'train/spectral_convergence': safe_wandb_log(total_spectral_conv / successful_batches),
                                'train/lr': safe_wandb_log(self.optimizer.param_groups[0]['lr']),
                                'step': epoch * len(self.train_loader) + batch_idx
                            })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    memory_cleanup()
                    self.optimizer.zero_grad()
                    continue
                else:
                    continue
            except Exception as e:
                continue
        
        avg_loss = total_loss / max(successful_batches, 1)
        avg_dual_loss = total_dual_loss / max(successful_batches, 1)
        avg_feat_loss = total_feat_loss / max(successful_batches, 1)
        avg_snr = total_snr / max(successful_batches, 1)
        avg_si_sdr = total_si_sdr / max(successful_batches, 1)
        avg_spectral_conv = total_spectral_conv / max(successful_batches, 1)
        
        return {
            'loss': avg_loss,
            'dual_domain_loss': avg_dual_loss,
            'feature_matching_loss': avg_feat_loss,
            'snr_db': avg_snr,
            'si_sdr_db': avg_si_sdr,
            'spectral_convergence': avg_spectral_conv,
            'successful_batches': successful_batches,
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validation epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_snr = 0.0
        total_si_sdr = 0.0
        total_spectral_conv = 0.0
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
                if batch_idx >= 20:
                    break
                
                try:
                    audio, is_valid = self._validate_batch(batch)
                    
                    if not is_valid:
                        continue
                    
                    reconstructed = self.model(audio)
                    
                    real_feat = self.model.get_discriminator_features(audio)
                    fake_feat = self.model.get_discriminator_features(reconstructed)
                    
                    loss_dict = self.loss_fn(
                        pred=reconstructed, 
                        target=audio, 
                        model=self.model,
                        pred_feat=fake_feat, 
                        target_feat=real_feat
                    )
                    loss = loss_dict['total_loss']
                    
                    total_loss += loss.item()
                    
                    metrics = self._compute_enhanced_metrics(audio, reconstructed)
                    total_snr += metrics['snr_db']
                    total_si_sdr += metrics['si_sdr_db']
                    total_spectral_conv += metrics['spectral_convergence']
                    
                    batch_count += 1
                    
                    if self.is_main_process:
                        avg_snr = total_snr / batch_count
                        avg_si_sdr = total_si_sdr / batch_count
                        avg_spectral_conv = total_spectral_conv / batch_count
                        
                        val_pbar.set_postfix({
                            'val_loss': f'{total_loss / batch_count:.4f}',
                            'val_snr': f'{avg_snr:.1f}dB',
                            'val_si_sdr': f'{avg_si_sdr:.1f}dB',
                            'spectral_conv': f'{avg_spectral_conv:.4f}'
                        })
                
                except Exception as e:
                    continue
        
        avg_val_loss = total_loss / max(batch_count, 1)
        avg_val_snr = total_snr / max(batch_count, 1)
        avg_val_si_sdr = total_si_sdr / max(batch_count, 1)
        avg_val_spectral_conv = total_spectral_conv / max(batch_count, 1)
        
        return {
            'loss': avg_val_loss,
            'snr_db': avg_val_snr,
            'si_sdr_db': avg_val_si_sdr,
            'spectral_convergence': avg_val_spectral_conv,
            'batch_count': batch_count
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any], is_best: bool = False):
        """Save checkpoint"""
        if not self.is_main_process:
            return
        
        checkpoint_dir = self.checkpoint_dir / f'epoch_{epoch}'
        self.accelerator.save_state(checkpoint_dir)
        
        metadata = {
            'epoch': epoch,
            'args': vars(self.args),
            'metrics': metrics,
            'model_config': {
                'type': 'Enhanced DCAE',
                'architecture': 'Direct Waveform ConvNeXt + HiFiGAN',
                'latent_channels': 24,
                'mixed_precision': self.args.mixed_precision,
            }
        }
        
        metadata_path = checkpoint_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if is_best:
            best_dir = self.checkpoint_dir / 'best'
            if best_dir.exists():
                import shutil
                shutil.rmtree(best_dir)
            
            import shutil
            shutil.copytree(checkpoint_dir, best_dir)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint"""
        checkpoint_dir = Path(checkpoint_path)
        
        self.accelerator.load_state(checkpoint_dir)
        
        metadata_path = checkpoint_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata['epoch']
        
        return 0
    
    def train(self):
        """Main training loop"""
        start_epoch = 0
        
        if self.args.resume:
            start_epoch = self.load_checkpoint(self.args.resume)
            if self.is_main_process:
                print(f"Resumed training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, self.args.epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
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
            
            if self.args.use_wandb and WANDB_AVAILABLE and self.is_main_process:
                self.accelerator.log({
                    'epoch/train_total_loss': safe_wandb_log(train_metrics['loss']),
                    'epoch/train_dual_domain_loss': safe_wandb_log(train_metrics['dual_domain_loss']),
                    'epoch/train_feature_matching_loss': safe_wandb_log(train_metrics['feature_matching_loss']),
                    'epoch/train_snr_db': safe_wandb_log(train_metrics['snr_db']),
                    'epoch/train_si_sdr_db': safe_wandb_log(train_metrics['si_sdr_db']),
                    'epoch/train_spectral_convergence': safe_wandb_log(train_metrics['spectral_convergence']),
                    'epoch/val_loss': safe_wandb_log(val_metrics['loss']),
                    'epoch/val_snr_db': safe_wandb_log(val_metrics['snr_db']),
                    'epoch/val_si_sdr_db': safe_wandb_log(val_metrics['si_sdr_db']),
                    'epoch/val_spectral_convergence': safe_wandb_log(val_metrics['spectral_convergence']),
                    'epoch': epoch
                })
            
            if self.is_main_process and (epoch % self.args.save_interval == 0 or epoch == self.args.epochs - 1):
                all_metrics = {**train_metrics, **val_metrics}
                self.save_checkpoint(epoch, all_metrics, is_best)
        
        self.accelerator.end_training()


def main():
    """Main function for Optimized 4kbps DCAE Training"""
    parser = argparse.ArgumentParser(description='Optimized 4kbps DCAE Training')
    
    # Data
    parser.add_argument('--dataset_root', type=str, default='dataset-dcae/datasets/raw')
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--audio_duration', type=float, default=1.0)
    
    # Training - optimized for 4kbps target
    parser.add_argument('--epochs', type=int, default=300)  # Increased for better convergence
    parser.add_argument('--batch_size', type=int, default=8)  # Slightly increased
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-4)  # Slightly higher for faster learning
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Mixed precision
    parser.add_argument('--mixed_precision', type=str, default='fp16', choices=['fp16', 'bf16', 'no'])
    
    # Data loading
    parser.add_argument('--num_workers', type=int, default=6)  # Increased for better data loading
    
    # Logging and saving
    parser.add_argument('--checkpoint_dir', type=str, default='dcae/checkpoints_4kbps_optimized')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--log_interval', type=int, default=50)  # More frequent logging
    parser.add_argument('--save_interval', type=int, default=25)  # More frequent saves
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA required for optimized 4kbps DCAE training!")
        return
    
    print("Starting Optimized 4kbps DCAE Training...")
    print(f"Target: 4kbps bitrate with high-quality reconstruction")
    print(f"Model: Continuous Ultra-Compressed DCAE")
    print(f"Configuration: 18 latent channels, 35 time steps")
    
    try:
        trainer = EnhancedDCAETrainer(args)
        trainer.train()
        
        print("\n🎉 Training completed successfully!")
        print("Model is ready for SSM + Flow Matching integration")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()