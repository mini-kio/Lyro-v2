# lyro/dcae/train_dcae.py
"""
Enhanced LYRO DCAE Training Script with Memory Optimization and Audio Length Control
Integrates chunked processing, expanded gradient checkpointing, and configurable audio duration
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import time
import librosa
import torchaudio
from typing import Dict, Optional, List
import math
import gc
import psutil
import warnings
from contextlib import nullcontext
import multiprocessing

# Multi-GPU support with Accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, DataLoaderConfiguration
from accelerate.logging import get_logger
import wandb
from tqdm.auto import tqdm

# LYRO 모듈 임포트
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enhanced modules with memory optimization
from dcae.model import (
    MemoryOptimizedLyroMusicDCAE, 
    create_memory_optimized_lyro_dcae
)
from dcae.training_utils import (
    EMAWrapper, 
    TrainingStateManager, 
    EnhancedDCAEConfig,
    compute_snr, 
    compute_si_sdr, 
    analyze_frequency_response
)
from dataset.dcae_dataset import DCAEDataset, DCAECollator

# 경고 억제
warnings.filterwarnings("ignore")


class AdvancedMemoryMonitor:
    """고성능 메모리 모니터링 with memory optimization tracking"""
    
    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.device = accelerator.device
        self._last_clear = time.time()
        self.memory_history = []
        self.peak_memory_usage = 0
        self.oom_events = 0
        
    def get_memory_stats(self):
        if torch.cuda.is_available() and self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            
            # Update peak usage
            self.peak_memory_usage = max(self.peak_memory_usage, allocated)
            
            stats = {
                'gpu_allocated_gb': allocated,
                'gpu_reserved_gb': reserved,
                'gpu_utilization': allocated / (reserved + 1e-8) * 100,
                'peak_memory_gb': self.peak_memory_usage
            }
            
            self.memory_history.append(allocated)
            if len(self.memory_history) > 100:
                self.memory_history.pop(0)
                
            return stats
        return {}
    
    def should_clear_cache(self):
        now = time.time()
        if now - self._last_clear > 120:  # 2분마다
            self._last_clear = now
            return True
        return False
    
    def intelligent_clear(self):
        """Intelligent memory management with optimization tracking"""
        gc.collect()
        
        if torch.cuda.is_available():
            # Check memory pressure
            if len(self.memory_history) > 10:
                recent_avg = np.mean(self.memory_history[-10:])
                if recent_avg > 8.0:  # High memory usage
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, 'synchronize'):
                        torch.cuda.synchronize()
    
    def log_oom_event(self):
        """Log OOM event"""
        self.oom_events += 1
    
    def get_memory_summary(self):
        """Get comprehensive memory usage summary"""
        stats = self.get_memory_stats()
        return {
            **stats,
            'oom_events': self.oom_events,
            'memory_history_length': len(self.memory_history),
            'avg_memory_usage': np.mean(self.memory_history) if self.memory_history else 0.0
        }


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for adversarial training"""
    
    def __init__(self, scales: List[int] = [1, 2, 4]):
        super().__init__()
        self.discriminators = nn.ModuleList()
        
        for scale in scales:
            disc = nn.Sequential(
                nn.Conv1d(2, 64, 15, stride=1, padding=7),
                nn.LeakyReLU(0.2),
                nn.Conv1d(64, 128, 41, stride=4, padding=20, groups=4),
                nn.LeakyReLU(0.2),
                nn.Conv1d(128, 256, 41, stride=4, padding=20, groups=16),
                nn.LeakyReLU(0.2),
                nn.Conv1d(256, 512, 41, stride=4, padding=20, groups=64),
                nn.LeakyReLU(0.2),
                nn.Conv1d(512, 1024, 41, stride=4, padding=20, groups=256),
                nn.LeakyReLU(0.2),
                nn.Conv1d(1024, 1, 3, padding=1),
            )
            self.discriminators.append(disc)
            
        # Downsampling layers
        self.downsamplers = nn.ModuleList()
        for i in range(len(scales) - 1):
            self.downsamplers.append(
                nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
            )
    
    def forward(self, x):
        results = []
        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                x = self.downsamplers[i-1](x)
            results.append(discriminator(x))
        return results


class MemoryOptimizedDCAETrainer:
    """
    Memory-Optimized Multi-GPU DCAE 학습 관리 클래스
    
    Features:
    - Chunked processing in StateSpaceKernel
    - Expanded gradient checkpointing
    - Configurable audio duration
    - Memory optimization tracking
    - Multi-GPU support with Accelerate
    """
    
    def __init__(self, args):
        self.args = args
        
        # Multi-GPU setup with optimizations
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=False,
            static_graph=True,
            bucket_cap_mb=150
        )
        
        dataloader_config = DataLoaderConfiguration(
            split_batches=True,
            use_stateful_dataloader=False
        )
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision='fp16',
            log_with="wandb" if args.use_wandb else None,
            project_dir=args.checkpoint_dir,
            kwargs_handlers=[ddp_kwargs],
            dataloader_config=dataloader_config
        )
        
        self.device = self.accelerator.device
        self.is_main_process = self.accelerator.is_main_process
        
        # Logger
        self.logger = get_logger(__name__)
        
        # Advanced monitoring with memory optimization tracking
        self.memory_monitor = AdvancedMemoryMonitor(self.accelerator)
        
        # Enhanced configuration with memory optimization
        self.config = EnhancedDCAEConfig()
        self.config.use_augmentation = False  # Force disable augmentation
        self._update_config_from_args()
        
        # Audio duration configuration
        self.audio_duration = args.audio_duration
        self.target_length = int(44100 * self.audio_duration)
        
        # Memory optimization parameters
        self.memory_chunk_size = args.chunk_size  # Renamed to avoid conflicts
        self.use_checkpointing = not args.disable_checkpointing
        self.memory_efficient = args.memory_efficient
        self.checkpointing_segments = args.checkpointing_segments
        
        # Training state manager (handles EMA, metrics)
        self.state_manager = TrainingStateManager(self.config)
        
        # Adaptive batch size with audio duration consideration
        self.current_batch_size = args.batch_size
        # Adjust minimum batch size based on audio duration
        duration_factor = max(1, self.audio_duration / 10.0)  # 10초 기준
        self.min_batch_size = max(1, int(args.batch_size / (8 * duration_factor)))
        self.oom_count = 0
        
        # Performance tracking
        self.best_metrics = {'val_loss': float('inf'), 'train_loss': float('inf')}
        
        if self.is_main_process:
            self.logger.info(f"🚀 Memory-Optimized DCAE Training")
            self.logger.info(f"⚡ Multi-GPU: {self.accelerator.num_processes}")
            self.logger.info(f"🧠 Model: SSM-based DCAE with chunked processing")
            self.logger.info(f"📈 EMA: {'Enabled' if self.config.use_ema else 'Disabled'}")
            self.logger.info(f"🔊 Audio Duration: {self.audio_duration}s ({self.target_length} samples)")
            self.logger.info(f"🧩 Chunk Size: {self.memory_chunk_size}")
            self.logger.info(f"✅ Checkpointing: {'Enabled' if self.use_checkpointing else 'Disabled'}")
            self.logger.info(f"💾 Memory Efficient: {self.memory_efficient}")
            self.logger.info(f"📊 Checkpointing Segments: {self.checkpointing_segments}")
        
        # Calculate optimal worker count
        self.num_workers = max(1, multiprocessing.cpu_count() - 2)
        
        # Initialize components
        self._initialize_models()
        self._setup_data()
        self._setup_optimization()
        self._prepare_training()
        
        # Checkpoints
        if self.is_main_process:
            self.checkpoint_dir = Path(args.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Wandb
        if self.is_main_process and args.use_wandb:
            self.accelerator.init_trackers(
                project_name="memory-optimized-dcae",
                config={
                    **vars(args),
                    'audio_duration': self.audio_duration,
                    'target_length': self.target_length,
                    'memory_chunk_size': self.memory_chunk_size,
                    'memory_optimization': {
                        'chunked_processing': True,
                        'expanded_checkpointing': self.use_checkpointing,
                        'memory_efficient': self.memory_efficient,
                        'checkpointing_segments': self.checkpointing_segments
                    }
                }
            )
        
        # Wait for all processes to be ready
        self.accelerator.wait_for_everyone()
        
        if self.is_main_process:
            self.logger.info("✅ Memory-Optimized Initialization Complete!")
    
    def _update_config_from_args(self):
        """Update config with command line arguments"""
        # Training parameters
        if hasattr(self.args, 'learning_rate') and self.args.learning_rate:
            self.config.learning_rate = self.args.learning_rate
        if hasattr(self.args, 'batch_size') and self.args.batch_size:
            self.config.batch_size = self.args.batch_size
        if hasattr(self.args, 'epochs') and self.args.epochs:
            self.config.epochs = self.args.epochs
            
        # EMA settings
        if hasattr(self.args, 'ema_decay') and self.args.ema_decay:
            self.config.ema_decay = self.args.ema_decay
        if hasattr(self.args, 'disable_ema') and self.args.disable_ema:
            self.config.use_ema = False
            
        # Augmentation disabled
        self.config.use_augmentation = False
    
    def _initialize_models(self):
        """모델 초기화 with memory optimization"""
        
        # Memory-optimized SSM-based DCAE 모델 - Clean parameter passing
        self.model = create_memory_optimized_lyro_dcae(
            model_size=getattr(self.args, 'model_size', 'base'),
            sample_rate=self.config.sample_rate,
            use_vq=self.config.use_vector_quantization,
            use_weight_norm=self.config.use_weight_norm,
            encoder_base_channels=self.config.encoder_base_channels,
            decoder_base_channels=self.config.decoder_base_channels,
            dropout=0.1,
            use_multiscale_ssm=True,
            # d_state will be set automatically based on model_size
            # Memory optimization parameters - passed directly
            chunk_size=self.memory_chunk_size,
            use_checkpointing=self.use_checkpointing,
            memory_efficient=self.memory_efficient,
            checkpointing_segments=self.checkpointing_segments,
        )
        
        # T-3: Setup EMA wrapper
        self.state_manager.setup_ema(self.model)
        
        # Discriminator for adversarial training (optional)
        if self.args.use_adversarial:
            self.discriminator = MultiScaleDiscriminator().to(self.device)
            self.adv_loss = nn.BCEWithLogitsLoss()
        else:
            self.discriminator = None
        
        if self.is_main_process:
            model_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"🧠 Memory-Optimized SSM-DCAE: {model_params:,} parameters")
            
            # Log memory optimization settings
            if hasattr(self.model, 'get_memory_stats'):
                memory_stats = self.model.get_memory_stats()
                self.logger.info(f"💾 Memory Settings: {memory_stats}")
            
            if self.discriminator:
                disc_params = sum(p.numel() for p in self.discriminator.parameters())
                self.logger.info(f"🥊 Multi-Scale Discriminator: {disc_params:,} parameters")
    
    def _setup_data(self):
        """데이터셋 및 로더 설정 with configurable audio duration"""
        if self.is_main_process:
            self.logger.info(f"📚 Setting up datasets with {self.audio_duration}s audio duration...")
            self.logger.info(f"🔧 Using {self.num_workers} workers per GPU")
        
        # Dataset configuration with configurable audio duration
        dataset_config = {
            'data_root': self.args.dataset_root,
            'sample_rate': self.config.sample_rate,
            'max_duration': self.audio_duration,  # Use configurable duration
            'min_duration': min(2.0, self.audio_duration * 0.2),  # Minimum 20% of max or 2s
            'augmentation': False,  # Explicitly disable augmentation
            'cache_audio': False,
            'skip_corrupted': True,
            'target_length': self.target_length  # Use calculated target length
        }
        
        # Wait for all processes to reach this point
        self.accelerator.wait_for_everyone()
        
        try:
            # Only main process creates dataset first (for validation)
            if self.is_main_process:
                self.logger.info("🔍 Main process validating dataset...")
                temp_dataset = DCAEDataset(**dataset_config)
                dataset_size = len(temp_dataset)
                del temp_dataset  # Free memory
                self.logger.info(f"✅ Dataset validation complete: {dataset_size} files")
            
            # Wait for main process to finish validation
            self.accelerator.wait_for_everyone()
            
            # Now all processes can create their datasets
            full_dataset = DCAEDataset(**dataset_config)
            
            # 85:15 split for better training
            total_size = len(full_dataset)
            train_size = int(total_size * 0.85)
            
            from torch.utils.data import Subset
            self.train_dataset = Subset(full_dataset, list(range(train_size)))
            self.val_dataset = Subset(full_dataset, list(range(train_size, total_size)))
            
            self._create_dataloaders()
            
            if self.is_main_process:
                self.logger.info(f"📚 Data: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")
                self.logger.info(f"🔊 Audio Duration: {self.audio_duration}s")
                self.logger.info(f"📏 Target Length: {self.target_length} samples")
                
                # Estimate memory usage per batch
                audio_memory_gb = (self.current_batch_size * 2 * self.target_length * 4) / (1024**3)
                self.logger.info(f"💾 Estimated audio memory per batch: {audio_memory_gb:.2f} GB")
                
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"❌ Data setup failed: {e}")
            raise
    
    def _create_dataloaders(self):
        """Create optimized dataloaders with audio duration consideration"""
        collator = DCAECollator(
            max_length=self.target_length,
            min_length=int(44100 * min(1.0, self.audio_duration * 0.1)),  # 10% of max or 1s
            pad_to_multiple=256
        )
        
        # Calculate workers per GPU (reduce for longer audio)
        workers_per_gpu = max(1, self.num_workers // self.accelerator.num_processes)
        if self.audio_duration > 20:  # Reduce workers for long audio
            workers_per_gpu = max(1, workers_per_gpu // 2)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.current_batch_size,
            shuffle=True,
            num_workers=workers_per_gpu,
            pin_memory=True,
            collate_fn=collator,
            drop_last=True,
            persistent_workers=True if workers_per_gpu > 0 else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.current_batch_size,
            shuffle=False,
            num_workers=max(1, workers_per_gpu // 2),
            pin_memory=True,
            collate_fn=collator,
            persistent_workers=True if workers_per_gpu > 1 else False
        )
        
        if self.is_main_process:
            self.logger.info(f"🔧 DataLoader workers: Train={workers_per_gpu}, Val={max(1, workers_per_gpu // 2)}")
    
    def _setup_optimization(self):
        """옵티마이저 및 스케줄러 설정"""
        # Fused AdamW with optimized settings
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.config.weight_decay,
            eps=1e-6,
            fused=True if torch.cuda.is_available() else False
        )
        
        if self.discriminator:
            self.disc_optimizer = optim.AdamW(
                self.discriminator.parameters(),
                lr=self.config.learning_rate * 0.5,
                betas=(0.5, 0.9),
                fused=True if torch.cuda.is_available() else False
            )
        
        # Advanced scheduling with warm restarts
        total_steps = self.config.epochs * len(self.train_loader)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_steps // 8,
            T_mult=1,
            eta_min=self.config.learning_rate * 0.001
        )
        
        if self.discriminator:
            self.disc_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.disc_optimizer,
                T_0=total_steps // 8,
                T_mult=1,
                eta_min=self.config.learning_rate * 0.0005
            )
    
    def _prepare_training(self):
        """Prepare with accelerate"""
        if self.discriminator:
            components = [
                self.model, self.discriminator,
                self.optimizer, self.disc_optimizer,
                self.train_loader, self.val_loader,
                self.scheduler, self.disc_scheduler
            ]
        else:
            components = [
                self.model, self.optimizer,
                self.train_loader, self.val_loader,
                self.scheduler
            ]
        
        prepared = self.accelerator.prepare(*components)
        
        if self.discriminator:
            (self.model, self.discriminator, self.optimizer, self.disc_optimizer,
             self.train_loader, self.val_loader, self.scheduler, self.disc_scheduler) = prepared
        else:
            (self.model, self.optimizer, self.train_loader, 
             self.val_loader, self.scheduler) = prepared
    
    def train_epoch(self, epoch):
        """한 에폭 학습 with memory optimization and chunked processing"""
        self.model.train()
        
        total_loss = 0
        total_stft_loss = 0
        total_vq_loss = 0
        total_adv_loss = 0
        successful_batches = 0
        
        # Quality metrics
        snr_scores = []
        si_sdr_scores = []
        
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        else:
            pbar = self.train_loader
        
        for batch_idx, audio in enumerate(pbar):
            try:
                # Progressive memory management
                if batch_idx % 50 == 0:  # More frequent for long audio
                    self.memory_monitor.intelligent_clear()
                
                with self.accelerator.accumulate(self.model):
                    # Mixed precision forward with memory optimization
                    with self.accelerator.autocast():
                        try:
                            # Memory-optimized forward pass with chunked processing
                            reconstructed, loss_dict = self.model(audio, return_loss=True)
                            
                            # Main reconstruction loss
                            loss = loss_dict['total_loss']
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                self.memory_monitor.log_oom_event()
                                if self._handle_oom(epoch):
                                    continue
                                else:
                                    raise
                            else:
                                raise
                    
                    # Advanced adversarial training
                    adv_loss = torch.tensor(0.0, device=audio.device)
                    if self.discriminator and batch_idx % 2 == 0:
                        try:
                            adv_loss = self._advanced_adversarial(audio, reconstructed)
                            loss += 0.05 * adv_loss
                        except Exception as e:
                            pass  # Skip if adversarial fails
                    
                    # Backward with gradient scaling
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        # Gradient clipping with adaptive threshold
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.config.grad_clip
                        )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # T-3: Update EMA
                    self.state_manager.update_ema()
                    self.state_manager.global_step += 1
                
                # Statistics
                total_loss += loss.item()
                total_stft_loss += loss_dict['stft_loss'].item()
                total_vq_loss += loss_dict.get('vq_loss', torch.tensor(0.0)).item()
                total_adv_loss += adv_loss.item()
                successful_batches += 1
                
                # Quality metrics (every 50 batches)
                if batch_idx % 50 == 0 and self.accelerator.sync_gradients:
                    try:
                        with torch.no_grad():
                            # Compute SNR and SI-SDR for first sample
                            snr = compute_snr(audio[0], reconstructed[0])
                            si_sdr = compute_si_sdr(audio[0].flatten(), reconstructed[0].flatten())
                            
                            snr_scores.append(snr)
                            si_sdr_scores.append(si_sdr)
                    except Exception:
                        pass
                
                # Progress update with memory optimization info
                if self.is_main_process and self.accelerator.sync_gradients:
                    mem_stats = self.memory_monitor.get_memory_stats()
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'stft': f'{loss_dict["stft_loss"].item():.3f}',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                        'gpu': f'{mem_stats.get("gpu_allocated_gb", 0):.1f}GB',
                        'oom': f'{self.memory_monitor.oom_events}'
                    })
                
                # Detailed logging with memory optimization metrics
                if (self.is_main_process and self.args.use_wandb and 
                    self.accelerator.sync_gradients and batch_idx % 500 == 0):
                    
                    log_dict = {
                        'train/total_loss': loss.item(),
                        'train/stft_loss': loss_dict['stft_loss'].item(),
                        'train/vq_loss': loss_dict.get('vq_loss', torch.tensor(0.0)).item(),
                        'train/time_loss': loss_dict.get('time_loss', torch.tensor(0.0)).item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/batch_size': self.current_batch_size,
                        'train/audio_duration': self.audio_duration,
                        'step': self.state_manager.global_step
                    }
                    
                    if adv_loss.item() > 0:
                        log_dict['train/adv_loss'] = adv_loss.item()
                    
                    if snr_scores:
                        log_dict['train/snr_db'] = np.mean(snr_scores[-5:])
                    if si_sdr_scores:
                        log_dict['train/si_sdr_db'] = np.mean(si_sdr_scores[-5:])
                    
                    # Memory optimization metrics
                    mem_summary = self.memory_monitor.get_memory_summary()
                    log_dict.update({f'memory/{k}': v for k, v in mem_summary.items()})
                    
                    # Model memory optimization info
                    log_dict['memory_optimization/chunk_size'] = self.memory_chunk_size
                    log_dict['memory_optimization/checkpointing_enabled'] = self.use_checkpointing
                    log_dict['memory_optimization/segments'] = self.checkpointing_segments
                    
                    self.accelerator.log(log_dict)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.memory_monitor.log_oom_event()
                    if not self._handle_oom(epoch):
                        raise
                    continue
                else:
                    raise
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"Batch {batch_idx} failed: {e}")
                continue
        
        # Epoch statistics
        avg_loss = total_loss / max(successful_batches, 1)
        avg_stft = total_stft_loss / max(successful_batches, 1)
        avg_vq = total_vq_loss / max(successful_batches, 1)
        avg_adv = total_adv_loss / max(successful_batches, 1)
        avg_snr = np.mean(snr_scores) if snr_scores else 0.0
        avg_si_sdr = np.mean(si_sdr_scores) if si_sdr_scores else 0.0
        
        return {
            'loss': avg_loss,
            'stft_loss': avg_stft,
            'vq_loss': avg_vq,
            'adv_loss': avg_adv,
            'snr': avg_snr,
            'si_sdr': avg_si_sdr
        }
    
    def _advanced_adversarial(self, real_audio, fake_audio):
        """Advanced multi-scale adversarial training"""
        # Train discriminators
        with self.accelerator.accumulate(self.discriminator):
            # Real samples
            real_scores = self.discriminator(real_audio.detach())
            real_losses = [self.adv_loss(score, torch.ones_like(score) * 0.9) 
                          for score in real_scores]
            
            # Fake samples  
            fake_scores = self.discriminator(fake_audio.detach())
            fake_losses = [self.adv_loss(score, torch.zeros_like(score) + 0.1)
                          for score in fake_scores]
            
            # Multi-scale discriminator loss
            disc_loss = sum(real_losses + fake_losses) / len(real_scores) / 2
            
            self.accelerator.backward(disc_loss)
            self.disc_optimizer.step()
            self.disc_scheduler.step()
            self.disc_optimizer.zero_grad()
        
        # Generator adversarial loss
        gen_scores = self.discriminator(fake_audio)
        gen_losses = [self.adv_loss(score, torch.ones_like(score)) 
                     for score in gen_scores]
        gen_adv_loss = sum(gen_losses) / len(gen_scores)
        
        return gen_adv_loss
    
    def _handle_oom(self, epoch):
        """Smart OOM handling with audio duration consideration"""
        self.oom_count += 1
        
        if self.current_batch_size > self.min_batch_size:
            old_bs = self.current_batch_size
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            
            if self.is_main_process:
                self.logger.warning(
                    f"💥 OOM! Reducing batch size: {old_bs} → {self.current_batch_size} "
                    f"(Audio: {self.audio_duration}s, OOM #{self.oom_count})"
                )
            
            # Clear memory before recreating dataloaders
            torch.cuda.empty_cache()
            gc.collect()
            
            self._recreate_dataloaders()
            return True
        
        # If we can't reduce batch size further, suggest reducing audio duration
        if self.is_main_process:
            self.logger.error(
                f"❌ Cannot reduce batch size further! Consider reducing --audio_duration "
                f"from {self.audio_duration}s or --chunk_size from {self.memory_chunk_size}"
            )
        
        return False
    
    def _recreate_dataloaders(self):
        """Recreate dataloaders with new batch size"""
        self._create_dataloaders()
        
        # Re-prepare with accelerate
        if self.discriminator:
            self.train_loader, self.val_loader = self.accelerator.prepare(
                self.train_loader, self.val_loader
            )
        else:
            self.train_loader, self.val_loader = self.accelerator.prepare(
                self.train_loader, self.val_loader
            )
        
        if self.is_main_process:
            self.logger.info(f"🔄 Dataloaders recreated with batch_size={self.current_batch_size}")
    
    def validate(self, epoch):
        """검증 수행 with EMA and memory optimization"""
        self.model.eval()
        
        total_loss = 0
        total_stft = 0
        total_vq = 0
        batch_count = 0
        snr_scores = []
        si_sdr_scores = []
        
        # T-3: Use EMA for validation if available
        ema_context = self.state_manager.get_ema_context()
        context_manager = ema_context if ema_context else torch.no_grad()
        
        with context_manager:
            for batch_idx, audio in enumerate(tqdm(self.val_loader, desc='Validation', 
                                                  disable=not self.is_main_process)):
                if batch_idx >= 20:  # Limit validation batches
                    break
                
                try:
                    with self.accelerator.autocast():
                        reconstructed, loss_dict = self.model(audio, return_loss=True)
                    
                    total_loss += loss_dict['total_loss'].item()
                    total_stft += loss_dict['stft_loss'].item()
                    total_vq += loss_dict.get('vq_loss', torch.tensor(0.0)).item()
                    batch_count += 1
                    
                    # Quality metrics
                    if batch_idx < 5:
                        try:
                            snr = compute_snr(audio[0], reconstructed[0])
                            si_sdr = compute_si_sdr(audio[0].flatten(), reconstructed[0].flatten())
                            snr_scores.append(snr)
                            si_sdr_scores.append(si_sdr)
                        except Exception:
                            pass
                
                except Exception as e:
                    if self.is_main_process:
                        self.logger.warning(f"Val batch {batch_idx} failed: {e}")
                    continue
        
        # Averages
        metrics = {
            'loss': total_loss / max(batch_count, 1),
            'stft_loss': total_stft / max(batch_count, 1),
            'vq_loss': total_vq / max(batch_count, 1),
            'snr': np.mean(snr_scores) if snr_scores else 0.0,
            'si_sdr': np.mean(si_sdr_scores) if si_sdr_scores else 0.0
        }
        
        # Logging with memory optimization info
        if self.is_main_process and self.args.use_wandb:
            log_dict = {
                'val/loss': metrics['loss'],
                'val/stft_loss': metrics['stft_loss'],
                'val/vq_loss': metrics['vq_loss'],
                'val/snr_db': metrics['snr'],
                'val/si_sdr_db': metrics['si_sdr'],
                'val/audio_duration': self.audio_duration,
                'epoch': epoch
            }
            
            # Add memory summary
            mem_summary = self.memory_monitor.get_memory_summary()
            log_dict.update({f'val_memory/{k}': v for k, v in mem_summary.items()})
            
            self.accelerator.log(log_dict)
        
        return metrics
    
    def generate_samples(self, epoch, num_samples=4):
        """검증 중 샘플 생성 with EMA and memory optimization"""
        if not self.is_main_process:
            return
            
        self.model.eval()
        
        # Use EMA for sample generation if available
        ema_context = self.state_manager.get_ema_context()
        context_manager = ema_context if ema_context else torch.no_grad()
        
        with context_manager:
            try:
                # Take a few samples from validation set
                val_batch = next(iter(self.val_loader))
                audio = val_batch[:num_samples]
                
                # Generate reconstructions
                reconstructed, _ = self.model(audio, return_loss=True)
                
                # Save samples
                if self.args.save_samples:
                    sample_dir = self.checkpoint_dir / f'samples_epoch_{epoch}'
                    sample_dir.mkdir(exist_ok=True)
                    
                    for i in range(num_samples):
                        # Original
                        original_path = sample_dir / f'original_{i}.wav'
                        torchaudio.save(
                            original_path,
                            audio[i].cpu(),
                            sample_rate=self.config.sample_rate
                        )
                        
                        # Reconstructed
                        recon_path = sample_dir / f'reconstructed_{i}.wav'
                        torchaudio.save(
                            recon_path,
                            reconstructed[i].cpu(),
                            sample_rate=self.config.sample_rate
                        )
                        
                        # Wandb audio logging
                        if self.args.use_wandb:
                            self.accelerator.log({
                                f'samples/original_{i}': wandb.Audio(
                                    audio[i].cpu().numpy(),
                                    sample_rate=self.config.sample_rate,
                                    caption=f'Original {i} ({self.audio_duration}s)'
                                ),
                                f'samples/reconstructed_{i}': wandb.Audio(
                                    reconstructed[i].cpu().numpy(),
                                    sample_rate=self.config.sample_rate,
                                    caption=f'Reconstructed {i} (Epoch {epoch}, {self.audio_duration}s)'
                                )
                            })
                    
                    if self.is_main_process:
                        self.logger.info(f"💾 Samples saved to {sample_dir}")
                    
            except Exception as e:
                if self.is_main_process:
                    self.logger.warning(f"Sample generation error: {e}")
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save checkpoint with memory optimization info"""
        if not self.is_main_process:
            return
        
        # Save accelerate state
        save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}'
        self.accelerator.save_state(str(save_path))
        
        # Enhanced metadata with memory optimization info
        metadata = {
            'epoch': epoch,
            'metrics': metrics,
            'batch_size': self.current_batch_size,
            'oom_count': self.oom_count,
            'best_metrics': self.best_metrics,
            'config': self.config.__dict__,
            'training_state': self.state_manager.state_dict(),
            'audio_duration': self.audio_duration,
            'target_length': self.target_length,
            'memory_optimization': {
                'chunk_size': self.memory_chunk_size,
                'use_checkpointing': self.use_checkpointing,
                'memory_efficient': self.memory_efficient,
                'checkpointing_segments': self.checkpointing_segments,
                'chunked_processing_enabled': True,
                'expanded_checkpointing_enabled': True
            },
            'memory_stats': self.memory_monitor.get_memory_summary()
        }
        
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model'
            self.accelerator.save_state(str(best_path))
            with open(best_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # T-3: Save EMA model separately for easy inference
            if self.state_manager.ema_wrapper:
                ema_model_state = {}
                for name, param in self.model.named_parameters():
                    if name in self.state_manager.ema_wrapper.shadow:
                        ema_model_state[name] = self.state_manager.ema_wrapper.shadow[name]
                
                ema_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': ema_model_state,
                    'config': self.config.__dict__,
                    'ema_decay': self.config.ema_decay,
                    'audio_duration': self.audio_duration,
                    'memory_optimization': metadata['memory_optimization']
                }
                
                ema_path = self.checkpoint_dir / 'best_model_ema.pt'
                torch.save(ema_checkpoint, ema_path)
                self.logger.info(f"EMA model saved to {ema_path}")
        
        # Cleanup old checkpoints (keep 3)
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*'))
        if len(checkpoints) > 3:
            for ckpt in checkpoints[:-3]:
                import shutil
                shutil.rmtree(ckpt, ignore_errors=True)
        
        self.logger.info(f"💾 Checkpoint saved: epoch {epoch}")
    
    def train(self):
        """전체 학습 루프 with memory optimization"""
        if self.is_main_process:
            self.logger.info(f"\n🚀 Memory-Optimized DCAE Training Started")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"🧠 Model: SSM-based DCAE with chunked processing")
            self.logger.info(f"📈 EMA: {'Enabled' if self.config.use_ema else 'Disabled'}")
            self.logger.info(f"🔊 Audio Duration: {self.audio_duration}s")
            self.logger.info(f"🧩 Chunk Size: {self.memory_chunk_size}")
            self.logger.info(f"✅ Checkpointing: {'Enabled' if self.use_checkpointing else 'Disabled'}")
            self.logger.info(f"💾 Memory Efficient: {self.memory_efficient}")
            self.logger.info(f"⚡ Multi-GPU: {self.accelerator.num_processes}")
            self.logger.info(f"🎯 Mixed Precision: fp16")
            self.logger.info(f"{'='*80}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            if self.is_main_process:
                self.logger.info(f"\n📅 Epoch {epoch+1}/{self.config.epochs}")
            
            # Memory management
            self.memory_monitor.intelligent_clear()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            if self.is_main_process:
                mem_stats = self.memory_monitor.get_memory_stats()
                self.logger.info(
                    f"🔥 Train - Loss: {train_metrics['loss']:.4f}, "
                    f"STFT: {train_metrics['stft_loss']:.4f}, "
                    f"VQ: {train_metrics['vq_loss']:.4f}, "
                    f"SNR: {train_metrics['snr']:.2f} dB, "
                    f"GPU: {mem_stats.get('gpu_allocated_gb', 0):.1f}GB"
                )
            
            # Validation (every 2 epochs)
            if epoch % 2 == 0:
                val_metrics = self.validate(epoch)
                
                if self.is_main_process:
                    self.logger.info(
                        f"✅ Val - Loss: {val_metrics['loss']:.4f}, "
                        f"STFT: {val_metrics['stft_loss']:.4f}, "
                        f"SNR: {val_metrics['snr']:.2f} dB"
                    )
                
                # Best model tracking
                is_best = val_metrics['loss'] < self.best_metrics['val_loss']
                if is_best:
                    self.best_metrics['val_loss'] = val_metrics['loss']
                    if self.is_main_process:
                        self.logger.info(f"🏆 New best model! Val loss: {val_metrics['loss']:.4f}")
            else:
                val_metrics = {}
                is_best = False
            
            # Sample generation
            if epoch % 10 == 0:
                self.generate_samples(epoch)
            
            # Checkpoint saving
            if epoch % 5 == 0 or is_best or epoch == self.config.epochs - 1:
                all_metrics = {**train_metrics, **val_metrics}
                self.save_checkpoint(epoch, all_metrics, is_best)
            
            # Timing and memory summary
            if self.is_main_process:
                epoch_time = time.time() - epoch_start
                mem_summary = self.memory_monitor.get_memory_summary()
                self.logger.info(
                    f"⏱️  Epoch: {epoch_time/60:.1f}m, "
                    f"Peak GPU: {mem_summary.get('peak_memory_gb', 0):.1f}GB, "
                    f"OOM Events: {mem_summary.get('oom_events', 0)}"
                )
        
        # Training completion
        if self.is_main_process:
            total_time = (time.time() - start_time) / 3600
            final_mem_summary = self.memory_monitor.get_memory_summary()
            
            self.logger.info(f"\n🎉 Memory-Optimized Training Completed!")
            self.logger.info(f"⏱️  Total Time: {total_time:.2f} hours")
            self.logger.info(f"🏆 Best Val Loss: {self.best_metrics['val_loss']:.4f}")
            self.logger.info(f"💥 Total OOM Events: {final_mem_summary.get('oom_events', 0)}")
            self.logger.info(f"💾 Peak Memory Usage: {final_mem_summary.get('peak_memory_gb', 0):.2f} GB")
            self.logger.info(f"🔊 Audio Duration: {self.audio_duration}s")
            self.logger.info(f"🧩 Final Chunk Size: {self.memory_chunk_size}")
            
            # Final save
            final_metrics = {
                'training_completed': True, 
                'total_hours': total_time,
                'final_memory_summary': final_mem_summary
            }
            self.save_checkpoint(self.config.epochs - 1, final_metrics, is_best=False)
            
            if self.args.use_wandb:
                self.accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description='Memory-Optimized LYRO DCAE Training with Chunked Processing')
    
    # 데이터 관련
    parser.add_argument('--dataset_root', type=str, default='dataset-dcae/datasets/raw',
                        help='Dataset root directory')
    
    # 모델 관련
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large'],
                        help='Model size')
    parser.add_argument('--sample_rate', type=int, default=44100,
                        help='Audio sample rate')
    parser.add_argument('--latent_channels', type=int, default=8,
                        help='Number of latent channels')
    
    # 오디오 길이 설정 (NEW)
    parser.add_argument('--audio_duration', type=float, default=10.0,
                        help='Audio duration in seconds for training (default: 10.0)')
    
    # 메모리 최적화 설정 (NEW)
    parser.add_argument('--chunk_size', type=int, default=1024,
                        help='Chunk size for StateSpaceKernel processing (default: 1024)')
    parser.add_argument('--disable_checkpointing', action='store_true',
                        help='Disable gradient checkpointing (saves computation but uses more memory)')
    parser.add_argument('--memory_efficient', action='store_true', default=True,
                        help='Enable memory efficient processing')
    parser.add_argument('--checkpointing_segments', type=int, default=4,
                        help='Number of segments for checkpointing very long sequences (default: 4)')
    
    # 학습 관련
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (will be adjusted based on audio duration and memory)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    
    # T-3: EMA 관련
    parser.add_argument('--disable_ema', action='store_true',
                        help='Disable EMA')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay rate')
    
    # Adversarial 관련
    parser.add_argument('--use_adversarial', action='store_true',
                        help='Use adversarial training')
    
    # 체크포인트 및 로깅
    parser.add_argument('--checkpoint_dir', type=str, default='dcae/checkpoints_memory_optimized',
                        help='Checkpoint directory')
    parser.add_argument('--exp_name', type=str, default='memory_optimized_dcae',
                        help='Experiment name')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--save_samples', action='store_true',
                        help='Save audio samples during training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.audio_duration <= 0:
        print("❌ Audio duration must be positive!")
        return
    
    if args.chunk_size <= 0:
        print("❌ Chunk size must be positive!")
        return
    
    # Create temporary accelerator to check if this is main process
    from accelerate import Accelerator
    temp_accelerator = Accelerator()
    is_main = temp_accelerator.is_main_process
    
    if not torch.cuda.is_available():
        if is_main:
            print("❌ CUDA required for multi-GPU training!")
        return
    
    if is_main:
        print(f"🚀 Memory-Optimized DCAE Training")
        print(f"⚡ Available GPUs: {torch.cuda.device_count()}")
        print(f"🔧 CPU Workers: {max(1, multiprocessing.cpu_count() - 2)}")
        print(f"🔊 Audio Duration: {args.audio_duration}s")
        print(f"🧩 Chunk Size: {args.chunk_size}")
        print(f"✅ Checkpointing: {'Disabled' if args.disable_checkpointing else 'Enabled'}")
        print(f"💾 Memory Efficient: {args.memory_efficient}")
        print(f"📊 Checkpointing Segments: {args.checkpointing_segments}")
    
    try:
        trainer = MemoryOptimizedDCAETrainer(args)
        trainer.train()
        if trainer.is_main_process:
            print("🎉 Memory-optimized training completed successfully!")
        
    except Exception as e:
        if is_main:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()