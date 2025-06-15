# lyro/dcae/train_dcae.py - Performance Optimized Version
"""
DDP-Compatible S6-SSM Compression LYRO DCAE Training Script
OPTIMIZED: Removed debug prints for maximum performance
FIXED: NaN loss issues + model size adjustment + deadlock prevention
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
from typing import Dict, Optional, List, Any, Union
import math
import gc
import psutil
import warnings
from contextlib import nullcontext
import multiprocessing
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import shutil
import traceback
import signal
import functools


import torch._dynamo
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True
torch._dynamo.reset()


os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_DISABLE_COMPILE'] = '1'

# Multi-GPU support with Accelerate - FORCE DDP ONLY
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed, DataLoaderConfiguration
from accelerate.logging import get_logger
try:
    import wandb
except ImportError:
    wandb = None
from tqdm import tqdm

# LYRO modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# S6-SSM Compression Optimized modules
from dcae.model import (
    S6SSMCompressionOptimizedDCAE,
    create_s6_ssm_compression_optimized_dcae
)
from dcae.training_utils import (
    DDPCompatibleEMAWrapper,
    StaticTrainingStateManager,
    S6SSMCompressionConfig,
    compute_compression_aware_snr,
    compute_si_sdr,
    analyze_compression_efficiency
)
from dataset.dcae_dataset import DCAEDataset, DCAECollator, create_s6_ssm_compression_datasets

warnings.filterwarnings("ignore")
os.environ.setdefault('WANDB_DIR', '/tmp/wandb')


def safe_wandb_value(value: Any) -> Any:
    """Safely convert values for wandb logging"""
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
            mean_val = value.mean().item()
            if math.isnan(mean_val) or math.isinf(mean_val):
                return 0.0
            return float(mean_val)
    
    if isinstance(value, (np.integer, np.floating)):
        val = float(value.item())
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    
    return 0.0


def check_tensor_validity(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Fast tensor validity check"""
    if tensor is None:
        return False
    return not (torch.isnan(tensor).any() or torch.isinf(tensor).any())


def aggressive_memory_cleanup():
    """Aggressive memory cleanup for V100 16GB"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def timeout_handler(signum, frame):
    """Timeout handler for DDP initialization"""
    raise TimeoutError("DDP initialization timeout")


class V100OptimizedMemoryMonitor:
    """V100 16GB optimized memory monitor"""
    
    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.device = accelerator.device
        self.peak_memory_usage = 0
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get V100 memory statistics"""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            self.peak_memory_usage = max(self.peak_memory_usage, allocated)
            return {
                'gpu_allocated_gb': allocated,
                'peak_memory_gb': self.peak_memory_usage,
                'memory_utilization': allocated / 16.0,
            }
        return {'gpu_allocated_gb': 0.0, 'peak_memory_gb': 0.0, 'memory_utilization': 0.0}
    
    def intelligent_cleanup(self):
        """V100 optimized memory cleanup"""
        aggressive_memory_cleanup()


class DDPCompatibleQualityAnalyzer:
    """DDP compatible quality analyzer with minimal overhead"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._analysis_count = 0
    
    def analyze_compression_quality(
        self, 
        original: torch.Tensor, 
        reconstructed: torch.Tensor,
        compression_info: Dict = None
    ) -> Dict[str, float]:
        """Fast compression quality analysis"""
        
        # Only analyze every 200th call to minimize overhead
        self._analysis_count += 1
        if self._analysis_count % 200 != 0:
            return {
                'snr_db': 0.0,
                'compression_ratio': 1.0,
                'compression_quality_score': 0.0
            }
        
        try:
            if not check_tensor_validity(original) or not check_tensor_validity(reconstructed):
                return {
                    'snr_db': 0.0,
                    'compression_ratio': 1.0,
                    'compression_quality_score': 0.0
                }
            
            snr = compute_compression_aware_snr(original, reconstructed)
            compression_ratio = 1.5
            
            if compression_info and 'compression_ratio' in compression_info:
                compression_ratio = float(compression_info['compression_ratio'])
            
            quality_score = snr / max(1.0, compression_ratio) * 10
            if math.isnan(quality_score) or math.isinf(quality_score):
                quality_score = 0.0
            
            return {
                'snr_db': safe_wandb_value(snr),
                'compression_ratio': safe_wandb_value(compression_ratio),
                'compression_quality_score': safe_wandb_value(quality_score)
            }
            
        except Exception:
            return {
                'snr_db': 0.0,
                'compression_ratio': 1.0,
                'compression_quality_score': 0.0
            }


class DDPCompatibleS6SSMTrainer:
    """
    DDP Compatible S6-SSM Compression Trainer - Performance Optimized
    """
    
    def __init__(self, args):
        self.args = args
        self.global_step = 0
        
        # S6-SSM Compression Configuration
        self.config = S6SSMCompressionConfig()
        self._update_config_from_args()
        
        # Audio configuration
        self.audio_duration = args.audio_duration
        self.target_length = int(args.sample_rate * self.audio_duration)
        
        # Compression optimization parameters
        self.compression_level = getattr(args, 'compression_level', 'high')
        self.enable_all_optimizations = getattr(args, 'enable_all_optimizations', True)
        

        self._initialize_ddp_safely()
        
        # Initialize components in stages to prevent deadlock
        self._initialize_models_staged()
        self._setup_data_v100_optimized()
        self._setup_optimization()
        self._prepare_training_staged()
        
        # Checkpoints
        if self.is_main_process:
            self.checkpoint_dir = Path(args.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # DDP compatible wandb setup
        if self.is_main_process and args.use_wandb:
            self._setup_wandb_safely()
        
        if self.is_main_process:
            num_processes = self.accelerator.num_processes
            if num_processes > 1:
                self.logger.info(f"DDP: Using per-device batch_size={self.batch_size} across {num_processes} processes")
            self.logger.info("S6-SSM Compression Training Ready!")
    
    def _initialize_ddp_safely(self):
        """DDP initialization"""
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=True,
            static_graph=False,
            bucket_cap_mb=25,
            gradient_as_bucket_view=True
        )
        
        dataloader_config = DataLoaderConfiguration(
            split_batches=False,
            use_stateful_dataloader=False,
            dispatch_batches=False
        )
        

        os.environ['ACCELERATE_USE_FSDP'] = 'false'
        os.environ['ACCELERATE_USE_DEEPSPEED'] = 'false'
        os.environ['FSDP_AUTO_WRAP_POLICY'] = 'DISABLE'
        
        try:
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(300)
            
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                mixed_precision='fp16',
                log_with="wandb" if self.args.use_wandb else None,
                project_dir=self.args.checkpoint_dir,
                kwargs_handlers=[ddp_kwargs],
                dataloader_config=dataloader_config,
                cpu=False,
                device_placement=True,
                fsdp_plugin=None,
                deepspeed_plugin=None
            )
            
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                
        except TimeoutError:
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                mixed_precision='fp16',
                cpu=False,
                device_placement=True
            )
        except Exception as e:
            raise
        
        self.device = self.accelerator.device
        self.is_main_process = self.accelerator.is_main_process
        

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        torch._dynamo.config.disable = True
    
        # Logger setup
        self.logger = get_logger(__name__)
        
        # V100 optimized monitoring
        self.memory_monitor = V100OptimizedMemoryMonitor(self.accelerator)
        self.quality_analyzer = DDPCompatibleQualityAnalyzer(self.args.sample_rate)
        
        # DDP compatible state manager
        self.state_manager = StaticTrainingStateManager(self.config)
        
        # Fixed batch size for DDP compatibility
        self.batch_size = self.args.batch_size
        
        # Minimal metrics tracking for DDP
        self.best_metrics = {
            'val_loss': float('inf'),
            'compression_quality_score': 0.0
        }
        
        # Performance tracking optimized for V100
        self.epoch_times = deque(maxlen=3)
        self.batch_times = deque(maxlen=20)

        # Wait for all processes after accelerator initialization
        self.accelerator.wait_for_everyone()
        
        if self.is_main_process:
            self.logger.info("DDP Accelerator initialized successfully")
    
    def _update_config_from_args(self):
        """Update configuration from command line arguments"""
        if hasattr(self.args, 'learning_rate') and self.args.learning_rate:
            self.config.learning_rate = self.args.learning_rate
        if hasattr(self.args, 'batch_size') and self.args.batch_size:
            self.config.batch_size = self.args.batch_size
        if hasattr(self.args, 'epochs') and self.args.epochs:
            self.config.epochs = self.args.epochs
        if hasattr(self.args, 'sample_rate'):
            self.config.sample_rate = self.args.sample_rate
        
        # Model size configurations
        if hasattr(self.args, 'model_size'):
            if self.args.model_size == 'base':
                # Base: 60M parameters
                self.config.encoder_base_channels = 80
                self.config.decoder_base_channels = 80
                self.config.latent_channels = 12
                self.config.d_state = 40
                self.config.cqt_projection_dims = 80
            elif self.args.model_size == 'large':
                # Large: 100M parameters
                self.config.encoder_base_channels = 112
                self.config.decoder_base_channels = 112
                self.config.latent_channels = 16
                self.config.d_state = 56
                self.config.cqt_projection_dims = 96
        
        # Compression-specific arguments
        if hasattr(self.args, 'compression_level'):
            compression_configs = {
                'low': {'skip_pruning_ratio': 0.2, 'adaptive_latent_channels': (5, 7)},
                'medium': {'skip_pruning_ratio': 0.4, 'adaptive_latent_channels': (4, 6)},
                'high': {'skip_pruning_ratio': 0.6, 'adaptive_latent_channels': (3, 5)}
            }
            if self.args.compression_level in compression_configs:
                config_updates = compression_configs[self.args.compression_level]
                for key, value in config_updates.items():
                    setattr(self.config, key, value)
    
    def _initialize_models_staged(self):
        """Staged model initialization"""
        
        if self.is_main_process:
            self.logger.info("Starting model initialization...")
        
        try:
            aggressive_memory_cleanup()
            
            model_params = {
                'model_size': getattr(self.args, 'model_size', 'base'),
                'sample_rate': self.config.sample_rate,
                'compression_level': self.compression_level,
                'enable_all_optimizations': self.enable_all_optimizations,
                'audio_duration': self.audio_duration,
                'latent_channels': self.config.latent_channels,
                'ddp_compatible': True,
                'static_parameters': True,
                'disable_progressive_unfreezing': True,
                'encoder_base_channels': self.config.encoder_base_channels,
                'decoder_base_channels': self.config.decoder_base_channels,
                'd_state': self.config.d_state,
                'cqt_projection_dims': self.config.cqt_projection_dims,
                'enable_enhanced_numerical_stability': True,
                'gradient_checkpointing': False,
                'use_safe_operations': True
            }
            
            self.model = create_s6_ssm_compression_optimized_dcae(**model_params)
            self.model = self.model.to(self.device)
            
            if self.is_main_process:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                self.logger.info(f"Model Created: {total_params:,} params ({trainable_params:,} trainable)")
            
            self.accelerator.wait_for_everyone()
            
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"Model creation failed: {e}")
            raise
        
        try:
            self.state_manager.setup_static_ema(self.model)
            if self.is_main_process:
                self.logger.info("EMA wrapper setup completed")
                
        except Exception as e:
            if self.is_main_process:
                self.logger.warning(f"EMA setup failed, continuing without EMA: {e}")
            
        self.accelerator.wait_for_everyone()
        
        if self.is_main_process:
            self.logger.info("Model initialization completed")
    
    def _setup_data_v100_optimized(self):
        """Setup V100 optimized datasets"""
        if self.is_main_process:
            self.logger.info("Setting up datasets...")
        
        self.num_workers = 0
        
        self.accelerator.wait_for_everyone()
        try:
            self.train_dataset, self.val_dataset, self.test_dataset = create_s6_ssm_compression_datasets(
                data_root=self.args.dataset_root,
                sample_rate=self.config.sample_rate,
                max_duration=self.audio_duration,
                target_length=self.target_length,
                fast_mode=getattr(self.args, 'fast_mode', True),
                skip_validation=getattr(self.args, 'skip_validation', True),
                use_cached_list=getattr(self.args, 'use_cached_list', True),
                min_duration=min(0.5, self.audio_duration * 0.1),
                augmentation=False,
                cache_audio=False,
                skip_corrupted=True,
                auto_delete_corrupted=True,
                train_split=self.config.train_split
            )
            
            self._create_v100_optimized_dataloaders()
            
            if self.is_main_process:
                self.logger.info(f"Data: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")
                
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"Data setup failed: {e}")
            raise
    
    def _create_v100_optimized_dataloaders(self):
        """Create V100 memory optimized dataloaders"""
        collator = DCAECollator(
            max_length=self.target_length,
            min_length=int(self.config.sample_rate * 0.5),
            pad_to_multiple=256,
            filter_corrupted=True
        )
        
        self.train_loader = DataLoader(
             self.train_dataset,
             batch_size=self.batch_size,
             shuffle=True,
             num_workers=self.num_workers,
             pin_memory=False,
             collate_fn=collator,
             drop_last=True,
             persistent_workers=False,
             prefetch_factor=2 if self.num_workers > 0 else None,
         )
        
        self.val_loader = DataLoader(
             self.val_dataset,
             batch_size=self.batch_size,
             shuffle=False,
             num_workers=0,
             pin_memory=False,
             collate_fn=collator,
             drop_last=False,
             persistent_workers=False,
             prefetch_factor=None,
         )
        
        if self.is_main_process:
            self.logger.info("DataLoaders created")
    
    def _setup_optimization(self):
        """Setup DDP compatible optimization"""
        use_fused = torch.cuda.is_available()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-6,
            weight_decay=self.config.weight_decay,
            fused=use_fused,
            foreach=not use_fused
        )
        
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=1024.0,
            growth_factor=1.2,
            backoff_factor=0.8,
            growth_interval=100
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=60,
            T_mult=1,
            eta_min=5e-7,
            last_epoch=-1
        )
        
        warmup_steps = 5 * len(self.train_loader)
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        self.current_warmup_step = 0
        self.warmup_steps = warmup_steps
        
        self.nan_count = 0
        self.max_nan_count = 5
    
    def _prepare_training_staged(self):
        """Prepare training with accelerate"""
        
        if self.is_main_process:
            self.logger.info("Starting training preparation...")
        
        try:
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            self.accelerator.wait_for_everyone()
            
            if self.is_main_process:
                self.logger.info("Model and optimizer prepared")
                
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"Model/optimizer preparation failed: {e}")
            raise
        
        try:
            self.train_loader, self.val_loader = self.accelerator.prepare(
                self.train_loader, self.val_loader
            )
            self.accelerator.wait_for_everyone()
            
            if self.is_main_process:
                self.logger.info("DataLoaders prepared")
                
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"DataLoader preparation failed: {e}")
            raise
        
        try:
            self.scheduler = self.accelerator.prepare(self.scheduler)
            self.accelerator.wait_for_everyone()
            
            if self.is_main_process:
                self.logger.info("Scheduler prepared")
                
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"Scheduler preparation failed: {e}")
            raise
        
        self.accelerator.wait_for_everyone()
        
        if self.is_main_process:
            self.logger.info("Training preparation completed")
    
    def _setup_wandb_safely(self):
        """Setup W&B with error handling"""
        try:
            self.accelerator.init_trackers(
                project_name="s6-ssm-ddp-compatible-dcae",
                config={
                    'model_type': 'S6-SSM DDP Compatible DCAE',
                    'compression_level': self.compression_level,
                    'audio_duration': self.audio_duration,
                    'target_length': self.target_length,
                    'batch_size': self.args.batch_size,
                    'learning_rate': self.args.learning_rate,
                    'epochs': self.args.epochs,
                    'ddp_compatible': True,
                    'progressive_unfreezing_disabled': True,
                    'static_parameters': True,
                    'v100_optimized': True,
                    'performance_optimized': True,
                    'model_size': getattr(self.args, 'model_size', 'base'),
                    'target_parameters': f"{60 if getattr(self.args, 'model_size', 'base') == 'base' else 100}M"
                }
            )
            
            if wandb.run is not None:
                wandb.define_metric("global_step")
                wandb.define_metric("train/*", step_metric="global_step")
                wandb.define_metric("val/*",   step_metric="global_step")
                
            self.logger.info("W&B initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Wandb initialization failed: {e}")
    
    def train_epoch(self, epoch):
        """Performance optimized training epoch"""
        expected_batches = len(self.train_loader)
        self.forward_counter = 0
        
        self.model.train()
        
        epoch_start_time = time.time()
        
        total_loss = 0
        successful_batches = 0
        
        compression_metrics = []
        batch_times = []
        
        if self.is_main_process:
            pbar = tqdm(
                self.train_loader, 
                desc=f'S6-SSM Epoch {epoch}',
                ncols=80,
                disable=False
            )
        else:
            pbar = self.train_loader
        
        self.accelerator.wait_for_everyone()
        
        try:
            for batch_idx, batch in enumerate(pbar):
                batch_start_time = time.time()
                
                try:
                    # Memory cleanup less frequently
                    if batch_idx % 10 == 0:
                        self.memory_monitor.intelligent_cleanup()
                    
                    if isinstance(batch, dict):
                        audio = batch.get('audio', None)
                        if audio is None:
                            continue
                        audio = audio.to(self.device, non_blocking=True)
                    else:
                        if batch is None:
                            continue
                        audio = batch.to(self.device, non_blocking=True)
                    
                    # Quick validity check without verbose logging
                    if audio is None or audio.numel() == 0:
                        continue
                    if audio.dim() == 0 or (audio.dim() >= 1 and audio.size(0) == 0):
                        continue
                    
                    # Fast NaN check
                    if not check_tensor_validity(audio):
                        continue
                    
                    with self.accelerator.accumulate(self.model):
                        with self.accelerator.autocast():
                            reconstructed, loss_dict = self.model(audio, return_loss=True)
                            self.forward_counter += 1
                        
                        loss = loss_dict['total_loss']
                        
                        # Fast NaN detection
                        if not check_tensor_validity(loss):
                            self.nan_count += 1
                            if self.nan_count >= self.max_nan_count:
                                raise RuntimeError("Too many NaN losses detected!")
                            self.optimizer.zero_grad(set_to_none=True)
                            continue
                        
                        self.accelerator.backward(loss)
                        
                        # Fast gradient checking
                        has_nan_grad = False
                        for p in self.model.parameters():
                            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                                has_nan_grad = True
                                break
                        
                        if has_nan_grad:
                            self.optimizer.zero_grad(set_to_none=True)
                            continue
                        
                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=self.config.grad_clip
                            )
                            
                            if math.isnan(grad_norm) or math.isinf(grad_norm):
                                self.optimizer.zero_grad(set_to_none=True)
                                continue
                        
                        self.optimizer.step()
                        
                        # Learning rate scheduling
                        if self.current_warmup_step < self.warmup_steps:
                            self.warmup_scheduler.step()
                            self.current_warmup_step += 1
                        else:
                            self.scheduler.step()
                        
                        self.optimizer.zero_grad(set_to_none=True)
                        
                        # Cleanup
                        del audio, reconstructed
                        if batch_idx % 10 == 0:
                            aggressive_memory_cleanup()
                    
                    # Performance tracking
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    self.batch_times.append(batch_time)
                    
                    # Statistics collection
                    total_loss += loss.item()
                    successful_batches += 1
                    
                    # Reduced quality analysis frequency
                    if batch_idx % 100 == 0 and self.accelerator.sync_gradients:
                        try:
                            compression_ratio = 1.5
                            quality_score = total_loss / max(successful_batches, 1)
                            compression_metrics.append({
                                'compression_ratio': compression_ratio,
                                'quality_score': quality_score
                            })
                        except Exception:
                            pass
                    
                    # Reduced progress update frequency
                    if self.is_main_process and self.accelerator.sync_gradients and batch_idx % 5 == 0:
                        avg_batch_time = np.mean(batch_times[-5:]) if batch_times else 0
                        
                        pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                            'time': f'{avg_batch_time*1000:.0f}ms'
                        })
                    
                    # Reduced W&B logging frequency
                    if self.is_main_process and self.args.use_wandb and self.accelerator.sync_gradients and batch_idx % 10 == 0:
                        self.global_step += 1
                        log_dict = {
                            'train/loss': loss.item(),
                            'train/lr': self.optimizer.param_groups[0]['lr']
                        }
                        try:
                            self.accelerator.log(log_dict, step=self.global_step)
                        except Exception:
                            pass
                    
                    # Break early if requested
                    if getattr(self.args, 'max_batches_per_epoch', None) is not None and batch_idx >= self.args.max_batches_per_epoch:
                        break
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        if self.is_main_process:
                            self.logger.error(f"OOM on V100! Reduce batch size or audio duration")
                        raise e
                    else:
                        continue
                except Exception:
                    continue
                    
        except KeyboardInterrupt:
            if self.is_main_process:
                self.logger.info("Training interrupted by user")
            raise
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"Training epoch failed: {e}")
            raise
        
        # Epoch statistics
        epoch_time = time.time() - epoch_start_time
        self.epoch_times.append(epoch_time)
        
        avg_loss = total_loss / max(successful_batches, 1)
        
        return {
            'loss': avg_loss,
            'epoch_time': epoch_time,
            'avg_batch_time': np.mean(batch_times) if batch_times else 0,
            'successful_batches': successful_batches,
            'total_batches': len(self.train_loader),
            'compression_metrics_count': len(compression_metrics),
            'nan_count': self.nan_count
        }
    
    def validate(self, epoch):
        """Fast validation"""
        self.model.eval()
        
        total_loss = 0
        batch_count = 0
        
        ema_context = self.state_manager.get_static_ema_context()
        context_manager = ema_context if ema_context else torch.no_grad()
        
        val_start_time = time.time()
        with context_manager:
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation', 
                                                  disable=not self.is_main_process)):
                if batch_idx >= 6:  # Limited validation
                    break
                
                try:
                    if isinstance(batch, dict):
                        audio = batch['audio'].to(self.device, non_blocking=True)
                    else:
                        audio = batch.to(self.device, non_blocking=True)
                    
                    if not check_tensor_validity(audio):
                        continue
                    
                    with self.accelerator.autocast():
                        reconstructed, loss_dict = self.model(audio, return_loss=True)
                    
                    loss = loss_dict['total_loss']
                    
                    if check_tensor_validity(loss):
                        total_loss += loss.item()
                        batch_count += 1
                    
                    del audio, reconstructed
                    if batch_idx % 2 == 0:
                        aggressive_memory_cleanup()
                
                except Exception:
                    continue
        
        val_time = time.time() - val_start_time
        
        metrics = {
            'loss': total_loss / max(batch_count, 1),
            'validation_time': val_time
        }
        
        if self.is_main_process and self.args.use_wandb:
            try:
                self.accelerator.log({'val/loss': metrics['loss']}, step=self.global_step)
            except Exception:
                pass
        
        return metrics
    
    def generate_samples(self, epoch, num_samples=2):
        """Generate samples with minimal logging"""
        if not self.is_main_process:
            return
            
        self.model.eval()
        
        ema_context = self.state_manager.get_static_ema_context()
        context_manager = ema_context if ema_context else torch.no_grad()
        
        generation_start_time = time.time()
        
        with context_manager:
            try:
                val_batch = next(iter(self.val_loader))
                if isinstance(val_batch, dict):
                    audio_batch = val_batch['audio']
                else:
                    audio_batch = val_batch
                
                if audio_batch.shape[0] < num_samples:
                    num_samples = audio_batch.shape[0]
                
                audio = audio_batch[:num_samples]
                
                if not check_tensor_validity(audio):
                    return
                
                reconstructed, _ = self.model(audio, return_loss=True)
                
                generation_time = time.time() - generation_start_time
                
                if self.args.save_samples:
                    sample_dir = self.checkpoint_dir / f'samples_epoch_{epoch}'
                    sample_dir.mkdir(exist_ok=True)
                    
                    for i in range(num_samples):
                        original_sample = audio[i] if audio.dim() >= 2 else audio
                        recon_sample = reconstructed[i] if reconstructed.dim() >= 2 else reconstructed
                        
                        original_path = sample_dir / f'original_{i}.wav'
                        torchaudio.save(original_path, original_sample.cpu(), sample_rate=self.config.sample_rate)
                        
                        recon_path = sample_dir / f'reconstructed_{i}.wav'
                        torchaudio.save(recon_path, recon_sample.cpu(), sample_rate=self.config.sample_rate)
                    
                    self.logger.info(f"Samples generated in {generation_time:.2f}s")
                
                del audio, reconstructed
                aggressive_memory_cleanup()
                
            except Exception as e:
                self.logger.error(f"Sample generation failed: {e}")
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save checkpoint"""
        if not self.is_main_process:
            return
        
        save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}'
        self.accelerator.save_state(str(save_path))
        
        try:
            metadata = {
                'epoch': epoch,
                'metrics': {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                'batch_size': self.batch_size,
                'audio_duration': self.audio_duration,
                'compression_level': self.compression_level,
                'ddp_compatible': True,
                'performance_optimized': True,
                'model_size': getattr(self.args, 'model_size', 'base'),
                'nan_count': self.nan_count,
                'total_parameters': sum(p.numel() for p in self.model.parameters())
            }
            
            with open(save_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model'
            self.accelerator.save_state(str(best_path))
        
        # Cleanup old checkpoints
        try:
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*'))
            if len(checkpoints) > 2:
                for ckpt in checkpoints[:-2]:
                    shutil.rmtree(ckpt, ignore_errors=True)
        except Exception:
            pass
        
        self.logger.info(f"Checkpoint saved: epoch {epoch}")
    
    def train(self):
        """Complete training loop - performance optimized"""
        if self.is_main_process:
            self.logger.info("S6-SSM Compression DCAE Training Started")
            self.logger.info(f"Model: S6-SSM DDP Compatible")
            self.logger.info(f"Multi-GPU: {self.accelerator.num_processes} (DDP)")
            self.logger.info(f"Audio Duration: {self.audio_duration}s")
            self.logger.info(f"Batch Size: {self.batch_size}")
            self.logger.info(f"Model Size: {getattr(self.args, 'model_size', 'base')}")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config.epochs):
                epoch_start = time.time()
                
                if self.is_main_process:
                    self.logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
                
                if epoch % 2 == 0:
                    self.memory_monitor.intelligent_cleanup()
                
                train_metrics = self.train_epoch(epoch)
                
                if self.is_main_process:
                    mem_stats = self.memory_monitor.get_memory_stats()
                    
                    self.logger.info(
                        f"Train - Loss: {train_metrics['loss']:.4f}, "
                        f"Time: {train_metrics['epoch_time']/60:.1f}m, "
                        f"Memory: {mem_stats.get('gpu_allocated_gb', 0):.1f}GB"
                    )
                
                # Validation every 5 epochs
                if epoch % 5 == 0:
                    val_metrics = self.validate(epoch)
                    
                    if self.is_main_process:
                        self.logger.info(
                            f"Val - Loss: {val_metrics['loss']:.4f}, "
                            f"Time: {val_metrics['validation_time']:.1f}s"
                        )
                    
                    is_best = val_metrics['loss'] < self.best_metrics['val_loss']
                    
                    if is_best:
                        self.best_metrics.update({
                            'val_loss': val_metrics['loss']
                        })
                        if self.is_main_process:
                            self.logger.info(f"New best model! Val loss: {val_metrics['loss']:.4f}")
                else:
                    val_metrics = {}
                    is_best = False
                
                # Sample generation every 20 epochs
                if epoch % 20 == 0:
                    self.generate_samples(epoch)
                
                # Checkpoint saving
                if epoch % 20 == 0 or is_best or epoch == self.config.epochs - 1:
                    all_metrics = {**train_metrics, **val_metrics}
                    self.save_checkpoint(epoch, all_metrics, is_best)
                
                if self.is_main_process:
                    epoch_time = time.time() - epoch_start
                    self.logger.info(f"Epoch: {epoch_time/60:.1f}m")
                    
        except KeyboardInterrupt:
            if self.is_main_process:
                self.logger.info("Training interrupted by user")
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"Training failed: {e}")
            raise
        
        if self.is_main_process:
            total_time = (time.time() - start_time) / 3600
            final_mem_stats = self.memory_monitor.get_memory_stats()
            
            self.logger.info("S6-SSM Training Completed!")
            self.logger.info(f"Total Time: {total_time:.2f} hours")
            self.logger.info(f"Best Val Loss: {self.best_metrics['val_loss']:.4f}")
            self.logger.info(f"Peak Memory: {final_mem_stats.get('peak_memory_gb', 0):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description='Performance Optimized S6-SSM Compression DCAE Training')
    
    # Data related
    parser.add_argument('--dataset_root', type=str, default='dataset-dcae/datasets/raw',
                        help='Dataset root directory')
    
    # Model related
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large', 'compressed'],
                        help='Model size (base=60M, large=100M)')
    parser.add_argument('--sample_rate', type=int, default=44100,
                        help='Audio sample rate')
    parser.add_argument('--latent_channels', type=int, default=12,
                        help='Number of latent channels')
    
    # Audio length settings
    parser.add_argument('--audio_duration', type=float, default=1.0,
                        help='Audio duration in seconds')
    
    # S6-SSM Compression optimization settings
    parser.add_argument('--compression_level', type=str, default='high',
                        choices=['low', 'medium', 'high'],
                        help='Compression optimization level')
    parser.add_argument('--enable_all_optimizations', action='store_true', default=True,
                        help='Enable all optimizations')
    
    # Training related
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=6,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1.2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='Weight decay')
    
    parser.add_argument('--s6_core_freeze_epochs', type=int, default=0,
                        help='S6 core freeze epochs (DISABLED)')
    
    # Performance optimization - DISABLED
    parser.add_argument('--use_torch_compile', action='store_true', default=False,
                        help='Enable torch.compile() (DISABLED)')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='Torch compile mode (not used)')
    
    # Checkpoints and logging
    parser.add_argument('--checkpoint_dir', type=str, default='dcae/checkpoints_s6_ssm_optimized',
                        help='Checkpoint directory')
    parser.add_argument('--exp_name', type=str, default='s6_ssm_optimized_dcae',
                        help='Experiment name')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--save_samples', action='store_true',
                        help='Save audio samples during training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    # Fast mode arguments
    parser.add_argument('--fast_mode', action='store_true', default=True,
                        help='Fast mode for quick startup')
    parser.add_argument('--skip_validation', action='store_true', default=True,
                        help='Skip file validation')
    parser.add_argument('--use_cached_list', action='store_true', default=True,
                        help='Use cached file lists')
    
    # DEBUG arguments
    parser.add_argument('--max_batches_per_epoch', type=int, default=None,
                        help='Maximum batches per epoch for testing')
    
    args = parser.parse_args()
    
    # Override problematic settings
    if args.s6_core_freeze_epochs > 0:
        args.s6_core_freeze_epochs = 0
    
    if args.use_torch_compile:
        args.use_torch_compile = False
    
    # Validation
    if args.audio_duration <= 0:
        print("Audio duration must be positive!")
        return
    
    if not torch.cuda.is_available():
        print("CUDA required for S6-SSM compression optimization!")
        return
    
    # V100 optimization adjustments
    if args.compression_level == 'high':
        args.batch_size = max(1, args.batch_size)
        args.gradient_accumulation_steps = max(4, args.gradient_accumulation_steps)
    
    # Create accelerator for main process check
    from accelerate import Accelerator
    
    temp_accelerator = Accelerator(mixed_precision='fp16')
    is_main = temp_accelerator.is_main_process
    
    if is_main:
        print(f"Performance Optimized S6-SSM Compression DCAE Training")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"Compression Level: {args.compression_level}")
        print(f"Model Size: {args.model_size}")
        print(f"Audio Duration: {args.audio_duration}s")
        print(f"DDP Batch Size: {args.batch_size}")
        print(f"Performance Optimized: Enabled")
    
    try:
        trainer = DDPCompatibleS6SSMTrainer(args)
        trainer.train()
        
        if trainer.is_main_process:
            if args.use_wandb:
                try:
                    wandb.finish()
                except:
                    pass
            print("Performance Optimized S6-SSM Training completed successfully!")
            
    except Exception as e:
        print(f"Training failed: {e}")
        print(traceback.format_exc())


if __name__ == '__main__':
    main()