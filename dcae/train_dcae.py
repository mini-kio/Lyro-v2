# lyro/dcae/train_dcae.py
"""
Enhanced LYRO DCAE Training Script with CQT-SSM and Memory Optimization
Integrates CQT, Harmonic-Percussive separation, and configurable audio duration
Now with 90% memory savings while maintaining SSM advantages
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

# Enhanced modules with CQT-SSM optimization
from dcae.model import (
    CQTSSMDCAE, 
    create_cqt_ssm_dcae
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
    """Enhanced memory monitor for CQT-SSM training"""
    
    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.device = accelerator.device
        self._last_clear = time.time()
        self.memory_history = []
        self.peak_memory_usage = 0
        self.oom_events = 0
        self.cqt_memory_savings = 0
        
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
                'peak_memory_gb': self.peak_memory_usage,
                'cqt_memory_savings_gb': self.cqt_memory_savings
            }
            
            self.memory_history.append(allocated)
            if len(self.memory_history) > 100:
                self.memory_history.pop(0)
                
            return stats
        return {}
    
    def log_cqt_savings(self, raw_audio_memory_estimate, actual_cqt_memory):
        """Log memory savings from CQT representation"""
        self.cqt_memory_savings = raw_audio_memory_estimate - actual_cqt_memory
    
    def should_clear_cache(self):
        now = time.time()
        if now - self._last_clear > 60:  # More frequent for CQT processing
            self._last_clear = now
            return True
        return False
    
    def intelligent_clear(self):
        """Enhanced memory management for CQT-SSM"""
        gc.collect()
        
        if torch.cuda.is_available():
            # Check memory pressure (lower threshold for CQT)
            if len(self.memory_history) > 5:
                recent_avg = np.mean(self.memory_history[-5:])
                if recent_avg > 6.0:  # Lower threshold due to CQT efficiency
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, 'synchronize'):
                        torch.cuda.synchronize()
    
    def log_oom_event(self):
        """Log OOM event"""
        self.oom_events += 1
    
    def get_memory_summary(self):
        """Get comprehensive memory usage summary with CQT stats"""
        stats = self.get_memory_stats()
        return {
            **stats,
            'oom_events': self.oom_events,
            'memory_history_length': len(self.memory_history),
            'avg_memory_usage': np.mean(self.memory_history) if self.memory_history else 0.0,
            'cqt_representation': 'CQT + Harmonic-Percussive',
            'memory_optimization': 'CQT-SSM Enhanced'
        }


class CQTAudioQualityAnalyzer:
    """Enhanced audio quality analyzer for CQT-based models"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    def analyze_music_quality(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """
        Comprehensive music quality analysis optimized for CQT models
        """
        metrics = {}
        
        # Convert to numpy for librosa analysis
        orig_np = original.detach().cpu().numpy()
        recon_np = reconstructed.detach().cpu().numpy()
        
        # Handle stereo - improved padding and shape handling
        if orig_np.ndim == 3:
            orig_np = orig_np[0].mean(axis=0)  # First batch, average channels
            recon_np = recon_np[0].mean(axis=0)
        elif orig_np.ndim == 2:
            orig_np = orig_np.mean(axis=0)
            recon_np = recon_np.mean(axis=0)
        
        # Ensure same length with stable padding
        min_len = min(len(orig_np), len(recon_np))
        if min_len > 0:
            orig_np = orig_np[:min_len]
            recon_np = recon_np[:min_len]
        else:
            # Fallback for empty arrays
            return self._fallback_metrics()
        
        try:
            # Standard metrics with CQT focus
            metrics['snr_db'] = compute_snr(original[0], reconstructed[0])
            metrics['si_sdr_db'] = compute_si_sdr(original[0].flatten(), reconstructed[0].flatten())
            
            # Music-specific metrics using librosa
            # Spectral centroid (brightness)
            orig_centroid = librosa.feature.spectral_centroid(y=orig_np, sr=self.sample_rate)[0]
            recon_centroid = librosa.feature.spectral_centroid(y=recon_np, sr=self.sample_rate)[0]
            metrics['spectral_centroid_error'] = np.mean(np.abs(orig_centroid - recon_centroid))
            
            # Tempo estimation accuracy
            try:
                orig_tempo = librosa.beat.tempo(y=orig_np, sr=self.sample_rate)[0]
                recon_tempo = librosa.beat.tempo(y=recon_np, sr=self.sample_rate)[0]
                metrics['tempo_error_bpm'] = abs(orig_tempo - recon_tempo)
            except:
                metrics['tempo_error_bpm'] = 0.0
            
            # Harmonic-percussive separation quality
            try:
                orig_harmonic, orig_percussive = librosa.effects.hpss(orig_np)
                recon_harmonic, recon_percussive = librosa.effects.hpss(recon_np)
                
                # Harmonic preservation
                harmonic_corr = np.corrcoef(orig_harmonic, recon_harmonic)[0, 1]
                metrics['harmonic_preservation'] = harmonic_corr if not np.isnan(harmonic_corr) else 0.0
                
                # Percussive preservation  
                percussive_corr = np.corrcoef(orig_percussive, recon_percussive)[0, 1]
                metrics['percussive_preservation'] = percussive_corr if not np.isnan(percussive_corr) else 0.0
                
            except:
                metrics['harmonic_preservation'] = 0.0
                metrics['percussive_preservation'] = 0.0
            
            # Chroma feature similarity (harmonic content)
            try:
                orig_chroma = librosa.feature.chroma_cqt(y=orig_np, sr=self.sample_rate)
                recon_chroma = librosa.feature.chroma_cqt(y=recon_np, sr=self.sample_rate)
                
                # Ensure same shape with stable handling
                min_frames = min(orig_chroma.shape[1], recon_chroma.shape[1])
                if min_frames > 0:
                    orig_chroma = orig_chroma[:, :min_frames]
                    recon_chroma = recon_chroma[:, :min_frames]
                    
                    chroma_similarity = np.mean([
                        np.corrcoef(orig_chroma[i], recon_chroma[i])[0, 1] 
                        for i in range(12) if not np.all(orig_chroma[i] == 0)
                    ])
                    metrics['chroma_similarity'] = chroma_similarity if not np.isnan(chroma_similarity) else 0.0
                else:
                    metrics['chroma_similarity'] = 0.0
                
            except:
                metrics['chroma_similarity'] = 0.0
            
        except Exception as e:
            # Enhanced fallback to basic metrics if advanced analysis fails
            return self._fallback_metrics(original, reconstructed)
        
        return metrics
    
    def _fallback_metrics(self, original=None, reconstructed=None):
        """Fallback metrics when analysis fails"""
        fallback = {
            'snr_db': 0.0,
            'si_sdr_db': 0.0,
            'spectral_centroid_error': 0.0,
            'tempo_error_bpm': 0.0,
            'harmonic_preservation': 0.0,
            'percussive_preservation': 0.0,
            'chroma_similarity': 0.0
        }
        
        # Try basic SNR if tensors provided
        if original is not None and reconstructed is not None:
            try:
                fallback['snr_db'] = compute_snr(original[0], reconstructed[0])
            except:
                pass
                
        return fallback


class CQTSSMDCAETrainer:
    """
    CQT-SSM-based Multi-GPU DCAE trainer with enhanced music understanding
    
    Features:
    - CQT representation (90% memory savings vs raw audio)
    - Harmonic-Percussive separation for music structure
    - SSM advantages maintained with spectral efficiency
    - Music-specific quality metrics
    """
    
    def __init__(self, args):
        self.args = args
        
        # Enhanced configuration - set first for stable initialization
        self.config = EnhancedDCAEConfig()
        self.config.use_augmentation = False
        self._update_config_from_args()
        
        # Audio duration configuration - early initialization
        self.audio_duration = args.audio_duration
        self.target_length = int(args.sample_rate * self.audio_duration)
        
        # Memory optimization parameters - early setup
        self.memory_chunk_size = args.chunk_size
        self.use_checkpointing = not args.disable_checkpointing
        self.memory_efficient = args.memory_efficient
        self.checkpointing_segments = args.checkpointing_segments
        
        # Multi-GPU setup with optimizations
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=False,
            static_graph=True,
            bucket_cap_mb=100  # Smaller buckets for CQT
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
        
        # Logger setup
        self.logger = get_logger(__name__)
        
        # Enhanced monitoring for CQT-SSM
        self.memory_monitor = AdvancedMemoryMonitor(self.accelerator)
        self.quality_analyzer = CQTAudioQualityAnalyzer(args.sample_rate)
        
        # Training state manager
        self.state_manager = TrainingStateManager(self.config)
        
        # Adaptive batch size (more conservative for CQT)
        self.current_batch_size = args.batch_size
        duration_factor = max(1, self.audio_duration / 10.0)
        self.min_batch_size = max(1, int(args.batch_size / (4 * duration_factor)))  # More conservative
        self.oom_count = 0
        
        # Performance tracking
        self.best_metrics = {
            'val_loss': float('inf'), 
            'train_loss': float('inf'),
            'harmonic_preservation': 0.0,
            'chroma_similarity': 0.0
        }
        
        if self.is_main_process:
            self.logger.info(f"🎵 CQT-SSM-based DCAE Training")
            self.logger.info(f"⚡ Multi-GPU: {self.accelerator.num_processes}")
            self.logger.info(f"🎼 Representation: CQT + Harmonic-Percussive")
            self.logger.info(f"🧠 Model: SSM-based with spectral efficiency")
            self.logger.info(f"📈 EMA: {'Enabled' if self.config.use_ema else 'Disabled'}")
            self.logger.info(f"🔊 Audio Duration: {self.audio_duration}s ({self.target_length} samples)")
            self.logger.info(f"🎯 Memory Savings: ~90% vs raw audio SSM")
            self.logger.info(f"✅ Checkpointing: {'Enabled' if self.use_checkpointing else 'Disabled'}")
        
        # Calculate optimal worker count
        self.num_workers = max(1, multiprocessing.cpu_count() - 2)
        
        # Initialize components in proper order
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
                project_name="cqt-ssm-dcae",
                config={
                    **vars(args),
                    'representation': 'CQT + Harmonic-Percussive',
                    'audio_duration': self.audio_duration,
                    'target_length': self.target_length,
                    'memory_optimization': 'CQT-SSM Enhanced',
                    'estimated_memory_savings': '90%'
                }
            )
        
        # Wait for all processes
        self.accelerator.wait_for_everyone()
        
        if self.is_main_process:
            self.logger.info("✅ CQT-SSM Initialization Complete!")
    
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
            
        # Sample rate
        if hasattr(self.args, 'sample_rate'):
            self.config.sample_rate = self.args.sample_rate
        
        # Augmentation disabled
        self.config.use_augmentation = False
    
    def _initialize_models(self):
        """모델 초기화 with CQT-SSM optimization"""
        
        # CQT-SSM-based DCAE 모델
        self.model = create_cqt_ssm_dcae(
            model_size=getattr(self.args, 'model_size', 'base'),
            sample_rate=self.config.sample_rate,
            use_vq=self.config.use_vector_quantization,
            use_weight_norm=self.config.use_weight_norm,
            encoder_base_channels=self.config.encoder_base_channels,
            decoder_base_channels=self.config.decoder_base_channels,
            dropout=0.1,
            use_multiscale_ssm=True,
            # CQT-specific parameters
            n_bins=84 if getattr(self.args, 'model_size', 'base') == 'base' else 72,
            hop_length=512,
            # Memory optimization parameters
            chunk_size=self.memory_chunk_size,
            use_checkpointing=self.use_checkpointing,
            memory_efficient=self.memory_efficient,
            checkpointing_segments=self.checkpointing_segments,
        )
        
        # Setup EMA wrapper
        self.state_manager.setup_ema(self.model)
        
        if self.is_main_process:
            model_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"🎼 CQT-SSM-DCAE: {model_params:,} parameters")
            
            # Log model configuration
            if hasattr(self.model, 'get_memory_stats'):
                memory_stats = self.model.get_memory_stats()
                self.logger.info(f"📊 Model Stats:")
                for key, value in memory_stats.items():
                    self.logger.info(f"  {key}: {value}")
    
    def _setup_data(self):
        """데이터셋 설정 with CQT-optimized duration"""
        if self.is_main_process:
            self.logger.info(f"📚 Setting up datasets for CQT processing...")
            self.logger.info(f"🔧 Using {self.num_workers} workers per GPU")
        
        # Dataset configuration optimized for CQT
        dataset_config = {
            'data_root': self.args.dataset_root,
            'sample_rate': self.config.sample_rate,
            'max_duration': self.audio_duration,
            'min_duration': min(1.0, self.audio_duration * 0.1),  # Shorter minimum for CQT
            'augmentation': False,
            'cache_audio': False,
            'skip_corrupted': True,
            'target_length': self.target_length
        }
        
        # Wait for all processes
        self.accelerator.wait_for_everyone()
        
        try:
            # Main process validates dataset
            if self.is_main_process:
                self.logger.info("🔍 Validating dataset for CQT processing...")
                temp_dataset = DCAEDataset(**dataset_config)
                dataset_size = len(temp_dataset)
                del temp_dataset
                self.logger.info(f"✅ Dataset validation complete: {dataset_size} files")
            
            self.accelerator.wait_for_everyone()
            
            # Create datasets
            full_dataset = DCAEDataset(**dataset_config)
            
            # 85:15 split
            total_size = len(full_dataset)
            train_size = int(total_size * 0.85)
            
            from torch.utils.data import Subset
            self.train_dataset = Subset(full_dataset, list(range(train_size)))
            self.val_dataset = Subset(full_dataset, list(range(train_size, total_size)))
            
            self._create_dataloaders()
            
            if self.is_main_process:
                self.logger.info(f"📚 Data: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")
                self.logger.info(f"🎵 Audio Duration: {self.audio_duration}s")
                
                # Estimate CQT memory usage
                cqt_frames = self.target_length // 512  # hop_length
                cqt_memory_gb = (self.current_batch_size * 84 * cqt_frames * 4) / (1024**3)
                raw_memory_gb = (self.current_batch_size * 2 * self.target_length * 4) / (1024**3)
                
                self.logger.info(f"💾 CQT memory per batch: {cqt_memory_gb:.2f} GB")
                self.logger.info(f"📊 Raw audio would be: {raw_memory_gb:.2f} GB")
                self.logger.info(f"🚀 Memory savings: {((raw_memory_gb - cqt_memory_gb) / raw_memory_gb * 100):.1f}%")
                
                # Log to memory monitor
                self.memory_monitor.log_cqt_savings(raw_memory_gb, cqt_memory_gb)
                
        except Exception as e:
            if self.is_main_process:
                self.logger.error(f"❌ Data setup failed: {e}")
            raise
    
    def _create_dataloaders(self):
        """Create optimized dataloaders for CQT processing"""
        collator = DCAECollator(
            max_length=self.target_length,
            min_length=int(self.config.sample_rate * min(1.0, self.audio_duration * 0.1)),
            pad_to_multiple=512  # Align with CQT hop_length
        )
        
        # Fewer workers needed due to CQT efficiency
        workers_per_gpu = max(1, self.num_workers // self.accelerator.num_processes // 2)
        
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
        # Fused AdamW with CQT-optimized settings
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.config.weight_decay,
            eps=1e-6,
            fused=True if torch.cuda.is_available() else False
        )
        
        # Cosine annealing with warm restarts
        total_steps = self.config.epochs * len(self.train_loader)
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_steps // 6,  # Shorter cycles for CQT training
            T_mult=1,
            eta_min=self.config.learning_rate * 0.001
        )
    
    def _prepare_training(self):
        """Prepare with accelerate"""
        components = [
            self.model, self.optimizer,
            self.train_loader, self.val_loader,
            self.scheduler
        ]
        
        prepared = self.accelerator.prepare(*components)
        (self.model, self.optimizer, self.train_loader, 
         self.val_loader, self.scheduler) = prepared
    
    def train_epoch(self, epoch):
        """한 에폭 학습 with CQT-SSM optimization"""
        self.model.train()
        
        total_loss = 0
        total_cqt_loss = 0
        total_time_loss = 0
        total_vq_loss = 0
        successful_batches = 0
        
        # Music quality metrics
        music_metrics = {
            'snr_scores': [],
            'harmonic_preservation': [],
            'chroma_similarity': []
        }
        
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} (CQT-SSM)')
        else:
            pbar = self.train_loader
        
        for batch_idx, audio in enumerate(pbar):
            try:
                # More frequent memory management for CQT
                if batch_idx % 20 == 0:
                    self.memory_monitor.intelligent_clear()
                
                with self.accelerator.accumulate(self.model):
                    # Mixed precision forward with CQT processing
                    with self.accelerator.autocast():
                        try:
                            # CQT-SSM forward pass
                            reconstructed, loss_dict = self.model(audio, return_loss=True)
                            
                            # Main loss from CQT processing
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
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        # Gradient clipping
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.config.grad_clip
                        )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Update EMA
                    self.state_manager.update_ema()
                    self.state_manager.global_step += 1
                
                # Statistics - use fallback for missing keys
                total_loss += loss.item()
                total_cqt_loss += loss_dict.get('cqt_loss', loss_dict.get('stft_loss', torch.tensor(0.0))).item()
                total_time_loss += loss_dict.get('time_loss', torch.tensor(0.0)).item()
                total_vq_loss += loss_dict.get('vq_loss', torch.tensor(0.0)).item()
                successful_batches += 1
                
                # Music quality analysis (every 50 batches)
                if batch_idx % 50 == 0 and self.accelerator.sync_gradients:
                    try:
                        with torch.no_grad():
                            quality_metrics = self.quality_analyzer.analyze_music_quality(
                                audio[0:1], reconstructed[0:1]
                            )
                            
                            music_metrics['snr_scores'].append(quality_metrics.get('snr_db', 0.0))
                            music_metrics['harmonic_preservation'].append(
                                quality_metrics.get('harmonic_preservation', 0.0)
                            )
                            music_metrics['chroma_similarity'].append(
                                quality_metrics.get('chroma_similarity', 0.0)
                            )
                    except Exception:
                        pass
                
                # Progress update
                if self.is_main_process and self.accelerator.sync_gradients:
                    mem_stats = self.memory_monitor.get_memory_stats()
                    # Use fallback for display
                    cqt_loss_value = loss_dict.get('cqt_loss', loss_dict.get('stft_loss', torch.tensor(0.0))).item()
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'cqt': f'{cqt_loss_value:.3f}',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                        'gpu': f'{mem_stats.get("gpu_allocated_gb", 0):.1f}GB',
                        'savings': f'{mem_stats.get("cqt_memory_savings_gb", 0):.1f}GB'
                    })
                
                # Detailed logging
                if (self.is_main_process and self.args.use_wandb and 
                    self.accelerator.sync_gradients and batch_idx % 200 == 0):
                    
                    # Use fallback values for logging
                    cqt_loss_value = loss_dict.get('cqt_loss', loss_dict.get('stft_loss', torch.tensor(0.0))).item()
                    
                    log_dict = {
                        'train/total_loss': loss.item(),
                        'train/cqt_loss': cqt_loss_value,  # Changed from stft_loss
                        'train/time_loss': loss_dict.get('time_loss', torch.tensor(0.0)).item(),
                        'train/vq_loss': loss_dict.get('vq_loss', torch.tensor(0.0)).item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/batch_size': self.current_batch_size,
                        'train/audio_duration': self.audio_duration,
                        'step': self.state_manager.global_step
                    }
                    
                    # Music quality metrics
                    if music_metrics['snr_scores']:
                        log_dict['train/snr_db'] = np.mean(music_metrics['snr_scores'][-3:])
                    if music_metrics['harmonic_preservation']:
                        log_dict['train/harmonic_preservation'] = np.mean(music_metrics['harmonic_preservation'][-3:])
                    if music_metrics['chroma_similarity']:
                        log_dict['train/chroma_similarity'] = np.mean(music_metrics['chroma_similarity'][-3:])
                    
                    # Memory optimization metrics
                    mem_summary = self.memory_monitor.get_memory_summary()
                    log_dict.update({f'memory/{k}': v for k, v in mem_summary.items() if isinstance(v, (int, float))})
                    
                    # CQT-SSM specific metrics
                    log_dict['cqt_ssm/representation'] = 'CQT + Harmonic-Percussive'
                    log_dict['cqt_ssm/n_bins'] = self.model.n_bins
                    log_dict['cqt_ssm/hop_length'] = self.model.hop_length
                    
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
        avg_cqt = total_cqt_loss / max(successful_batches, 1)
        avg_time = total_time_loss / max(successful_batches, 1)
        avg_vq = total_vq_loss / max(successful_batches, 1)
        
        # Music metrics averages
        avg_snr = np.mean(music_metrics['snr_scores']) if music_metrics['snr_scores'] else 0.0
        avg_harmonic = np.mean(music_metrics['harmonic_preservation']) if music_metrics['harmonic_preservation'] else 0.0
        avg_chroma = np.mean(music_metrics['chroma_similarity']) if music_metrics['chroma_similarity'] else 0.0
        
        return {
            'loss': avg_loss,
            'cqt_loss': avg_cqt,  # Changed from stft_loss
            'time_loss': avg_time,
            'vq_loss': avg_vq,
            'snr': avg_snr,
            'harmonic_preservation': avg_harmonic,
            'chroma_similarity': avg_chroma
        }
    
    def _handle_oom(self, epoch):
        """Smart OOM handling for CQT processing"""
        self.oom_count += 1
        
        if self.current_batch_size > self.min_batch_size:
            old_bs = self.current_batch_size
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            
            if self.is_main_process:
                self.logger.warning(
                    f"💥 OOM! Reducing batch size: {old_bs} → {self.current_batch_size} "
                    f"(CQT-SSM, OOM #{self.oom_count})"
                )
            
            # Clear memory
            torch.cuda.empty_cache()
            gc.collect()
            
            self._recreate_dataloaders()
            return True
        
        if self.is_main_process:
            self.logger.error(
                f"❌ Cannot reduce batch size further with CQT-SSM! "
                f"Consider reducing --audio_duration from {self.audio_duration}s"
            )
        
        return False
    
    def _recreate_dataloaders(self):
        """Recreate dataloaders with new batch size"""
        self._create_dataloaders()
        
        self.train_loader, self.val_loader = self.accelerator.prepare(
            self.train_loader, self.val_loader
        )
        
        if self.is_main_process:
            self.logger.info(f"🔄 Dataloaders recreated with batch_size={self.current_batch_size}")
    
    def validate(self, epoch):
        """검증 수행 with CQT-SSM and music quality analysis"""
        self.model.eval()
        
        total_loss = 0
        total_cqt = 0
        total_time = 0
        total_vq = 0
        batch_count = 0
        
        # Music quality metrics
        music_metrics = {
            'snr_scores': [],
            'harmonic_preservation': [],
            'chroma_similarity': [],
            'tempo_errors': [],
            'spectral_centroid_errors': []
        }
        
        # Use EMA for validation
        ema_context = self.state_manager.get_ema_context()
        context_manager = ema_context if ema_context else torch.no_grad()
        
        with context_manager:
            for batch_idx, audio in enumerate(tqdm(self.val_loader, desc='Validation (CQT-SSM)', 
                                                  disable=not self.is_main_process)):
                if batch_idx >= 15:  # Limit validation batches for CQT
                    break
                
                try:
                    with self.accelerator.autocast():
                        reconstructed, loss_dict = self.model(audio, return_loss=True)
                    
                    total_loss += loss_dict['total_loss'].item()
                    # Use fallback for missing keys
                    total_cqt += loss_dict.get('cqt_loss', loss_dict.get('stft_loss', torch.tensor(0.0))).item()
                    total_time += loss_dict.get('time_loss', torch.tensor(0.0)).item()
                    total_vq += loss_dict.get('vq_loss', torch.tensor(0.0)).item()
                    batch_count += 1
                    
                    # Comprehensive music quality analysis
                    if batch_idx < 5:
                        try:
                            quality_metrics = self.quality_analyzer.analyze_music_quality(
                                audio[0:1], reconstructed[0:1]
                            )
                            
                            music_metrics['snr_scores'].append(quality_metrics.get('snr_db', 0.0))
                            music_metrics['harmonic_preservation'].append(
                                quality_metrics.get('harmonic_preservation', 0.0)
                            )
                            music_metrics['chroma_similarity'].append(
                                quality_metrics.get('chroma_similarity', 0.0)
                            )
                            music_metrics['tempo_errors'].append(
                                quality_metrics.get('tempo_error_bpm', 0.0)
                            )
                            music_metrics['spectral_centroid_errors'].append(
                                quality_metrics.get('spectral_centroid_error', 0.0)
                            )
                        except Exception:
                            pass
                
                except Exception as e:
                    if self.is_main_process:
                        self.logger.warning(f"Val batch {batch_idx} failed: {e}")
                    continue
        
        # Compute averages
        metrics = {
            'loss': total_loss / max(batch_count, 1),
            'cqt_loss': total_cqt / max(batch_count, 1),  # Changed from stft_loss
            'time_loss': total_time / max(batch_count, 1),
            'vq_loss': total_vq / max(batch_count, 1),
            'snr': np.mean(music_metrics['snr_scores']) if music_metrics['snr_scores'] else 0.0,
            'harmonic_preservation': np.mean(music_metrics['harmonic_preservation']) if music_metrics['harmonic_preservation'] else 0.0,
            'chroma_similarity': np.mean(music_metrics['chroma_similarity']) if music_metrics['chroma_similarity'] else 0.0,
            'tempo_error_bpm': np.mean(music_metrics['tempo_errors']) if music_metrics['tempo_errors'] else 0.0,
            'spectral_centroid_error': np.mean(music_metrics['spectral_centroid_errors']) if music_metrics['spectral_centroid_errors'] else 0.0
        }
        
        # Logging
        if self.is_main_process and self.args.use_wandb:
            log_dict = {
                'val/loss': metrics['loss'],
                'val/cqt_loss': metrics['cqt_loss'],  # Changed from stft_loss
                'val/time_loss': metrics['time_loss'],
                'val/vq_loss': metrics['vq_loss'],
                'val/snr_db': metrics['snr'],
                'val/harmonic_preservation': metrics['harmonic_preservation'],
                'val/chroma_similarity': metrics['chroma_similarity'],
                'val/tempo_error_bpm': metrics['tempo_error_bpm'],
                'val/spectral_centroid_error': metrics['spectral_centroid_error'],
                'val/audio_duration': self.audio_duration,
                'epoch': epoch
            }
            
            # Add memory summary
            mem_summary = self.memory_monitor.get_memory_summary()
            log_dict.update({f'val_memory/{k}': v for k, v in mem_summary.items() if isinstance(v, (int, float))})
            
            self.accelerator.log(log_dict)
        
        return metrics
    
    def generate_samples(self, epoch, num_samples=4):
        """Generate samples with CQT-SSM and music quality analysis"""
        if not self.is_main_process:
            return
            
        self.model.eval()
        
        # Use EMA for sample generation
        ema_context = self.state_manager.get_ema_context()
        context_manager = ema_context if ema_context else torch.no_grad()
        
        with context_manager:
            try:
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
                        
                        # Quality analysis for sample
                        if i == 0:
                            quality_metrics = self.quality_analyzer.analyze_music_quality(
                                audio[i:i+1], reconstructed[i:i+1]
                            )
                            
                            # Save quality report
                            quality_path = sample_dir / f'quality_analysis_{i}.json'
                            with open(quality_path, 'w') as f:
                                json.dump(quality_metrics, f, indent=2)
                        
                        # Wandb audio logging
                        if self.args.use_wandb:
                            self.accelerator.log({
                                f'samples/original_{i}': wandb.Audio(
                                    audio[i].cpu().numpy(),
                                    sample_rate=self.config.sample_rate,
                                    caption=f'Original {i} (CQT-SSM, {self.audio_duration}s)'
                                ),
                                f'samples/reconstructed_{i}': wandb.Audio(
                                    reconstructed[i].cpu().numpy(),
                                    sample_rate=self.config.sample_rate,
                                    caption=f'CQT-SSM Reconstructed {i} (Epoch {epoch})'
                                )
                            })
                    
                    self.logger.info(f"💾 CQT-SSM samples saved to {sample_dir}")
                    
            except Exception as e:
                self.logger.warning(f"Sample generation error: {e}")
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save checkpoint with CQT-SSM model info"""
        if not self.is_main_process:
            return
        
        # Save accelerate state
        save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}'
        self.accelerator.save_state(str(save_path))
        
        # Enhanced metadata with CQT-SSM info
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
            'model_type': 'CQT-SSM-DCAE',
            'representation': 'CQT + Harmonic-Percussive',
            'memory_optimization': {
                'cqt_based': True,
                'harmonic_percussive_separation': True,
                'chunk_size': self.memory_chunk_size,
                'use_checkpointing': self.use_checkpointing,
                'memory_efficient': self.memory_efficient,
                'estimated_savings': '90% vs raw audio SSM'
            },
            'memory_stats': self.memory_monitor.get_memory_summary(),
            'model_stats': self.model.get_memory_stats() if hasattr(self.model, 'get_memory_stats') else {}
        }
        
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model'
            self.accelerator.save_state(str(best_path))
            with open(best_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Save EMA model separately
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
                    'model_type': 'CQT-SSM-DCAE',
                    'representation': 'CQT + Harmonic-Percussive',
                    'audio_duration': self.audio_duration,
                    'memory_optimization': metadata['memory_optimization']
                }
                
                ema_path = self.checkpoint_dir / 'best_model_ema.pt'
                torch.save(ema_checkpoint, ema_path)
                self.logger.info(f"CQT-SSM EMA model saved to {ema_path}")
        
        # Cleanup old checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*'))
        if len(checkpoints) > 3:
            for ckpt in checkpoints[:-3]:
                import shutil
                shutil.rmtree(ckpt, ignore_errors=True)
        
        self.logger.info(f"💾 CQT-SSM checkpoint saved: epoch {epoch}")
    
    def train(self):
        """Complete CQT-SSM training loop"""
        if self.is_main_process:
            self.logger.info(f"\n🎵 CQT-SSM-DCAE Training Started")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"🎼 Representation: CQT + Harmonic-Percussive Separation")
            self.logger.info(f"🧠 Model: SSM-based with spectral efficiency")
            self.logger.info(f"📈 EMA: {'Enabled' if self.config.use_ema else 'Disabled'}")
            self.logger.info(f"🔊 Audio Duration: {self.audio_duration}s")
            self.logger.info(f"🚀 Memory Savings: ~90% vs raw audio SSM")
            self.logger.info(f"⚡ Multi-GPU: {self.accelerator.num_processes}")
            self.logger.info(f"🎯 Mixed Precision: fp16")
            self.logger.info(f"{'='*80}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            if self.is_main_process:
                self.logger.info(f"\n📅 Epoch {epoch+1}/{self.config.epochs} (CQT-SSM)")
            
            # Memory management
            self.memory_monitor.intelligent_clear()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            if self.is_main_process:
                mem_stats = self.memory_monitor.get_memory_stats()
                self.logger.info(
                    f"🎵 Train - Loss: {train_metrics['loss']:.4f}, "
                    f"CQT: {train_metrics['cqt_loss']:.4f}, "  # Changed from stft_loss
                    f"Harmonic: {train_metrics['harmonic_preservation']:.3f}, "
                    f"Chroma: {train_metrics['chroma_similarity']:.3f}, "
                    f"GPU: {mem_stats.get('gpu_allocated_gb', 0):.1f}GB"
                )
            
            # Validation (every 2 epochs)
            if epoch % 2 == 0:
                val_metrics = self.validate(epoch)
                
                if self.is_main_process:
                    self.logger.info(
                        f"✅ Val - Loss: {val_metrics['loss']:.4f}, "
                        f"CQT: {val_metrics['cqt_loss']:.4f}, "  # Changed from stft_loss
                        f"Harmonic: {val_metrics['harmonic_preservation']:.3f}, "
                        f"Chroma: {val_metrics['chroma_similarity']:.3f}"
                    )
                
                # Best model tracking with music-specific metrics
                is_best = (val_metrics['loss'] < self.best_metrics['val_loss'] or
                          val_metrics['harmonic_preservation'] > self.best_metrics['harmonic_preservation'])
                
                if is_best:
                    self.best_metrics.update({
                        'val_loss': val_metrics['loss'],
                        'harmonic_preservation': val_metrics['harmonic_preservation'],
                        'chroma_similarity': val_metrics['chroma_similarity']
                    })
                    if self.is_main_process:
                        self.logger.info(f"🏆 New best CQT-SSM model! Val loss: {val_metrics['loss']:.4f}")
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
                    f"CQT Savings: {mem_summary.get('cqt_memory_savings_gb', 0):.1f}GB"
                )
        
        # Training completion
        if self.is_main_process:
            total_time = (time.time() - start_time) / 3600
            final_mem_summary = self.memory_monitor.get_memory_summary()
            
            self.logger.info(f"\n🎉 CQT-SSM Training Completed!")
            self.logger.info(f"⏱️  Total Time: {total_time:.2f} hours")
            self.logger.info(f"🏆 Best Val Loss: {self.best_metrics['val_loss']:.4f}")
            self.logger.info(f"🎼 Best Harmonic Preservation: {self.best_metrics['harmonic_preservation']:.3f}")
            self.logger.info(f"🎵 Best Chroma Similarity: {self.best_metrics['chroma_similarity']:.3f}")
            self.logger.info(f"💾 Peak Memory Usage: {final_mem_summary.get('peak_memory_gb', 0):.2f} GB")
            self.logger.info(f"🚀 Total Memory Savings: {final_mem_summary.get('cqt_memory_savings_gb', 0):.2f} GB")
            self.logger.info(f"🔊 Audio Duration: {self.audio_duration}s")
            
            # Final save
            final_metrics = {
                'training_completed': True, 
                'total_hours': total_time,
                'final_memory_summary': final_mem_summary,
                'representation': 'CQT + Harmonic-Percussive'
            }
            self.save_checkpoint(self.config.epochs - 1, final_metrics, is_best=False)
            
            if self.args.use_wandb:
                self.accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description='CQT-SSM-based LYRO DCAE Training')
    
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
    
    # 오디오 길이 설정
    parser.add_argument('--audio_duration', type=float, default=10.0,
                        help='Audio duration in seconds for training (default: 10.0)')
    
    # 메모리 최적화 설정
    parser.add_argument('--chunk_size', type=int, default=256,
                        help='Chunk size for CQT-SSM processing (default: 256)')
    parser.add_argument('--disable_checkpointing', action='store_true',
                        help='Disable gradient checkpointing')
    parser.add_argument('--memory_efficient', action='store_true', default=True,
                        help='Enable memory efficient processing')
    parser.add_argument('--checkpointing_segments', type=int, default=4,
                        help='Number of segments for checkpointing (default: 4)')
    
    # 학습 관련
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (reduced default for CQT efficiency)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    
    # EMA 관련
    parser.add_argument('--disable_ema', action='store_true',
                        help='Disable EMA')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay rate')
    
    # 체크포인트 및 로깅
    parser.add_argument('--checkpoint_dir', type=str, default='dcae/checkpoints_cqt_ssm',
                        help='Checkpoint directory')
    parser.add_argument('--exp_name', type=str, default='cqt_ssm_dcae',
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
            print("❌ CUDA required for CQT-SSM training!")
        return
    
    if is_main:
        print(f"🎵 CQT-SSM-based DCAE Training")
        print(f"⚡ Available GPUs: {torch.cuda.device_count()}")
        print(f"🔧 CPU Workers: {max(1, multiprocessing.cpu_count() - 2)}")
        print(f"🎼 Representation: CQT + Harmonic-Percussive")
        print(f"🔊 Audio Duration: {args.audio_duration}s")
        print(f"🧩 Chunk Size: {args.chunk_size}")
        print(f"✅ Checkpointing: {'Disabled' if args.disable_checkpointing else 'Enabled'}")
        print(f"💾 Memory Efficient: {args.memory_efficient}")
        print(f"🚀 Expected Memory Savings: ~90% vs raw audio SSM")
    
    try:
        trainer = CQTSSMDCAETrainer(args)
        trainer.train()
        if trainer.is_main_process:
            print("🎉 CQT-SSM training completed successfully!")
        
    except Exception as e:
        if is_main:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
