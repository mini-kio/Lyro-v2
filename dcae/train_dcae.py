# lyro/dcae/train_dcae.py
"""
Enhanced LYRO DCAE Training Script
Integrates all improvements: A-3, T-3, T-2, A-5, T-1, A-2
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import time
import librosa  # ✅ 추가: 누락된 import
import torchaudio  # ✅ 추가: 누락된 import
from typing import Dict, Optional, List

# LYRO 모듈 임포트
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enhanced modules
from dcae.model import LyroMusicDCAE, create_enhanced_lyro_dcae  # ✅ 수정: 올바른 import
from dcae.training_utils import (
    EMAWrapper, 
    TrainingStateManager, 
    EnhancedDCAEConfig,
    compute_snr, 
    compute_si_sdr, 
    analyze_frequency_response
)
from dataset.dataset import LyroDataset, LyroCollator


class MultiScaleDiscriminator(nn.Module):
    """✅ 추가: 누락된 Discriminator 클래스"""
    
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


class EnhancedDCAETrainer:
    """
    Enhanced DCAE 학습 관리 클래스
    
    All improvements integrated:
    - A-3: FIR Low-pass filtering in upsampling
    - T-3: Exponential Moving Average
    - T-2: Weight Normalization
    - A-5: Extended STFT resolutions
    - T-1: Mix-scale augmentation
    - A-2: Enhanced skip connections
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Enhanced configuration
        self.config = EnhancedDCAEConfig()
        self._update_config_from_args()
        
        # Training state manager (handles EMA, augmentation, metrics)
        self.state_manager = TrainingStateManager(self.config)
        
        # 모델 초기화
        self._initialize_models()
        
        # 옵티마이저 및 스케줄러
        self._setup_optimization()
        
        # 데이터셋 및 로더
        self._setup_data()
        
        # 체크포인트 디렉토리
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Wandb 초기화
        if args.use_wandb:
            wandb.init(
                project="lyro-dcae-enhanced",
                name=args.exp_name,
                config={**vars(args), **self.config.__dict__}
            )
            
            # Log augmentation stats
            if hasattr(self.train_dataset, 'get_augmentation_stats'):
                wandb.log(self.train_dataset.get_augmentation_stats())
    
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
            
        # Augmentation settings
        if hasattr(self.args, 'disable_augmentation') and self.args.disable_augmentation:
            self.config.use_augmentation = False
        if hasattr(self.args, 'augmentation_prob') and self.args.augmentation_prob:
            self.config.augmentation_prob = self.args.augmentation_prob
    
    def _initialize_models(self):
        """모델 초기화 with all enhancements"""
        # Enhanced DCAE 모델
        self.model = create_enhanced_lyro_dcae(  # ✅ 수정: 올바른 함수 사용
            model_size="base",
            sample_rate=self.config.sample_rate,
            use_vq=self.config.use_vector_quantization,
            use_weight_norm=self.config.use_weight_norm,
            encoder_base_channels=self.config.encoder_base_channels,
            decoder_base_channels=self.config.decoder_base_channels,
            dropout=0.1
        ).to(self.device)
        
        # T-3: Setup EMA wrapper
        self.state_manager.setup_ema(self.model)
        
        # Discriminator for adversarial training (optional)
        if self.args.use_adversarial:
            self.discriminator = MultiScaleDiscriminator().to(self.device)
            self.disc_optimizer = optim.AdamW(
                self.discriminator.parameters(),
                lr=self.config.learning_rate * 0.5,
                betas=(0.5, 0.999)
            )
            self.adv_loss = nn.BCEWithLogitsLoss()
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        if self.config.use_ema:
            print(f"EMA enabled with decay {self.config.ema_decay}")
            
    def _setup_optimization(self):
        """옵티마이저 및 스케줄러 설정"""
        # 옵티마이저
        self.optimizer = self.config.get_optimizer(self.model)
        
        # 스케줄러
        total_steps = self.config.epochs * len(self.train_loader) if hasattr(self, 'train_loader') else 1000
        self.scheduler = self.config.get_scheduler(self.optimizer, total_steps)
        
    def _setup_data(self):
        """데이터셋 및 로더 설정 with enhanced augmentation"""
        # Enhanced augmentation configuration
        augmentation_config = {
            'sample_rate': self.config.sample_rate,
            'augmentation_prob': self.config.augmentation_prob,
            'gain_range': self.config.gain_range,
            'pitch_range': self.config.pitch_range,
            'tempo_range': self.config.tempo_range,
            'noise_level': self.config.noise_level,
            'reverb_prob': self.config.reverb_prob,
            'eq_prob': self.config.eq_prob
        }
        
        # 데이터셋
        self.train_dataset = LyroDataset(
            metadata_path=self.args.train_metadata,
            dataset_root=self.args.dataset_root,
            sample_rate=self.config.sample_rate,
            augmentation=self.config.use_augmentation,
            augmentation_config=augmentation_config  # T-1: Enhanced augmentation
        )
        
        self.val_dataset = LyroDataset(
            metadata_path=self.args.val_metadata,
            dataset_root=self.args.dataset_root,
            sample_rate=self.config.sample_rate,
            augmentation=False  # No augmentation for validation
        )
        
        # Enhanced Collator
        collator = LyroCollator(
            max_audio_length=self.config.max_length,
            batch_mix_prob=0.1 if self.config.use_augmentation else 0.0  # T-1: Batch mixing
        )
        
        # 데이터 로더
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collator,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collator
        )
        
        # Update scheduler with actual data size
        total_steps = self.config.epochs * len(self.train_loader)
        self.scheduler = self.config.get_scheduler(self.optimizer, total_steps)
        
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
    def train_epoch(self, epoch):
        """한 에폭 학습 with all enhancements"""
        self.model.train()
        self.train_dataset.train()  # Enable augmentation
        
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        total_adv_loss = 0
        
        # Quality metrics
        snr_scores = []
        si_sdr_scores = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                audio = batch['audio'].to(self.device)
                
                # Forward pass through enhanced model
                start_time = time.time()
                
                # Use enhanced forward pass (with skip connections)
                if hasattr(self.model, 'forward'):
                    reconstructed, loss_dict = self.model(audio, return_loss=True)
                else:
                    # Fallback for compatibility
                    latent, skip_features = self.model.encode(audio)
                    reconstructed = self.model.decode(latent, skip_features)
                    
                    # Compute losses manually
                    stft_loss = self.model.stft_loss(reconstructed, audio)
                    time_loss = nn.functional.l1_loss(reconstructed, audio)
                    loss_dict = {
                        'total_loss': stft_loss + 0.1 * time_loss,
                        'stft_loss': stft_loss,
                        'time_loss': time_loss,
                        'vq_loss': torch.tensor(0.0, device=audio.device)
                    }
                
                forward_time = time.time() - start_time
                
                # Adversarial training (optional)
                adv_loss = torch.tensor(0.0, device=audio.device)
                if self.args.use_adversarial and hasattr(self, 'discriminator'):
                    adv_loss = self._train_adversarial(audio, reconstructed)
                    loss_dict['total_loss'] += self.config.adv_weight * adv_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.grad_clip
                )
                
                self.optimizer.step()
                self.scheduler.step()
                
                # T-3: Update EMA
                self.state_manager.update_ema()
                self.state_manager.global_step += 1
                
                # 손실 기록
                total_loss += loss_dict['total_loss'].item()
                total_recon_loss += loss_dict['stft_loss'].item()
                total_vq_loss += loss_dict['vq_loss'].item()
                total_adv_loss += adv_loss.item()
                
                # Quality metrics (sample a few batches)
                if batch_idx % 20 == 0:  # Every 20 batches
                    with torch.no_grad():
                        # Compute SNR and SI-SDR for first sample in batch
                        snr = compute_snr(audio[0], reconstructed[0])
                        si_sdr = compute_si_sdr(audio[0].flatten(), reconstructed[0].flatten())
                        snr_scores.append(snr)
                        si_sdr_scores.append(si_sdr)
                
                # Progress bar 업데이트
                pbar.set_postfix({
                    'loss': f'{loss_dict["total_loss"].item():.4f}',
                    'recon': f'{loss_dict["stft_loss"].item():.4f}',
                    'vq': f'{loss_dict["vq_loss"].item():.4f}',
                    'grad': f'{grad_norm:.3f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'fwd_ms': f'{forward_time*1000:.1f}'
                })
                
                # Wandb 로깅
                if self.args.use_wandb and batch_idx % self.args.log_interval == 0:
                    log_dict = {
                        'train/loss': loss_dict['total_loss'].item(),
                        'train/stft_loss': loss_dict['stft_loss'].item(),
                        'train/time_loss': loss_dict.get('time_loss', torch.tensor(0.0)).item(),
                        'train/vq_loss': loss_dict['vq_loss'].item(),
                        'train/grad_norm': grad_norm,
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/forward_time_ms': forward_time * 1000,
                        'step': self.state_manager.global_step
                    }
                    
                    if adv_loss.item() > 0:
                        log_dict['train/adv_loss'] = adv_loss.item()
                    
                    if snr_scores:
                        log_dict['train/snr_db'] = np.mean(snr_scores[-5:])  # Last 5 scores
                    if si_sdr_scores:
                        log_dict['train/si_sdr_db'] = np.mean(si_sdr_scores[-5:])
                    
                    wandb.log(log_dict)
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM error at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # 에폭 평균
        num_batches = len(self.train_loader)
        avg_loss = total_loss / max(num_batches, 1)
        avg_recon = total_recon_loss / max(num_batches, 1)
        avg_vq = total_vq_loss / max(num_batches, 1)
        avg_adv = total_adv_loss / max(num_batches, 1)
        
        # Record metrics in state manager
        self.state_manager.record_metrics({
            'train_loss': avg_loss,
            'train_recon_loss': avg_recon,
            'train_vq_loss': avg_vq,
            'train_adv_loss': avg_adv,
            'train_snr': np.mean(snr_scores) if snr_scores else 0.0,
            'train_si_sdr': np.mean(si_sdr_scores) if si_sdr_scores else 0.0
        })
        
        return avg_loss, avg_recon, avg_vq, avg_adv
    
    def _train_adversarial(self, real_audio, fake_audio):
        """Adversarial training step"""
        # Train discriminator
        self.disc_optimizer.zero_grad()
        
        # Real samples
        real_scores = self.discriminator(real_audio.detach())
        real_loss = sum(self.adv_loss(score, torch.ones_like(score)) for score in real_scores)
        
        # Fake samples
        fake_scores = self.discriminator(fake_audio.detach())
        fake_loss = sum(self.adv_loss(score, torch.zeros_like(score)) for score in fake_scores)
        
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # Generator adversarial loss
        fake_scores = self.discriminator(fake_audio)
        gen_adv_loss = sum(self.adv_loss(score, torch.ones_like(score)) for score in fake_scores)
        
        return gen_adv_loss / len(fake_scores)
    
    def validate(self, epoch):
        """검증 수행 with EMA"""
        self.model.eval()
        self.val_dataset.eval()  # Disable augmentation
        
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        
        # Quality metrics
        snr_scores = []
        si_sdr_scores = []
        freq_analysis = []
        
        # T-3: Use EMA for validation if available
        ema_context = self.state_manager.get_ema_context()
        
        context_manager = ema_context if ema_context else torch.no_grad()
        
        with context_manager:
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                try:
                    audio = batch['audio'].to(self.device)
                    
                    # Forward pass
                    if hasattr(self.model, 'forward'):
                        reconstructed, loss_dict = self.model(audio, return_loss=True)
                    else:
                        latent, skip_features = self.model.encode(audio)
                        reconstructed = self.model.decode(latent, skip_features)
                        
                        stft_loss = self.model.stft_loss(reconstructed, audio)
                        time_loss = nn.functional.l1_loss(reconstructed, audio)
                        loss_dict = {
                            'total_loss': stft_loss + 0.1 * time_loss,
                            'stft_loss': stft_loss,
                            'time_loss': time_loss,
                            'vq_loss': torch.tensor(0.0, device=audio.device)
                        }
                    
                    total_loss += loss_dict['total_loss'].item()
                    total_recon_loss += loss_dict['stft_loss'].item()
                    total_vq_loss += loss_dict['vq_loss'].item()
                    
                    # Quality metrics for first sample in each batch
                    if batch_idx < 10:  # Analyze first 10 batches
                        snr = compute_snr(audio[0], reconstructed[0])
                        si_sdr = compute_si_sdr(audio[0].flatten(), reconstructed[0].flatten())
                        snr_scores.append(snr)
                        si_sdr_scores.append(si_sdr)
                        
                        # Frequency analysis
                        freq_stats = analyze_frequency_response(
                            audio[0], reconstructed[0], self.config.sample_rate
                        )
                        freq_analysis.append(freq_stats)
                        
                except Exception as e:
                    print(f"Validation error at batch {batch_idx}: {e}")
                    continue
        
        # 평균 계산
        num_batches = max(len(self.val_loader), 1)
        avg_loss = total_loss / num_batches
        avg_recon = total_recon_loss / num_batches
        avg_vq = total_vq_loss / num_batches
        
        # Quality metrics
        avg_snr = np.mean(snr_scores) if snr_scores else 0.0
        avg_si_sdr = np.mean(si_sdr_scores) if si_sdr_scores else 0.0
        
        # Frequency analysis
        avg_freq_stats = {}
        if freq_analysis:
            for key in freq_analysis[0].keys():
                avg_freq_stats[key] = np.mean([stats[key] for stats in freq_analysis])
        
        # Record metrics
        val_metrics = {
            'val_loss': avg_loss,
            'val_recon_loss': avg_recon,
            'val_vq_loss': avg_vq,
            'val_snr': avg_snr,
            'val_si_sdr': avg_si_sdr,
            **{f'val_{k}': v for k, v in avg_freq_stats.items()}
        }
        
        self.state_manager.record_metrics(val_metrics)
        
        # Wandb 로깅
        if self.args.use_wandb:
            log_dict = {
                'val/loss': avg_loss,
                'val/recon_loss': avg_recon,
                'val/vq_loss': avg_vq,
                'val/snr_db': avg_snr,
                'val/si_sdr_db': avg_si_sdr,
                'epoch': epoch
            }
            
            # Add frequency analysis
            for key, value in avg_freq_stats.items():
                log_dict[f'val/{key}'] = value
            
            wandb.log(log_dict)
            
        return avg_loss, avg_recon, avg_vq, avg_snr, avg_si_sdr
    
    def generate_samples(self, epoch, num_samples=4):
        """검증 중 샘플 생성 with EMA"""
        self.model.eval()
        
        # Use EMA for sample generation if available
        ema_context = self.state_manager.get_ema_context()
        context_manager = ema_context if ema_context else torch.no_grad()
        
        with context_manager:
            try:
                # Take a few samples from validation set
                val_batch = next(iter(self.val_loader))
                audio = val_batch['audio'][:num_samples].to(self.device)
                
                # Generate reconstructions
                if hasattr(self.model, 'forward'):
                    reconstructed, _ = self.model(audio, return_loss=True)
                else:
                    latent, skip_features = self.model.encode(audio)
                    reconstructed = self.model.decode(latent, skip_features)
                
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
                            wandb.log({
                                f'samples/original_{i}': wandb.Audio(
                                    audio[i].cpu().numpy(),
                                    sample_rate=self.config.sample_rate,
                                    caption=f'Original {i}'
                                ),
                                f'samples/reconstructed_{i}': wandb.Audio(
                                    reconstructed[i].cpu().numpy(),
                                    sample_rate=self.config.sample_rate,
                                    caption=f'Reconstructed {i} (Epoch {epoch})'
                                )
                            })
                    
                    print(f"Samples saved to {sample_dir}")
                    
            except Exception as e:
                print(f"Sample generation error: {e}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """체크포인트 저장 with EMA"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'args': self.args,
            'training_state': self.state_manager.state_dict()  # Includes EMA
        }
        
        # 일반 체크포인트
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # 베스트 모델
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
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
                    'ema_decay': self.config.ema_decay
                }
                
                ema_path = self.checkpoint_dir / 'best_model_ema.pt'
                torch.save(ema_checkpoint, ema_path)
                print(f"EMA model saved to {ema_path}")
        
        # 최근 체크포인트만 유지
        if self.args.keep_last_n > 0:
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
            if len(checkpoints) > self.args.keep_last_n:
                for ckpt in checkpoints[:-self.args.keep_last_n]:
                    ckpt.unlink()
                    
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """체크포인트 로드 with EMA"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state (includes EMA)
        if 'training_state' in checkpoint:
            self.state_manager.load_state_dict(checkpoint['training_state'])
        
        return checkpoint['epoch']
    
    def train(self):
        """전체 학습 루프 with all enhancements"""
        best_val_loss = float('inf')
        start_epoch = 0
        
        # 체크포인트 로드 (있는 경우)
        if self.args.resume:
            start_epoch = self.load_checkpoint(self.args.resume)
            print(f"Resumed from epoch {start_epoch}")
            
        print(f"\nStarting Enhanced DCAE training")
        print(f"{'='*60}")
        print(f"Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"EMA: {'Enabled' if self.config.use_ema else 'Disabled'}")
        print(f"Augmentation: {'Enabled' if self.config.use_augmentation else 'Disabled'}")
        print(f"Weight Norm: {'Enabled' if self.config.use_weight_norm else 'Disabled'}")
        print(f"Enhanced STFT: {len(self.model.stft_loss.fft_sizes)} resolutions")
        print(f"{'='*60}")
        
        # 학습 시작
        for epoch in range(start_epoch, self.config.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config.epochs}")
            print(f"{'='*50}")
            
            # 학습
            train_loss, train_recon, train_vq, train_adv = self.train_epoch(epoch)
            print(f"Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, "
                  f"VQ: {train_vq:.4f}, Adv: {train_adv:.4f}")
            
            # 검증
            val_loss, val_recon, val_vq, val_snr, val_si_sdr = self.validate(epoch)
            print(f"Val - Loss: {val_loss:.4f}, Recon: {val_recon:.4f}, "
                  f"VQ: {val_vq:.4f}")
            print(f"Quality - SNR: {val_snr:.2f} dB, SI-SDR: {val_si_sdr:.2f} dB")
            
            # 샘플 생성 (주기적으로)
            if epoch % self.args.sample_interval == 0:
                self.generate_samples(epoch)
                
            # 체크포인트 저장
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"New best model! Val loss: {val_loss:.4f}")
                
            if epoch % self.args.save_interval == 0:
                self.save_checkpoint(epoch, is_best)
                
            # Update state manager
            self.state_manager.current_epoch = epoch
        
        print("Enhanced DCAE training completed!")
        
        # 최종 모델 저장
        self.save_checkpoint(self.config.epochs, is_best=False)
        
        # Latent 추출 및 저장
        if self.args.extract_latents:
            self.extract_all_latents()
    
    def extract_all_latents(self):
        """학습 후 모든 데이터의 latent 추출 with EMA"""
        print("\nExtracting latents with EMA model...")
        self.model.eval()
        
        latent_dir = Path(self.args.dataset_root) / "processed" / "latents"
        latent_dir.mkdir(parents=True, exist_ok=True)
        
        # Use EMA for latent extraction
        ema_context = self.state_manager.get_ema_context()
        context_manager = ema_context if ema_context else torch.no_grad()
        
        with context_manager:
            for item in tqdm(self.train_dataset.items):
                try:
                    # 오디오 로드
                    audio_path = Path(self.args.dataset_root) / item['audio_path']
                    audio, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=False)
                    
                    if audio.ndim == 1:
                        audio = np.stack([audio, audio], axis=0)
                        
                    # Latent 추출
                    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
                    latent, skip_features = self.model.encode(audio_tensor)
                    
                    # 저장
                    save_path = latent_dir / f"{item['id']}_latent.pt"
                    torch.save({
                        'latent': latent.cpu(),
                        'skip_features': [sf.cpu() for sf in skip_features]
                    }, save_path)
                    
                except Exception as e:
                    print(f"Error extracting latent for {item['id']}: {e}")
                    continue
                
        print(f"Enhanced latents saved to {latent_dir}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced LYRO DCAE Training')
    
    # 데이터 관련
    parser.add_argument('--train_metadata', type=str, required=True,
                        help='Training metadata path')
    parser.add_argument('--val_metadata', type=str, required=True,
                        help='Validation metadata path')
    parser.add_argument('--dataset_root', type=str, default='dataset/',
                        help='Dataset root directory')
    
    # 모델 관련
    parser.add_argument('--sample_rate', type=int, default=44100,
                        help='Audio sample rate')
    parser.add_argument('--latent_channels', type=int, default=8,
                        help='Number of latent channels')
    
    # 학습 관련
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping')
    
    # T-3: EMA 관련
    parser.add_argument('--disable_ema', action='store_true',
                        help='Disable EMA')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay rate')
    
    # T-1: Augmentation 관련
    parser.add_argument('--disable_augmentation', action='store_true',
                        help='Disable augmentation')
    parser.add_argument('--augmentation_prob', type=float, default=0.8,
                        help='Augmentation probability')
    
    # Adversarial 관련
    parser.add_argument('--use_adversarial', action='store_true',
                        help='Use adversarial training')
    
    # 기타
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='dcae/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--exp_name', type=str, default='enhanced_dcae',
                        help='Experiment name')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Checkpoint save interval')
    parser.add_argument('--sample_interval', type=int, default=10,
                        help='Sample generation interval')
    parser.add_argument('--keep_last_n', type=int, default=5,
                        help='Keep last N checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--extract_latents', action='store_true',
                        help='Extract latents after training')
    parser.add_argument('--save_samples', action='store_true',
                        help='Save audio samples during training')
    
    args = parser.parse_args()
    
    # 학습 시작
    trainer = EnhancedDCAETrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()