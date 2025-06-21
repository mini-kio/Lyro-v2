# lyro/ssm/train_lyro.py - Refactored Large Model Training
"""
LYRO S6 Training - Large Model Only + FP16 Enforced + Fixed GradScaler
Flow Matching Based S6 Music Generation Training
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional, List, Any
import math
import torchaudio
import warnings
import gc
import time

# W&B handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from ssm.model import create_lyro_s6_model, TaskController, EOSTokenHandler
from ssm.flow_matching import create_s6_flow_matching, FlowConfig
from dcae.model import create_large_dcae_model
from dataset.dataset import LyroDataset, LyroCollator
from dataset.tokenizer import LyroTokenizer

warnings.filterwarnings("ignore")


# ==================== Simplified Utilities ====================

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
        return float(value.mean().item())
    
    return 0.0


def safe_memory_cleanup():
    """Simple memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_tensor_health(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Simple tensor health check"""
    if tensor is None or tensor.numel() == 0:
        return False
    
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return False
    
    return True


# ==================== Fixed GradScaler ====================

class FixedGradScaler:
    """Fixed GradScaler with user-specified parameters"""
    
    def __init__(
        self,
        init_scale: float = 1024.0,
        growth_factor: float = 1.02,
        backoff_factor: float = 0.5,
        growth_interval: int = 100,
    ):
        self.scaler = GradScaler(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval
        )
        
        self.total_steps = 0
        self.successful_steps = 0
        self.overflow_steps = 0
        
        tqdm.write(f"🔧 Fixed GradScaler: scale={init_scale}, growth={growth_factor}, backoff={backoff_factor}")
    
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return self.scaler.scale(loss)
    
    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self.scaler.unscale_(optimizer)
    
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.total_steps += 1
        old_scale = self.scaler.get_scale()
        self.scaler.step(optimizer)
        new_scale = self.scaler.get_scale()
        
        if old_scale <= new_scale:
            self.successful_steps += 1
        else:
            self.overflow_steps += 1
    
    def update(self) -> None:
        self.scaler.update()
    
    def get_scale(self) -> float:
        return float(self.scaler.get_scale())
    
    def get_stats(self) -> Dict[str, float]:
        success_rate = self.successful_steps / max(self.total_steps, 1)
        return {
            'scale': self.get_scale(),
            'success_rate': success_rate,
            'total_steps': self.total_steps,
            'overflow_steps': self.overflow_steps
        }


# ==================== Simple Memory Monitor ====================

class SimpleMemoryMonitor:
    """Simplified memory monitor for large model"""
    
    def __init__(self):
        self.oom_count = 0
    
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    def log_oom(self):
        """Log OOM event"""
        self.oom_count += 1
        tqdm.write(f"💥 OOM Event #{self.oom_count}")
    
    def cleanup_if_needed(self, batch_idx: int):
        """Cleanup memory if needed"""
        if batch_idx % 100 == 0:
            safe_memory_cleanup()


# ==================== Main Trainer ====================

class LyroS6Trainer:
    """
    Simplified LYRO S6 Trainer for Large Model Only
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Force FP16
        tqdm.write("🎯 FP16 Training Enforced for Large Model")
        
        # Simple memory monitor
        self.memory_monitor = SimpleMemoryMonitor()
        
        # Initialize models
        self._initialize_models()
        
        # Setup flow matching
        self._setup_flow_matching()
        
        # Setup optimization
        self._setup_optimization()
        
        # Setup data
        self._setup_data()
        
        # Setup checkpoint dir
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup wandb
        if args.use_wandb and WANDB_AVAILABLE:
            self._setup_wandb()
        
        # Setup tokenizer
        self.tokenizer = LyroTokenizer(
            text_tokenizer_path=getattr(args, 'text_tokenizer_path', None),
            audio_vocab_size=1024,
            num_codebooks=8
        )
        
        # Best metrics
        self.best_metrics = {
            'val_loss': float('inf'),
            'train_loss': float('inf')
        }
        
        tqdm.write("✅ LYRO S6 Trainer Initialized (Large Model)")
        tqdm.write(f"📊 S6 Model: {sum(p.numel() for p in self.s6_model.parameters()):,} parameters")
        tqdm.write(f"💾 FP16 Enforced: True")
    
    def _initialize_models(self):
        """Initialize large models only"""
        
        # Large S6 U-Net model
        self.s6_model = create_lyro_s6_model(
            input_channels=16,  # Large DCAE latent channels
            model_size="large",
            hidden_dims=[256, 512, 768],
            s6_layers=[3, 4, 4],
            d_state=64,
            max_seq_len=2048,
        ).to(self.device)
        
        # Force FP16
        self.s6_model = self.s6_model.half()
        
        # Large DCAE model
        self.dcae_model = create_large_dcae_model(
            sample_rate=44100,
            latent_channels=16,
        ).to(self.device).half().eval()
        
        # Load DCAE checkpoint
        if self.args.dcae_checkpoint:
            try:
                dcae_ckpt = torch.load(self.args.dcae_checkpoint, map_location=self.device)
                if 'model_state_dict' in dcae_ckpt:
                    self.dcae_model.load_state_dict(dcae_ckpt['model_state_dict'])
                else:
                    self.dcae_model.load_state_dict(dcae_ckpt)
                tqdm.write(f"✅ Loaded Large DCAE from {self.args.dcae_checkpoint}")
            except Exception as e:
                tqdm.write(f"⚠️ Failed to load DCAE: {e}")
        
        tqdm.write("🧠 Large Models Initialized")
    
    def _setup_flow_matching(self):
        """Setup flow matching for large model"""
        
        # Flow config
        self.flow_config = FlowConfig()
        self.flow_config.apply_preset("standard")
        self.flow_config.latent_channels = 16  # Large model
        
        # Create flow matching
        self.flow_matching = create_s6_flow_matching(
            model=self.s6_model,
            config=self.flow_config,
        )
        
        tqdm.write("⚡ Flow Matching Setup Complete")
    
    def _setup_optimization(self):
        """Setup optimization for large model"""
        
        # AdamW optimizer
        self.optimizer = optim.AdamW(
            self.s6_model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay,
            eps=1e-8,
            fused=True if torch.cuda.is_available() else False
        )
        
        # Fixed GradScaler
        self.scaler = FixedGradScaler(
            init_scale=1024.0,
            growth_factor=1.02,
            backoff_factor=0.5,
            growth_interval=100
        )
        
        # Simple cosine scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=self.args.learning_rate * 0.01
        )
        
        tqdm.write("⚙️ Optimization Setup Complete")
    
    def _setup_data(self):
        """Setup datasets for large model"""
        
        # Fixed task ratios for simplicity
        task_ratios = {'SONG': 0.6, 'INST': 0.2, 'COVER': 0.2}
        
        # Training dataset
        self.train_dataset = LyroDataset(
            metadata_path=self.args.train_metadata,
            dataset_root=self.args.dataset_root,
            task_ratios=task_ratios,
            augmentation=True,
            use_processed=True,
            max_duration=self.args.max_audio_length / 44100
        )
        
        # Validation dataset
        self.val_dataset = LyroDataset(
            metadata_path=self.args.val_metadata,
            dataset_root=self.args.dataset_root,
            task_ratios=task_ratios,
            augmentation=False,
            use_processed=True,
            max_duration=self.args.max_audio_length / 44100
        )
        
        # Collator
        collator = LyroCollator(
            tokenizer=self.tokenizer,
            max_audio_length=self.args.max_audio_length,
            max_text_length=self.args.max_text_length,
            pad_to_multiple=256
        )
        
        # Data loaders
        num_workers = min(self.args.num_workers, 4)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collator,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            pin_memory=True,
            collate_fn=collator,
            persistent_workers=True if num_workers > 1 else False
        )
        
        tqdm.write(f"📚 Dataset: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")
    
    def _setup_wandb(self):
        """Setup wandb logging"""
        try:
            wandb.init(
                project="lyro-s6-large",
                name=f"large_s6_fp16",
                config={
                    'model_type': 'Large S6 UNet',
                    'latent_channels': 16,
                    'hidden_dims': [256, 512, 768],
                    's6_layers': [3, 4, 4],
                    'batch_size': self.args.batch_size,
                    'learning_rate': self.args.learning_rate,
                    'epochs': self.args.epochs,
                    'fp16_enforced': True,
                    'flow_matching': True,
                }
            )
            tqdm.write("📊 Wandb initialized")
        except Exception as e:
            tqdm.write(f"⚠️ Wandb failed: {e}")
    
    def _validate_batch(self, batch) -> bool:
        """Simple batch validation"""
        try:
            audio = batch['audio']
            return not (torch.isnan(audio).any() or torch.isinf(audio).any())
        except Exception:
            return False
    
    def _prepare_conditions(self, batch) -> Dict:
        """Simplified condition preparation"""
        conditions = {
            'task_token': [],
            'lyrics': batch.get('lyrics_tokens'),
            'style_prompt': []
        }
        
        # Task tokens
        for task_token in batch['task_tokens']:
            task_type = task_token.split('=')[1].rstrip('>')
            conditions['task_token'].append(
                TaskController.TASK_TOKENS.get(task_type, 0)
            )
        
        conditions['task_token'] = torch.tensor(
            conditions['task_token'], 
            device=self.device,
            dtype=torch.long
        )
        
        # Simple style prompts (genre encoding)
        style_prompts = []
        for genres in batch['genres']:
            # Simplified genre encoding
            style_vec = torch.zeros(512, dtype=torch.float16, device=self.device)
            for i, genre in enumerate(genres[:4]):
                if i < len(genres):
                    start_idx = (hash(genre) % 16) * 32
                    style_vec[start_idx:start_idx+32] = torch.randn(32) * 0.1
            style_prompts.append(style_vec)
        
        conditions['style_prompt'] = torch.stack(style_prompts)
        
        return conditions
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Training epoch for large model"""
        self.s6_model.train()
        self.dcae_model.eval()
        
        total_loss = 0.0
        successful_batches = 0
        batch_times = []
        
        # Progress bar
        pbar = tqdm(
            self.train_loader, 
            desc=f'Large S6 Epoch {epoch}',
            dynamic_ncols=True,
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            batch_start = time.time()
            
            try:
                # Memory cleanup
                self.memory_monitor.cleanup_if_needed(batch_idx)
                
                # Validate batch
                if not self._validate_batch(batch):
                    continue
                
                # Get audio data
                audio = batch['audio'].to(self.device, non_blocking=True).half()
                audio_lengths = batch['audio_lengths'].to(self.device, non_blocking=True)
                
                # Check tensor health
                if not check_tensor_health(audio, "input_audio"):
                    continue
                
                # DCAE encoding
                with torch.no_grad():
                    try:
                        latents, _ = self.dcae_model.encode(audio)
                        latents = latents.half()
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            self.memory_monitor.log_oom()
                            safe_memory_cleanup()
                            continue
                        else:
                            raise
                
                # Prepare conditions
                conditions = self._prepare_conditions(batch)
                
                # Forward pass with autocast
                with autocast():
                    loss = self.flow_matching.training_loss(latents, conditions)
                
                # Check loss health
                if not check_tensor_health(loss, "loss"):
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.s6_model.parameters(), 
                    self.args.grad_clip
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Track metrics
                total_loss += loss.item()
                successful_batches += 1
                batch_times.append(time.time() - batch_start)
                
                # Update progress bar
                avg_loss = total_loss / successful_batches
                avg_time = np.mean(batch_times[-10:])
                memory_usage = self.memory_monitor.get_memory_usage()
                scaler_stats = self.scaler.get_stats()
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'grad': f'{grad_norm:.2f}',
                    'scale': f'{scaler_stats["scale"]:.0f}',
                    'mem': f'{memory_usage:.1f}GB',
                    'time': f'{avg_time:.2f}s'
                })
                
                # Wandb logging
                if (self.args.use_wandb and WANDB_AVAILABLE and 
                    batch_idx % self.args.log_interval == 0):
                    
                    wandb.log({
                        'train/loss': safe_wandb_log(loss),
                        'train/grad_norm': safe_wandb_log(grad_norm),
                        'train/lr': safe_wandb_log(self.optimizer.param_groups[0]['lr']),
                        'train/memory_gb': safe_wandb_log(memory_usage),
                        'train/scaler_scale': safe_wandb_log(scaler_stats['scale']),
                        'train/scaler_success_rate': safe_wandb_log(scaler_stats['success_rate']),
                        'step': epoch * len(self.train_loader) + batch_idx
                    })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.memory_monitor.log_oom()
                    safe_memory_cleanup()
                    continue
                else:
                    tqdm.write(f"❌ Runtime error: {e}")
                    continue
            except Exception as e:
                tqdm.write(f"❌ Batch {batch_idx} failed: {e}")
                continue
        
        # Scheduler step
        self.scheduler.step()
        
        # Epoch summary
        avg_loss = total_loss / max(successful_batches, 1)
        avg_time = np.mean(batch_times) if batch_times else 0
        success_rate = successful_batches / len(self.train_loader)
        
        tqdm.write(f"📈 Epoch {epoch} Summary:")
        tqdm.write(f"   - Avg Loss: {avg_loss:.4f}")
        tqdm.write(f"   - Success Rate: {success_rate:.2%}")
        tqdm.write(f"   - Avg Batch Time: {avg_time:.2f}s")
        
        return {
            'loss': avg_loss,
            'success_rate': success_rate,
            'avg_batch_time': avg_time,
            'scaler_stats': self.scaler.get_stats()
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validation for large model"""
        self.s6_model.eval()
        self.dcae_model.eval()
        
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            val_pbar = tqdm(
                self.val_loader, 
                desc='Validation',
                dynamic_ncols=True,
                leave=False
            )
            
            for batch_idx, batch in enumerate(val_pbar):
                if batch_idx >= 15:  # Limit validation batches
                    break
                
                try:
                    # Validate batch
                    if not self._validate_batch(batch):
                        continue
                    
                    # Get audio data
                    audio = batch['audio'].to(self.device, non_blocking=True).half()
                    
                    # Check tensor health
                    if not check_tensor_health(audio):
                        continue
                    
                    # DCAE encoding
                    latents, _ = self.dcae_model.encode(audio)
                    latents = latents.half()
                    
                    # Prepare conditions
                    conditions = self._prepare_conditions(batch)
                    
                    # Forward pass
                    with autocast():
                        loss = self.flow_matching.training_loss(latents, conditions)
                    
                    if check_tensor_health(loss):
                        total_loss += loss.item()
                        batch_count += 1
                    
                    # Update progress bar
                    if batch_count > 0:
                        val_pbar.set_postfix({
                            'val_loss': f'{total_loss / batch_count:.4f}'
                        })
                
                except Exception as e:
                    tqdm.write(f"⚠️ Val batch {batch_idx} failed: {e}")
                    continue
        
        avg_val_loss = total_loss / max(batch_count, 1)
        
        tqdm.write(f"✅ Validation: Loss={avg_val_loss:.4f} ({batch_count} batches)")
        
        return {
            'loss': avg_val_loss,
            'batch_count': batch_count
        }
    
    def generate_samples(self, epoch: int, num_samples: int = 2):
        """Generate samples for large model"""
        self.s6_model.eval()
        self.dcae_model.eval()
        
        tasks = ['SONG', 'INST']
        generated_samples = []
        
        with torch.no_grad():
            for task in tasks:
                try:
                    # Prepare conditions
                    conditions = {
                        'task_token': torch.tensor([TaskController.TASK_TOKENS[task]]).to(self.device),
                        'lyrics': None,
                        'style_prompt': torch.randn(1, 512, device=self.device, dtype=torch.float16),
                    }
                    
                    # Add lyrics for SONG task
                    if task == 'SONG':
                        sample_lyrics = "Walking down the street tonight"
                        lyrics_tokens = self.tokenizer.encode_text(sample_lyrics)
                        conditions['lyrics'] = torch.tensor([lyrics_tokens]).to(self.device)
                    
                    # Generate latents
                    shape = (1, 16, 64, 64)  # Large model latent shape
                    
                    start_time = time.time()
                    generated_latent, _ = self.flow_matching.generate(
                        shape=shape,
                        conditions=conditions,
                        num_steps=self.flow_config.flow_steps,
                        cfg_scale=1.5
                    )
                    generation_time = time.time() - start_time
                    
                    # Decode to audio
                    dummy_skip_features = [torch.zeros_like(generated_latent) for _ in range(3)]
                    audio = self.dcae_model.decode(generated_latent, dummy_skip_features)
                    
                    generated_samples.append({
                        'task': task,
                        'audio': audio.cpu(),
                        'generation_time': generation_time
                    })
                    
                except Exception as e:
                    tqdm.write(f"❌ Sample generation failed for {task}: {e}")
                    continue
        
        # Save samples
        if self.args.save_samples and generated_samples:
            sample_dir = self.checkpoint_dir / f'samples_epoch_{epoch}'
            sample_dir.mkdir(exist_ok=True)
            
            for i, sample in enumerate(generated_samples):
                try:
                    # Save audio
                    audio_path = sample_dir / f'{sample["task"]}_{i}.wav'
                    torchaudio.save(
                        audio_path,
                        sample['audio'][0],
                        sample_rate=44100
                    )
                    
                    # Wandb logging
                    if self.args.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            f'samples/{sample["task"]}_{i}': wandb.Audio(
                                sample['audio'][0].numpy(),
                                sample_rate=44100,
                                caption=f'Large S6 {sample["task"]} sample (Epoch {epoch})'
                            ),
                            f'generation_time/{sample["task"]}': sample['generation_time']
                        })
                        
                except Exception as e:
                    tqdm.write(f"❌ Sample saving failed: {e}")
        
        tqdm.write(f"🎼 Generated {len(generated_samples)} samples")
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any], is_best: bool = False):
        """Save checkpoint for large model"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.s6_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.scaler.state_dict(),
            'args': self.args,
            'metrics': metrics,
            'model_config': {
                'type': 'Large S6 UNet',
                'input_channels': 16,
                'hidden_dims': [256, 512, 768],
                's6_layers': [3, 4, 4],
                'd_state': 64,
                'fp16_enforced': True,
                'flow_matching': True
            }
        }
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'large_s6_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Best model
        if is_best:
            best_path = self.checkpoint_dir / 'large_s6_best.pt'
            torch.save(checkpoint, best_path)
            tqdm.write(f"🏆 New best Large S6 model saved!")
        
        tqdm.write(f"💾 Checkpoint saved: epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.s6_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint:
            self.scaler.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint['epoch']
    
    def train(self):
        """Main training loop for large model"""
        start_epoch = 0
        
        # Resume from checkpoint
        if self.args.resume:
            start_epoch = self.load_checkpoint(self.args.resume)
            tqdm.write(f"🔄 Resumed from epoch {start_epoch}")
        
        tqdm.write("\n" + "="*60)
        tqdm.write("🚀 Large S6 LYRO Training Started")
        tqdm.write(f"🧠 Model: Large S6 UNet + Flow Matching")
        tqdm.write(f"🎯 FP16 Enforced + Fixed GradScaler")
        tqdm.write(f"⚡ DCAE Integration Ready")
        tqdm.write("="*60 + "\n")
        
        # Training loop
        for epoch in range(start_epoch, self.args.epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate(epoch)
            
            # Check if best model
            is_best = val_metrics['loss'] < self.best_metrics['val_loss']
            if is_best:
                self.best_metrics['val_loss'] = val_metrics['loss']
                self.best_metrics['train_loss'] = train_metrics['loss']
            
            # Wandb logging
            if self.args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch/train_loss': safe_wandb_log(train_metrics['loss']),
                    'epoch/val_loss': safe_wandb_log(val_metrics['loss']),
                    'epoch/success_rate': safe_wandb_log(train_metrics['success_rate']),
                    'epoch/scaler_success_rate': safe_wandb_log(train_metrics['scaler_stats']['success_rate']),
                    'epoch': epoch
                })
            
            # Generate samples
            if epoch % self.args.sample_interval == 0:
                tqdm.write(f"🎼 Generating samples...")
                self.generate_samples(epoch)
            
            # Save checkpoint
            if epoch % self.args.save_interval == 0 or epoch == self.args.epochs - 1:
                all_metrics = {**train_metrics, **val_metrics}
                self.save_checkpoint(epoch, all_metrics, is_best)
        
        tqdm.write("\n" + "="*60)
        tqdm.write("🎉 Large S6 LYRO Training Completed!")
        tqdm.write(f"🏆 Best Val Loss: {self.best_metrics['val_loss']:.4f}")
        tqdm.write(f"📊 Model ready for deployment!")
        tqdm.write("="*60)


def main():
    """Main function for large S6 training"""
    parser = argparse.ArgumentParser(description='LYRO Large S6 Training')
    
    # Data related
    parser.add_argument('--train_metadata', type=str, default='dataset/metadata/train_metadata.jsonl')
    parser.add_argument('--val_metadata', type=str, default='dataset/metadata/val_metadata.jsonl')
    parser.add_argument('--dataset_root', type=str, default='dataset/')
    
    # Model related
    parser.add_argument('--dcae_checkpoint', type=str, required=True)
    parser.add_argument('--text_tokenizer_path', type=str, default=None)
    
    # Training related
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)  # Large model uses smaller batch
    parser.add_argument('--learning_rate', type=float, default=8e-5)  # Lower LR for large model
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Data processing
    parser.add_argument('--max_audio_length', type=int, default=441000)  # 10 seconds
    parser.add_argument('--max_text_length', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Logging and saving
    parser.add_argument('--checkpoint_dir', type=str, default='ssm/checkpoints_large_s6')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--sample_interval', type=int, default=20)
    parser.add_argument('--save_samples', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # CUDA check
    if not torch.cuda.is_available():
        print("❌ CUDA required for Large S6 training!")
        return
    
    tqdm.write("🚀 LYRO Large S6 Training")
    tqdm.write(f"⚡ Available GPUs: {torch.cuda.device_count()}")
    tqdm.write(f"🧠 Model: Large S6 UNet (16 latent channels)")
    tqdm.write(f"🎯 FP16 Enforced: True")
    tqdm.write(f"⚙️ Fixed GradScaler: True")
    tqdm.write(f"⚡ Flow Matching: True")
    
    try:
        trainer = LyroS6Trainer(args)
        trainer.train()
        tqdm.write("🎉 Large S6 training completed successfully!")
        
    except Exception as e:
        tqdm.write(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()