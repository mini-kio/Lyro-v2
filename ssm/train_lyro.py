# lyro/ssm/train_lyro.py
"""
LYRO S6 Training Script
Flow Matching 기반 S6 음악 생성 모델 학습 (Mixed Precision + Enhanced Memory Efficiency)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast  # Mixed Precision support
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional, List
import math
import torchaudio
import warnings
import gc
import psutil
from contextlib import nullcontext

# Wandb import handling
WANDB_AVAILABLE = False
wandb = None

try:
    import importlib
    _wandb = importlib.import_module('wandb')
    WANDB_AVAILABLE = True
    wandb = _wandb
except ImportError:
    class MockWandb:
        @staticmethod
        def init(*args, **kwargs):
            pass
        @staticmethod
        def log(*args, **kwargs):
            pass
        class Audio:
            def __init__(self, *args, **kwargs):
                pass
    wandb = MockWandb()
    print("Warning: wandb not available. Using mock logging.")

# LYRO 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssm.model import LyroS6UNet, TaskController, EOSTokenHandler, create_lyro_s6_model, benchmark_s6_model
from ssm.flow_matching import LyroFlowMatching, FlowConfig, create_flow_matching
from dcae.model import create_cqt_ssm_dcae
from dataset.dataset import LyroDataset, LyroCollator
from dataset.tokenizer import LyroTokenizer

# 경고 억제
warnings.filterwarnings("ignore")


class AdvancedMemoryMonitor:
    """S6 특화 메모리 모니터링"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._last_clear = 0
        self.memory_history = []
        self.peak_memory_usage = 0
        self.oom_events = 0
        
    def get_memory_stats(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            
            self.peak_memory_usage = max(self.peak_memory_usage, allocated)
            self.memory_history.append(allocated)
            
            if len(self.memory_history) > 100:
                self.memory_history.pop(0)
                
            return {
                'gpu_allocated_gb': allocated,
                'gpu_reserved_gb': reserved,
                'gpu_utilization': allocated / (reserved + 1e-8) * 100,
                'peak_memory_gb': self.peak_memory_usage,
                'avg_memory_gb': np.mean(self.memory_history) if self.memory_history else 0.0
            }
        return {}
    
    def should_clear_cache(self):
        current_time = torch.cuda.Event(enable_timing=True)
        current_time.record()
        
        if current_time.elapsed_time(self._last_clear) > 30000:  # 30초마다
            self._last_clear = current_time
            return True
        return False
    
    def intelligent_clear(self):
        """S6 모델을 위한 지능적 메모리 관리"""
        gc.collect()
        
        if torch.cuda.is_available():
            # 메모리 압박 상황 감지
            if len(self.memory_history) > 5:
                recent_avg = np.mean(self.memory_history[-5:])
                if recent_avg > 8.0:  # 8GB 이상 사용시
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, 'synchronize'):
                        torch.cuda.synchronize()
    
    def log_oom_event(self):
        """OOM 이벤트 기록"""
        self.oom_events += 1
        print(f"⚠️ OOM Event #{self.oom_events} detected")


class LyroS6Trainer:
    """
    LYRO S6 학습 관리 클래스
    
    S6 (Mamba-2) + Flow Matching을 사용하여 고효율 음악 생성 모델 학습
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stage = args.stage
        
        # Mixed Precision 설정
        self.use_mixed_precision = getattr(args, 'use_mixed_precision', True)
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # 메모리 모니터링
        self.memory_monitor = AdvancedMemoryMonitor()
        
        # 모델 초기화
        self._initialize_models()
        
        # Flow Matching 설정 (S6 최적화)
        self.flow_config = FlowConfig()
        self.flow_config.apply_preset("standard")  # S6에 최적화된 설정
        
        self.flow_matching = create_flow_matching(
            model=self.s6_model,
            config=self.flow_config,
            use_torch_compile=getattr(args, 'use_torch_compile', True),
            compile_mode=getattr(args, 'compile_mode', 'default')
        )
        
        # 옵티마이저 및 스케줄러
        self._setup_optimization()
        
        # 데이터셋 및 로더
        self._setup_data()
        
        # 체크포인트 디렉토리
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Wandb 초기화
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="lyro-s6-ssm",
                name=f"{args.exp_name}_stage_{args.stage}",
                config={
                    **vars(args),
                    'model_type': 'S6 (Mamba-2)',
                    'architecture': 'S6-UNet + Flow Matching',
                    'state_space_type': 'State Space Dual (SSD)',
                    'memory_efficient': True,
                    'chunk_size': getattr(args, 'chunk_size', 256)
                }
            )
        elif args.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available. Continuing without logging.")
            
        # Adaptive Weight Manager (Stage C 이상)
        if self.stage in ['C', 'D', 'E']:
            self._setup_adaptive_weights()
            
        # 토크나이저
        self.tokenizer = LyroTokenizer(
            text_tokenizer_path=args.text_tokenizer_path,
            audio_vocab_size=1024,
            num_codebooks=8
        )
        
        # 성능 기록
        self.best_metrics = {
            'val_loss': float('inf'),
            'train_loss': float('inf'),
            'generation_quality': 0.0
        }
        
        print(f"🚀 S6-based LYRO Trainer Initialized")
        print(f"📊 Model: S6-UNet with {sum(p.numel() for p in self.s6_model.parameters()):,} parameters")
        print(f"🧠 Stage: {self.stage}")
        print(f"💾 Memory Efficient: {args.chunk_size} chunk size")
        print(f"⚡ Mixed Precision: {self.use_mixed_precision}")
        
    def _initialize_models(self):
        """S6 모델 초기화"""
        
        # S6 + U-Net 모델 초기화
        self.s6_model = create_lyro_s6_model(
            input_channels=8,
            model_size=getattr(self.args, 'model_size', 'base'),
            hidden_dims=[128, 128, 256, 256, 512],
            s6_layers=[2, 2, 3, 3, 4],
            d_state=128,
            d_head=64,
            max_seq_len=self.args.max_seq_len,
            use_torch_compile=getattr(self.args, 'use_torch_compile', True),
            use_mixed_precision=getattr(self.args, 'use_mixed_precision', True),
            chunk_size=getattr(self.args, 'chunk_size', 256),
            use_mem_eff_path=True,
        ).to(self.device)
        
        # DCAE 모델 (인코딩용)
        self.dcae_model = create_cqt_ssm_dcae(
            sample_rate=44100,
            latent_channels=8,
            model_size="base",
            use_torch_compile=getattr(self.args, 'use_torch_compile', True),
            use_mixed_precision=getattr(self.args, 'use_mixed_precision', True),
            compile_mode=getattr(self.args, 'compile_mode', 'default')
        ).to(self.device)
        
        # DCAE 체크포인트 로드
        if self.args.dcae_checkpoint:
            try:
                dcae_ckpt = torch.load(self.args.dcae_checkpoint, map_location=self.device)
                if 'model_state_dict' in dcae_ckpt:
                    self.dcae_model.load_state_dict(dcae_ckpt['model_state_dict'])
                else:
                    self.dcae_model.load_state_dict(dcae_ckpt)
                self.dcae_model.eval()
                print(f"✅ Loaded DCAE from {self.args.dcae_checkpoint}")
            except Exception as e:
                print(f"⚠️ Failed to load DCAE checkpoint: {e}")
                
        # 이전 스테이지 체크포인트 로드
        if self.args.prev_stage_checkpoint:
            self._load_prev_stage(self.args.prev_stage_checkpoint)
            
        # S6 모델 벤치마크
        if torch.cuda.is_available():
            try:
                benchmark_results = benchmark_s6_model(
                    self.s6_model, 
                    batch_size=1, 
                    seq_len=1024,
                    device=str(self.device)
                )
                print(f"📈 S6 Model Benchmark:")
                print(f"  Forward Time: {benchmark_results['avg_forward_time']:.4f}s")
                print(f"  Throughput: {benchmark_results['throughput_tokens_per_sec']:.0f} tokens/s")
                print(f"  Memory: {benchmark_results['memory_allocated_gb']:.2f}GB")
            except Exception as e:
                print(f"⚠️ Benchmark failed: {e}")
                
    def _setup_optimization(self):
        """S6 최적화된 옵티마이저 설정"""
        
        # Stage별 학습률 (S6 최적화)
        stage_lrs = {
            'B': 8e-4,  # S6는 더 높은 학습률 허용
            'C': 6e-4,
            'D': 4e-4,
            'E': 2e-4
        }
        
        lr = self.args.learning_rate or stage_lrs.get(self.stage, 4e-4)
        
        # AdamW with S6 optimized parameters
        self.optimizer = optim.AdamW(
            self.s6_model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),  # S6에 더 적합한 베타값
            weight_decay=self.args.weight_decay,
            eps=1e-8,
            fused=True if torch.cuda.is_available() else False  # Fused AdamW
        )
        
        # Enhanced scheduler for S6
        if self.stage == 'E':
            # Cosine Annealing with Warm Restarts
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.args.epochs // 4,
                T_mult=1,
                eta_min=lr * 0.01
            )
        else:
            # Warmup + Cosine Decay (S6에 더 효과적)
            warmup_steps = self.args.warmup_steps
            total_steps = self.args.epochs * self.args.steps_per_epoch
            
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
                    return 0.5 * (1.0 + math.cos(math.pi * progress))
            
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            
    def _setup_data(self):
        """S6 최적화된 데이터 설정"""
        
        # Stage별 태스크 비율 (S6 성능 고려)
        stage_task_ratios = {
            'B': {'SONG': 1.0, 'INST': 0.0, 'COVER': 0.0},
            'C': {'SONG': 0.7, 'INST': 0.3, 'COVER': 0.0},
            'D': {'SONG': 0.5, 'INST': 0.2, 'COVER': 0.3},
            'E': {'SONG': 0.6, 'INST': 0.15, 'COVER': 0.25}
        }
        
        task_ratios = stage_task_ratios.get(self.stage, {'SONG': 0.7, 'INST': 0.15, 'COVER': 0.15})
        
        # 데이터셋 초기화
        train_metadata = self.args.train_metadata
        if self.stage == 'E' and self.args.gold_metadata:
            train_metadata = self.args.gold_metadata
            
        self.train_dataset = LyroDataset(
            metadata_path=train_metadata,
            dataset_root=self.args.dataset_root,
            task_ratios=task_ratios,
            augmentation=True,
            use_processed=True,
            max_duration=self.args.max_audio_length / 44100  # 초 단위로 변환
        )
        
        self.val_dataset = LyroDataset(
            metadata_path=self.args.val_metadata,
            dataset_root=self.args.dataset_root,
            task_ratios=task_ratios,
            augmentation=False,
            use_processed=True,
            max_duration=self.args.max_audio_length / 44100
        )
        
        # S6 최적화된 Collator
        collator = LyroCollator(
            tokenizer=self.tokenizer,
            max_audio_length=self.args.max_audio_length,
            max_text_length=self.args.max_text_length,
            pad_to_multiple=getattr(self.args, 'chunk_size', 256)  # S6 chunk size로 정렬
        )
        
        # 데이터 로더 (S6 메모리 효율성 고려)
        num_workers = min(self.args.num_workers, 6)  # S6는 메모리 효율적이므로 더 많은 워커 허용
        
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
        
        # Steps per epoch 계산
        self.args.steps_per_epoch = len(self.train_loader)
        
        print(f"📚 Dataset loaded: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}")
        print(f"🔧 DataLoader: {num_workers} workers, batch_size={self.args.batch_size}")
        
    def _setup_adaptive_weights(self):
        """S6 최적화된 적응적 가중치 시스템"""
        self.task_weights = {
            'SONG': 0.5,
            'INST': 0.3,
            'COVER': 0.2
        }
        
        self.task_grad_norms = {
            'SONG': [],
            'INST': [],
            'COVER': []
        }
        
        self.weight_update_interval = 100
        
    def _validate_batch(self, batch):
        """배치 데이터 검증 (S6 특화)"""
        try:
            audio = batch['audio']
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                print("⚠️ Invalid audio data detected")
                return False
                
            if audio.shape[1] != 2:
                print(f"⚠️ Expected stereo audio, got {audio.shape[1]} channels")
                return False
            
            # S6 시퀀스 길이 검증
            seq_len = audio.shape[-1]
            chunk_size = getattr(self.args, 'chunk_size', 256)
            if seq_len < chunk_size:
                print(f"⚠️ Sequence too short for S6: {seq_len} < {chunk_size}")
                return False
                
            return True
        except Exception as e:
            print(f"❌ Batch validation error: {e}")
            return False
    
    def train_epoch(self, epoch):
        """S6 최적화된 한 에폭 학습"""
        self.s6_model.train()
        self.dcae_model.eval()
        
        total_loss = 0
        task_losses = {'SONG': 0, 'INST': 0, 'COVER': 0}
        task_counts = {'SONG': 0, 'INST': 0, 'COVER': 0}
        successful_batches = 0
        
        # S6 성능 메트릭
        s6_metrics = {
            'memory_efficiency': [],
            'processing_speed': [],
            'gradient_stability': []
        }
        
        pbar = tqdm(self.train_loader, desc=f'S6 Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 메모리 관리 (S6 최적화)
                if batch_idx % 20 == 0:
                    self.memory_monitor.intelligent_clear()
                
                # 데이터 검증
                if not self._validate_batch(batch):
                    continue
                    
                # 데이터 준비
                audio = batch['audio'].to(self.device, non_blocking=True)
                audio_lengths = batch['audio_lengths'].to(self.device, non_blocking=True)
                task_tokens = batch['task_tokens']
                
                # DCAE 인코딩 (메모리 효율적)
                with torch.no_grad():
                    try:
                        latents, _ = self.dcae_model.encode(audio)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            self.memory_monitor.log_oom_event()
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise
                
                # 조건 준비
                conditions = self._prepare_conditions(batch)
                
                # S6 Forward Pass with Mixed Precision
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                
                if self.use_mixed_precision:
                    with autocast():
                        # S6 Flow Matching 학습
                        loss = self.flow_matching.training_loss(latents, conditions)
                else:
                    # S6 Flow Matching 학습
                    loss = self.flow_matching.training_loss(latents, conditions)
                
                end_time.record()
                torch.cuda.synchronize()
                
                # 성능 메트릭 기록
                processing_time = start_time.elapsed_time(end_time)
                s6_metrics['processing_speed'].append(processing_time)
                
                # Task별 손실 기록
                for i, task_token in enumerate(task_tokens):
                    task_type = task_token.split('=')[1].rstrip('>')
                    if task_type in task_losses:
                        task_losses[task_type] += loss.item()
                        task_counts[task_type] += 1
                
                # S6 Backward pass with enhanced gradient handling
                if self.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                    
                    # Enhanced gradient clipping for S6
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.s6_model.parameters(), 
                        self.args.grad_clip
                    )
                    
                    # Gradient stability check for S6
                    if torch.isfinite(grad_norm):
                        s6_metrics['gradient_stability'].append(grad_norm.item())
                        self.scaler.step(self.optimizer)
                    else:
                        print(f"⚠️ Unstable gradients detected at step {batch_idx}")
                        
                    self.scaler.update()
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.s6_model.parameters(), 
                        self.args.grad_clip
                    )
                    s6_metrics['gradient_stability'].append(grad_norm.item())
                    
                    self.optimizer.step()
                
                self.scheduler.step()
                
                # 메모리 효율성 기록
                mem_stats = self.memory_monitor.get_memory_stats()
                s6_metrics['memory_efficiency'].append(mem_stats.get('gpu_allocated_gb', 0))
                
                # Adaptive weight update (S6 특화)
                if hasattr(self, 'task_weights') and batch_idx % self.weight_update_interval == 0:
                    self._update_adaptive_weights_s6(grad_norm, s6_metrics)
                    
                # 손실 기록
                total_loss += loss.item()
                successful_batches += 1
                
                # Progress bar 업데이트 (S6 메트릭 포함)
                avg_speed = np.mean(s6_metrics['processing_speed'][-10:]) if s6_metrics['processing_speed'] else 0
                avg_memory = np.mean(s6_metrics['memory_efficiency'][-10:]) if s6_metrics['memory_efficiency'] else 0
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'grad': f'{grad_norm:.3f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'mem': f'{avg_memory:.1f}GB',
                    'speed': f'{avg_speed:.1f}ms'
                })
                
                # Enhanced Wandb 로깅
                if (self.args.use_wandb and WANDB_AVAILABLE and 
                    batch_idx % self.args.log_interval == 0):
                    
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/grad_norm': grad_norm.item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/processing_speed_ms': avg_speed,
                        'train/memory_efficiency_gb': avg_memory,
                        'train/s6_chunk_size': getattr(self.args, 'chunk_size', 256),
                        'step': epoch * len(self.train_loader) + batch_idx
                    }
                    
                    # Task별 손실 로깅
                    for task in ['SONG', 'INST', 'COVER']:
                        if task_counts[task] > 0:
                            log_dict[f'train/loss_{task}'] = task_losses[task] / task_counts[task]
                    
                    # S6 특화 메트릭
                    if s6_metrics['gradient_stability']:
                        log_dict['train/s6_gradient_stability'] = np.std(s6_metrics['gradient_stability'][-50:])
                    
                    wandb.log(log_dict)
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.memory_monitor.log_oom_event()
                    print(f"💥 OOM at batch {batch_idx}, clearing cache...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"❌ Runtime error at batch {batch_idx}: {e}")
                    continue
                    
            except Exception as e:
                print(f"❌ Error processing batch {batch_idx}: {e}")
                continue
                
        # 에폭 평균 및 S6 성능 요약
        avg_loss = total_loss / max(successful_batches, 1)
        
        s6_performance = {
            'avg_processing_speed': np.mean(s6_metrics['processing_speed']) if s6_metrics['processing_speed'] else 0,
            'avg_memory_usage': np.mean(s6_metrics['memory_efficiency']) if s6_metrics['memory_efficiency'] else 0,
            'gradient_stability': np.std(s6_metrics['gradient_stability']) if s6_metrics['gradient_stability'] else 0,
            'successful_batches': successful_batches,
            'total_batches': len(self.train_loader)
        }
        
        return avg_loss, task_losses, task_counts, s6_performance
    
    def _update_adaptive_weights_s6(self, grad_norm, s6_metrics):
        """S6 특화 적응적 가중치 업데이트"""
        
        # S6의 메모리 효율성과 처리 속도를 고려한 가중치 조정
        recent_memory = np.mean(s6_metrics['memory_efficiency'][-10:]) if s6_metrics['memory_efficiency'] else 0
        recent_speed = np.mean(s6_metrics['processing_speed'][-10:]) if s6_metrics['processing_speed'] else 0
        
        # 메모리 사용량이 높으면 복잡한 task 비중 감소
        if recent_memory > 8.0:  # 8GB 이상
            complexity_factor = 0.95
        else:
            complexity_factor = 1.05
            
        # 처리 속도가 느리면 시퀀스 길이가 긴 task 비중 조정
        if recent_speed > 100:  # 100ms 이상
            sequence_factor = 0.98
        else:
            sequence_factor = 1.02
            
        # Task별 가중치 조정
        self.task_weights['SONG'] *= complexity_factor  # 가장 복잡
        self.task_weights['INST'] *= sequence_factor    # 중간 복잡도
        self.task_weights['COVER'] *= (complexity_factor + sequence_factor) / 2  # 가장 복잡
        
        # 정규화
        total_weight = sum(self.task_weights.values())
        for task in self.task_weights:
            self.task_weights[task] /= total_weight
    
    def _prepare_conditions(self, batch):
        """S6 최적화된 조건 정보 준비"""
        conditions = {
            'task_token': [],
            'lyrics': batch.get('lyrics_tokens'),
            'style_prompt': [],
            'icl_reference': None
        }
        
        # Task token 변환
        for task_token in batch['task_tokens']:
            task_type = task_token.split('=')[1].rstrip('>')
            if task_type in TaskController.TASK_TOKENS:
                conditions['task_token'].append(
                    TaskController.TASK_TOKENS[task_type]
                )
            else:
                conditions['task_token'].append(0)
                
        conditions['task_token'] = torch.tensor(
            conditions['task_token'], 
            device=self.device,
            dtype=torch.long
        )
        
        # Enhanced style prompt for S6
        style_prompts = []
        for genres in batch['genres']:
            genre_vec = self._encode_genres_s6(genres)
            style_prompts.append(genre_vec)
        conditions['style_prompt'] = torch.stack(style_prompts).to(self.device)
        
        # S6 최적화된 ICL reference
        if 'reference_audios' in batch and batch['reference_audios'] is not None:
            ref_audios = batch['reference_audios'].to(self.device)
            with torch.no_grad():
                conditions['icl_reference'], _ = self.dcae_model.encode(ref_audios)
                
        return conditions
    
    def _encode_genres_s6(self, genres: List[str]) -> torch.Tensor:
        """S6 최적화된 장르 인코딩"""
        # Enhanced genre mapping for S6
        genre_map = {
            'Pop': 0, 'Rock': 1, 'Jazz': 2, 'Classical': 3,
            'Blues': 4, 'Electronic': 5, 'Hip-Hop': 6, 'Folk': 7,
            'Country': 8, 'R&B': 9, 'Metal': 10, 'Indie': 11,
            'Dance': 12, 'Reggae': 13, 'Funk': 14, 'Soul': 15,
            'Unknown': 16
        }
        
        # S6에 최적화된 더 큰 벡터 공간
        vec = torch.zeros(512, dtype=torch.float32)
        
        for i, genre in enumerate(genres[:4]):  # 최대 4개 장르
            if genre in genre_map:
                idx = genre_map[genre]
                start = idx * 30
                end = min(start + 30, 512)
                if start < 512:
                    # Learnable genre embedding space
                    vec[start:end] = torch.randn(end - start) * 0.1
                    vec[start] = 1.0  # Primary indicator
                
        return vec
    
    def validate(self, epoch):
        """S6 최적화된 검증"""
        self.s6_model.eval()
        self.dcae_model.eval()
        
        total_loss = 0
        task_losses = {'SONG': 0, 'INST': 0, 'COVER': 0}
        task_counts = {'SONG': 0, 'INST': 0, 'COVER': 0}
        batch_count = 0
        
        # S6 검증 메트릭
        s6_val_metrics = {
            'inference_speed': [],
            'memory_usage': [],
            'sequence_lengths': []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='S6 Validation')):
                if batch_idx >= 20:  # S6 효율성으로 더 많은 배치 처리 가능
                    break
                
                try:
                    # 데이터 준비
                    audio = batch['audio'].to(self.device, non_blocking=True)
                    task_tokens = batch['task_tokens']
                    
                    # S6 inference timing
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    
                    start_time.record()
                    
                    # DCAE 인코딩
                    latents, _ = self.dcae_model.encode(audio)
                    
                    # 조건 준비
                    conditions = self._prepare_conditions(batch)
                    
                    # S6 Flow Matching 검증
                    if self.use_mixed_precision:
                        with autocast():
                            loss = self.flow_matching.training_loss(latents, conditions)
                    else:
                        loss = self.flow_matching.training_loss(latents, conditions)
                    
                    end_time.record()
                    torch.cuda.synchronize()
                    
                    # 성능 메트릭 기록
                    inference_time = start_time.elapsed_time(end_time)
                    s6_val_metrics['inference_speed'].append(inference_time)
                    s6_val_metrics['sequence_lengths'].append(latents.shape[-1])
                    
                    mem_stats = self.memory_monitor.get_memory_stats()
                    s6_val_metrics['memory_usage'].append(mem_stats.get('gpu_allocated_gb', 0))
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # Task별 손실 기록
                    for i, task_token in enumerate(task_tokens):
                        task_type = task_token.split('=')[1].rstrip('>')
                        if task_type in task_losses:
                            task_losses[task_type] += loss.item()
                            task_counts[task_type] += 1
                
                except Exception as e:
                    print(f"⚠️ Validation error at batch {batch_idx}: {e}")
                    continue
        
        # 평균 계산
        avg_loss = total_loss / max(batch_count, 1)
        
        # Task별 평균
        task_avg_losses = {}
        for task in ['SONG', 'INST', 'COVER']:
            if task_counts[task] > 0:
                task_avg_losses[task] = task_losses[task] / task_counts[task]
            else:
                task_avg_losses[task] = 0.0
        
        # S6 검증 성능 요약
        s6_val_performance = {
            'avg_inference_speed': np.mean(s6_val_metrics['inference_speed']) if s6_val_metrics['inference_speed'] else 0,
            'avg_memory_usage': np.mean(s6_val_metrics['memory_usage']) if s6_val_metrics['memory_usage'] else 0,
            'avg_sequence_length': np.mean(s6_val_metrics['sequence_lengths']) if s6_val_metrics['sequence_lengths'] else 0,
            'total_batches': batch_count
        }
        
        # Enhanced Wandb 로깅
        if self.args.use_wandb and WANDB_AVAILABLE:
            log_dict = {
                'val/loss': avg_loss,
                'val/s6_inference_speed': s6_val_performance['avg_inference_speed'],
                'val/s6_memory_usage': s6_val_performance['avg_memory_usage'],
                'val/s6_sequence_length': s6_val_performance['avg_sequence_length'],
                'epoch': epoch
            }
            
            for task, avg in task_avg_losses.items():
                log_dict[f'val/loss_{task}'] = avg
                
            wandb.log(log_dict)
            
        return avg_loss, task_avg_losses, s6_val_performance
    
    def generate_samples(self, epoch, num_samples=4):
        """S6 최적화된 샘플 생성"""
        self.s6_model.eval()
        self.dcae_model.eval()
        
        # S6에 최적화된 태스크 선택
        tasks = ['SONG', 'INST']
        if self.stage in ['D', 'E']:
            tasks.append('COVER')
            
        generated_samples = []
        generation_metrics = []
        
        with torch.no_grad():
            for task in tasks:
                try:
                    # S6 generation timing
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    
                    start_time.record()
                    
                    # S6 최적화된 조건 준비
                    conditions = {
                        'task_token': torch.tensor([TaskController.TASK_TOKENS[task]]).to(self.device),
                        'lyrics': None,
                        'style_prompt': torch.randn(1, 512).to(self.device),
                        'icl_reference': None
                    }
                    
                    # 가사 추가 (SONG 태스크)
                    if task == 'SONG':
                        sample_lyrics = "Walking down the street tonight\nEverything feels so right"
                        lyrics_tokens = self.tokenizer.encode_text(sample_lyrics)
                        conditions['lyrics'] = torch.tensor([lyrics_tokens]).to(self.device)
                    
                    # S6 Flow Matching 생성 (최적화된 스텝 수)
                    chunk_size = getattr(self.args, 'chunk_size', 256)
                    shape = (1, 8, chunk_size * 4)  # S6 chunk size의 4배
                    
                    generated_latent, trajectory = self.flow_matching.generate(
                        shape=shape,
                        conditions=conditions,
                        num_steps=self.flow_config.flow_steps,
                        cfg_scale=1.5
                    )
                    
                    # DCAE 디코딩
                    dummy_skip_features = [
                        torch.zeros_like(generated_latent) for _ in range(5)
                    ]
                    audio = self.dcae_model.decode(generated_latent, dummy_skip_features)
                    
                    end_time.record()
                    torch.cuda.synchronize()
                    
                    # 생성 메트릭 기록
                    generation_time = start_time.elapsed_time(end_time)
                    generation_metrics.append({
                        'task': task,
                        'generation_time_ms': generation_time,
                        'sequence_length': shape[-1],
                        'rtf': generation_time / (shape[-1] * 512 / 44100 * 1000)  # Real-time factor
                    })
                    
                    generated_samples.append({
                        'task': task,
                        'audio': audio.cpu(),
                        'latent': generated_latent.cpu(),
                        'generation_time': generation_time
                    })
                    
                except Exception as e:
                    print(f"❌ S6 sample generation error for {task}: {e}")
                    continue
        
        # 저장 및 로깅
        if self.args.save_samples and generated_samples:
            sample_dir = self.checkpoint_dir / f's6_samples_epoch_{epoch}'
            sample_dir.mkdir(exist_ok=True)
            
            for i, sample in enumerate(generated_samples):
                try:
                    # 오디오 저장
                    audio_path = sample_dir / f's6_{sample["task"]}_{i}.wav'
                    torchaudio.save(
                        audio_path,
                        sample['audio'][0],
                        sample_rate=44100
                    )
                    
                    # S6 generation 메타데이터 저장
                    meta_path = sample_dir / f's6_{sample["task"]}_{i}_meta.json'
                    with open(meta_path, 'w') as f:
                        json.dump({
                            'task': sample['task'],
                            'generation_time_ms': sample['generation_time'],
                            'model': 'S6-UNet',
                            'chunk_size': getattr(self.args, 'chunk_size', 256),
                            'epoch': epoch
                        }, f, indent=2)
                    
                    # Enhanced Wandb 로깅
                    if self.args.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            f'samples/s6_{sample["task"]}_{i}': wandb.Audio(
                                sample['audio'][0].numpy(),
                                sample_rate=44100,
                                caption=f'S6 {sample["task"]} sample {i} (Epoch {epoch})'
                            ),
                            f'generation/s6_{sample["task"]}_time_ms': sample['generation_time']
                        })
                        
                except Exception as e:
                    print(f"❌ Sample saving error: {e}")
                    continue
        
        # S6 generation 성능 요약 로깅
        if generation_metrics and self.args.use_wandb and WANDB_AVAILABLE:
            avg_generation_time = np.mean([m['generation_time_ms'] for m in generation_metrics])
            avg_rtf = np.mean([m['rtf'] for m in generation_metrics])
            
            wandb.log({
                'generation/s6_avg_time_ms': avg_generation_time,
                'generation/s6_avg_rtf': avg_rtf,
                'generation/s6_samples_generated': len(generated_samples),
                'epoch': epoch
            })
            
        print(f"✅ S6 generated {len(generated_samples)} samples")
        if generation_metrics:
            avg_time = np.mean([m['generation_time_ms'] for m in generation_metrics])
            print(f"⚡ Average S6 generation time: {avg_time:.1f}ms")
            
    def save_checkpoint(self, epoch, metrics, s6_performance, is_best=False):
        """S6 특화 체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'stage': self.stage,
            'model_state_dict': self.s6_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'args': self.args,
            'metrics': metrics,
            's6_performance': s6_performance,
            'model_type': 'S6-UNet',
            'architecture_info': {
                'state_space_type': 'S6 (Mamba-2)',
                'chunk_size': getattr(self.args, 'chunk_size', 256),
                'use_mem_eff_path': True,
                'mixed_precision': self.use_mixed_precision
            }
        }
        
        # Adaptive weights 저장
        if hasattr(self, 'task_weights'):
            checkpoint['task_weights'] = self.task_weights
        
        # Mixed precision scaler 저장
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        # 일반 체크포인트
        checkpoint_path = self.checkpoint_dir / f's6_checkpoint_stage_{self.stage}_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # 베스트 모델
        if is_best:
            best_path = self.checkpoint_dir / f's6_best_model_stage_{self.stage}.pt'
            torch.save(checkpoint, best_path)
            print(f"🏆 New best S6 model saved!")
            
        # Stage 완료 체크포인트
        if epoch == self.args.epochs - 1:
            final_path = self.checkpoint_dir / f's6_final_stage_{self.stage}.pt'
            torch.save(checkpoint, final_path)
            print(f"🎉 S6 Stage {self.stage} training completed! Saved to {final_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """S6 체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.s6_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'task_weights' in checkpoint:
            self.task_weights = checkpoint['task_weights']
        
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        return checkpoint['epoch']
    
    def _load_prev_stage(self, checkpoint_path):
        """이전 스테이지 S6 체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 모델 가중치만 로드
        self.s6_model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Loaded S6 previous stage checkpoint from {checkpoint_path}")
    
    def train(self):
        """S6 전체 학습 루프"""
        best_val_loss = float('inf')
        start_epoch = 0
        
        # 체크포인트 로드
        if self.args.resume:
            start_epoch = self.load_checkpoint(self.args.resume)
            print(f"🔄 Resumed S6 training from epoch {start_epoch}")
            
        print(f"\n🚀 Starting S6 Stage {self.stage} training")
        print(f"{'='*60}")
        print(f"🧠 Model: S6 (Mamba-2) + U-Net")
        print(f"⚡ Architecture: State Space Dual (SSD)")
        print(f"💾 Memory Efficient: {getattr(self.args, 'chunk_size', 256)} chunk size")
        print(f"🎯 Mixed Precision: {self.use_mixed_precision}")
        print(f"{'='*60}")
        
        # 학습 시작
        for epoch in range(start_epoch, self.args.epochs):
            print(f"\n{'='*60}")
            print(f"S6 Stage {self.stage} - Epoch {epoch+1}/{self.args.epochs}")
            print(f"{'='*60}")
            
            # 학습
            train_loss, task_losses, task_counts, s6_performance = self.train_epoch(epoch)
            
            # 학습 결과 출력 (S6 메트릭 포함)
            print(f"🎵 Train - Loss: {train_loss:.4f}")
            print(f"⚡ S6 Performance:")
            print(f"  Avg Speed: {s6_performance['avg_processing_speed']:.1f}ms")
            print(f"  Avg Memory: {s6_performance['avg_memory_usage']:.1f}GB")
            print(f"  Gradient Stability: {s6_performance['gradient_stability']:.4f}")
            print(f"  Success Rate: {s6_performance['successful_batches']}/{s6_performance['total_batches']}")
            
            # Task별 손실 출력
            for task in ['SONG', 'INST', 'COVER']:
                if task_counts[task] > 0:
                    avg_loss = task_losses[task] / task_counts[task]
                    print(f"  {task}: {avg_loss:.4f} (count: {task_counts[task]})")
            
            # 검증
            val_loss, task_val_losses, s6_val_performance = self.validate(epoch)
            print(f"✅ Val - Loss: {val_loss:.4f}")
            print(f"⚡ S6 Val Performance:")
            print(f"  Inference Speed: {s6_val_performance['avg_inference_speed']:.1f}ms")
            print(f"  Memory Usage: {s6_val_performance['avg_memory_usage']:.1f}GB")
            
            for task, loss in task_val_losses.items():
                if loss > 0:
                    print(f"  {task}: {loss:.4f}")
            
            # 샘플 생성
            if epoch % self.args.sample_interval == 0:
                print(f"🎼 Generating S6 samples...")
                self.generate_samples(epoch)
            
            # 베스트 모델 체크
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.best_metrics['val_loss'] = val_loss
                
            # 체크포인트 저장
            if epoch % self.args.save_interval == 0 or epoch == self.args.epochs - 1:
                all_metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'task_losses': task_val_losses
                }
                self.save_checkpoint(epoch, all_metrics, s6_performance, is_best)
                
        print(f"\n🎉 S6 Stage {self.stage} training completed!")
        print(f"🏆 Best Val Loss: {best_val_loss:.4f}")
        
        # 다음 스테이지 안내
        next_stage_map = {'B': 'C', 'C': 'D', 'D': 'E'}
        if self.stage in next_stage_map:
            next_stage = next_stage_map[self.stage]
            print(f"\n➡️ To continue with S6 Stage {next_stage}, run:")
            print(f"python ssm/train_lyro.py --stage {next_stage} \\")
            print(f"  --prev_stage_checkpoint {self.checkpoint_dir}/s6_final_stage_{self.stage}.pt")


def main():
    parser = argparse.ArgumentParser(description='LYRO S6 (Mamba-2) Training')
    
    # Stage 설정
    parser.add_argument('--stage', type=str, required=True, choices=['B', 'C', 'D', 'E'],
                        help='Training stage')
    
    # 데이터 관련
    parser.add_argument('--train_metadata', type=str, default='dataset/metadata/train_metadata.jsonl')
    parser.add_argument('--val_metadata', type=str, default='dataset/metadata/val_metadata.jsonl')
    parser.add_argument('--gold_metadata', type=str, default='dataset/metadata/gold_metadata.jsonl')
    parser.add_argument('--dataset_root', type=str, default='dataset/')
    
    # S6 모델 관련
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--max_seq_len', type=str, default=8192)
    parser.add_argument('--chunk_size', type=int, default=256,
                        help='S6 chunk size for memory efficiency')
    parser.add_argument('--dcae_checkpoint', type=str, required=True)
    parser.add_argument('--prev_stage_checkpoint', type=str, default=None)
    parser.add_argument('--text_tokenizer_path', type=str, default=None)
    
    # 학습 관련
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    
    # 데이터 처리
    parser.add_argument('--max_audio_length', type=int, default=441000)  # 10초
    parser.add_argument('--max_text_length', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # S6 최적화 관련
    parser.add_argument('--use_mixed_precision', action='store_true', default=True)
    parser.add_argument('--use_torch_compile', action='store_true', default=True)
    parser.add_argument('--compile_mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'])
    
    # 기타
    parser.add_argument('--checkpoint_dir', type=str, default='ssm/checkpoints_s6')
    parser.add_argument('--exp_name', type=str, default='lyro_s6')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--sample_interval', type=int, default=10)
    parser.add_argument('--save_samples', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # CUDA 확인
    if not torch.cuda.is_available():
        print("❌ CUDA required for S6 training!")
        return
    
    print(f"🚀 LYRO S6 (Mamba-2) Training")
    print(f"⚡ Available GPUs: {torch.cuda.device_count()}")
    print(f"🧠 Model: S6 State Space Dual (SSD)")
    print(f"💾 Chunk Size: {args.chunk_size}")
    print(f"🎯 Mixed Precision: {args.use_mixed_precision}")
    print(f"⚡ Torch Compile: {args.use_torch_compile}")
    
    try:
        trainer = LyroS6Trainer(args)
        trainer.train()
        print("🎉 S6 training completed successfully!")
        
    except Exception as e:
        print(f"❌ S6 training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()