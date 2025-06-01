# lyro/ssm/train_lyro.py
"""
LYRO SSM Training Script
Flow Matching 기반 음악 생성 모델 학습
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional, List
import math
import torchaudio  # ✅ 추가: 누락된 import

# Wandb import handling
WANDB_AVAILABLE = False
wandb = None

try:
    # Use importlib to avoid static analysis issues
    import importlib
    _wandb = importlib.import_module('wandb')
    WANDB_AVAILABLE = True
    wandb = _wandb
except ImportError:
    # Create a mock wandb class to avoid undefined variable errors
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

from ssm.model import LyroSSMUNet, TaskController, EOSTokenHandler
from ssm.flow_matching import LyroFlowMatching, FlowConfig  # ✅ 수정: FlowMatching -> LyroFlowMatching
from dcae.model import LyroMusicDCAE  # ✅ 수정: MusicDCAE -> LyroMusicDCAE
from dataset.dataset import LyroDataset, LyroCollator
from dataset.tokenizer import LyroTokenizer


class LyroTrainer:
    """
    LYRO SSM 학습 관리 클래스
    
    Flow Matching을 사용하여 SSM + U-Net 모델을 학습하고,
    다단계 학습 전략을 구현합니다.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stage = args.stage  # 학습 단계 (B, C, D, E)
        
        # 모델 초기화
        self._initialize_models()
        
        # Flow Matching 설정
        self.flow_config = FlowConfig()
        self.flow_matching = LyroFlowMatching(  # ✅ 수정: FlowMatching -> LyroFlowMatching
            model=self.ssm_model,
            scheduler_type="cosine",
            solver_type="heun",
            sigma=1e-4,
            flow_type="rectified"
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
                project="lyro-ssm",
                name=f"{args.exp_name}_stage_{args.stage}",
                config=vars(args)
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
        
    def _initialize_models(self):
        """모델 초기화"""
        # SSM + U-Net 모델
        self.ssm_model = LyroSSMUNet(
            input_channels=8,  # DCAE latent channels
            hidden_dims=[128, 128, 256, 256, 512],
            mamba_layers=[2, 2, 3, 3, 4],
            max_seq_len=8192  # ✅ 수정: self.args.max_seq_len -> 8192
        ).to(self.device)
        
        # DCAE 모델 (인코딩용)
        self.dcae_model = LyroMusicDCAE(  # ✅ 수정: MusicDCAE -> LyroMusicDCAE
            sample_rate=44100,
            latent_channels=8  # ✅ 수정: compression_factor 제거, latent_channels 사용
        ).to(self.device)
        
        # DCAE 체크포인트 로드
        if self.args.dcae_checkpoint:
            dcae_ckpt = torch.load(self.args.dcae_checkpoint, map_location=self.device)
            self.dcae_model.load_state_dict(dcae_ckpt['model_state_dict'])
            self.dcae_model.eval()
            print(f"Loaded DCAE from {self.args.dcae_checkpoint}")
            
        # 이전 스테이지 체크포인트 로드
        if self.args.prev_stage_checkpoint:
            self._load_prev_stage(self.args.prev_stage_checkpoint)
            
    def _setup_optimization(self):
        """옵티마이저 및 스케줄러 설정"""
        # Stage별 학습률 설정
        stage_lrs = {
            'B': 1e-3,
            'C': 5e-4,
            'D': 3e-4,
            'E': 1e-4
        }
        
        lr = self.args.learning_rate or stage_lrs.get(self.stage, 3e-4)
        
        # 옵티마이저
        self.optimizer = optim.AdamW(
            self.ssm_model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay,
            eps=1e-8
        )
        
        # 스케줄러 (Stage E는 Cosine Annealing)
        if self.stage == 'E':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs,
                eta_min=3e-5
            )
        else:
            # Warmup + Linear Decay
            warmup_steps = self.args.warmup_steps
            total_steps = self.args.epochs * self.args.steps_per_epoch
            
            self.scheduler = self._get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            
    def _get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """Warmup이 있는 선형 스케줄러"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / 
                float(max(1, num_training_steps - num_warmup_steps))
            )
            
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _setup_data(self):
        """데이터셋 및 로더 설정"""
        # Stage별 태스크 비율
        stage_task_ratios = {
            'B': {'SONG': 1.0, 'INST': 0.0, 'COVER': 0.0},  # 기본 생성만
            'C': {'SONG': 0.7, 'INST': 0.3, 'COVER': 0.0},  # Dual task
            'D': {'SONG': 0.5, 'INST': 0.2, 'COVER': 0.3},  # ICL 추가
            'E': {'SONG': 0.6, 'INST': 0.15, 'COVER': 0.25}  # Quality focus
        }
        
        task_ratios = stage_task_ratios.get(self.stage, {'SONG': 0.7, 'INST': 0.15, 'COVER': 0.15})
        
        # 데이터셋
        train_metadata = self.args.train_metadata
        if self.stage == 'E' and self.args.gold_metadata:
            # Stage E는 고품질 데이터만 사용
            train_metadata = self.args.gold_metadata
            
        self.train_dataset = LyroDataset(
            metadata_path=train_metadata,
            dataset_root=self.args.dataset_root,
            task_ratios=task_ratios,
            augmentation=True,
            use_processed=True
        )
        
        self.val_dataset = LyroDataset(
            metadata_path=self.args.val_metadata,
            dataset_root=self.args.dataset_root,
            task_ratios=task_ratios,
            augmentation=False,
            use_processed=True
        )
        
        # Collator
        collator = LyroCollator(
            tokenizer=self.tokenizer,
            max_audio_length=self.args.max_audio_length,
            max_text_length=self.args.max_text_length
        )
        
        # 데이터 로더
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collator,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=collator
        )
        
        # Steps per epoch 계산
        self.args.steps_per_epoch = len(self.train_loader)
        
    def _setup_adaptive_weights(self):
        """Adaptive Weight Manager 설정"""
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
        
        self.weight_update_interval = 100  # 100 스텝마다 가중치 업데이트
        
    def _validate_batch(self, batch):
        """배치 데이터 검증"""
        try:
            audio = batch['audio']
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                print("Warning: Invalid audio data detected")
                return False
                
            if audio.shape[1] != 2:
                print(f"Warning: Expected stereo audio, got {audio.shape[1]} channels")
                return False
                
            return True
        except Exception as e:
            print(f"Batch validation error: {e}")
            return False
    
    def train_epoch(self, epoch):
        """한 에폭 학습"""
        self.ssm_model.train()
        self.dcae_model.eval()  # DCAE는 고정
        
        total_loss = 0
        task_losses = {'SONG': 0, 'INST': 0, 'COVER': 0}
        task_counts = {'SONG': 0, 'INST': 0, 'COVER': 0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 데이터 검증
                if not self._validate_batch(batch):
                    continue
                    
                # 데이터 준비
                audio = batch['audio'].to(self.device)
                audio_lengths = batch['audio_lengths'].to(self.device)
                task_tokens = batch['task_tokens']
                
                # DCAE 인코딩
                with torch.no_grad():
                    latents, _ = self.dcae_model.encode(audio)  # ✅ 수정: skip_features 처리
                    
                # 조건 준비
                conditions = self._prepare_conditions(batch)
                
                # Flow Matching 학습
                loss = self.flow_matching.training_loss(latents, conditions)
                
                # Task별 손실 기록
                for i, task_token in enumerate(task_tokens):
                    task_type = task_token.split('=')[1].rstrip('>')
                    if task_type in task_losses:
                        task_losses[task_type] += loss.item()
                        task_counts[task_type] += 1
                    
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.ssm_model.parameters(), 
                    self.args.grad_clip
                )
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Adaptive weight update (Stage C, D, E)
                if hasattr(self, 'task_weights') and batch_idx % self.weight_update_interval == 0:
                    self._update_adaptive_weights(grad_norm)
                    
                # 손실 기록
                total_loss += loss.item()
                
                # Progress bar 업데이트
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'grad_norm': f'{grad_norm:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                  # Wandb 로깅
                if self.args.use_wandb and WANDB_AVAILABLE and batch_idx % self.args.log_interval == 0:
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/grad_norm': grad_norm,
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'step': epoch * len(self.train_loader) + batch_idx
                    }
                    
                    # Task별 손실 로깅
                    for task in ['SONG', 'INST', 'COVER']:
                        if task_counts[task] > 0:
                            log_dict[f'train/loss_{task}'] = task_losses[task] / task_counts[task]
                            
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
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, task_losses, task_counts
    
    def _prepare_conditions(self, batch):
        """배치 데이터에서 조건 정보 준비"""
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
                conditions['task_token'].append(0)  # 기본값
        conditions['task_token'] = torch.tensor(
            conditions['task_token'], 
            device=self.device
        )
        
        # Style prompt (genre를 벡터로 변환)
        for genres in batch['genres']:
            # 간단한 genre 임베딩 (실제로는 더 복잡한 처리 필요)
            genre_vec = self._encode_genres(genres)
            conditions['style_prompt'].append(genre_vec)
        conditions['style_prompt'] = torch.stack(conditions['style_prompt']).to(self.device)
        
        # ICL reference (COVER 태스크)
        if 'reference_audios' in batch:
            ref_audios = batch['reference_audios'].to(self.device)
            with torch.no_grad():
                conditions['icl_reference'], _ = self.dcae_model.encode(ref_audios)  # ✅ 수정: skip_features 처리
                
        return conditions
    
    def _encode_genres(self, genres: List[str]) -> torch.Tensor:
        """장르 리스트를 벡터로 인코딩"""
        # 간단한 원핫 인코딩 (실제로는 learned embedding 사용)
        genre_map = {
            'Pop': 0, 'Rock': 1, 'Jazz': 2, 'Classical': 3,
            'Blues': 4, 'Electronic': 5, 'Hip-Hop': 6, 'Folk': 7,
            'Country': 8, 'R&B': 9, 'Metal': 10, 'Indie': 11,
            'Unknown': 12
        }
        
        vec = torch.zeros(512)  # Style vector size
        for genre in genres:
            if genre in genre_map:
                idx = genre_map[genre]
                if idx * 40 + 40 <= 512:  # 범위 검증
                    vec[idx * 40:(idx + 1) * 40] = 1.0  # 각 장르당 40차원
                
        return vec
    
    def _update_adaptive_weights(self, current_grad_norm):
        """Gradient norm 기반 적응적 가중치 업데이트"""
        # 현재 gradient norm을 task별로 추정
        # (실제로는 task별로 분리된 loss가 필요)
        
        # 간단한 휴리스틱: gradient norm이 높으면 학습이 어려운 task
        if current_grad_norm > 1.0:
            # 어려운 task의 가중치 증가
            for task in self.task_weights:
                self.task_weights[task] *= 1.01
        else:
            # 쉬운 task의 가중치 감소
            for task in self.task_weights:
                self.task_weights[task] *= 0.99
                
        # 정규화
        total_weight = sum(self.task_weights.values())
        for task in self.task_weights:
            self.task_weights[task] /= total_weight
            
    def validate(self, epoch):
        """검증 수행"""
        self.ssm_model.eval()
        self.dcae_model.eval()
        
        total_loss = 0
        task_losses = {'SONG': 0, 'INST': 0, 'COVER': 0}
        task_counts = {'SONG': 0, 'INST': 0, 'COVER': 0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                try:
                    # 데이터 준비
                    audio = batch['audio'].to(self.device)
                    task_tokens = batch['task_tokens']
                    
                    # DCAE 인코딩
                    latents, _ = self.dcae_model.encode(audio)  # ✅ 수정: skip_features 처리
                    
                    # 조건 준비
                    conditions = self._prepare_conditions(batch)
                    
                    # Flow Matching 손실
                    loss = self.flow_matching.training_loss(latents, conditions)
                    
                    # Task별 손실 기록
                    for i, task_token in enumerate(task_tokens):
                        task_type = task_token.split('=')[1].rstrip('>')
                        if task_type in task_losses:
                            task_losses[task_type] += loss.item()
                            task_counts[task_type] += 1
                        
                    total_loss += loss.item()
                    
                except Exception as e:
                    print(f"Validation error: {e}")
                    continue
                
        # 평균 계산
        avg_loss = total_loss / max(len(self.val_loader), 1)
        
        # Task별 평균
        task_avg_losses = {}
        for task in ['SONG', 'INST', 'COVER']:
            if task_counts[task] > 0:
                task_avg_losses[task] = task_losses[task] / task_counts[task]
            else:
                task_avg_losses[task] = 0.0
                  # Wandb 로깅
        if self.args.use_wandb and WANDB_AVAILABLE:
            log_dict = {
                'val/loss': avg_loss,
                'epoch': epoch
            }
            
            for task, avg in task_avg_losses.items():
                log_dict[f'val/loss_{task}'] = avg
                
            wandb.log(log_dict)
            
        return avg_loss, task_avg_losses
    
    def generate_samples(self, epoch, num_samples=4):
        """검증 중 샘플 생성"""
        self.ssm_model.eval()
        self.dcae_model.eval()
        
        # 각 태스크별로 샘플 생성
        tasks = ['SONG', 'INST']
        if self.stage in ['D', 'E']:
            tasks.append('COVER')
            
        generated_samples = []
        
        with torch.no_grad():
            for task in tasks:
                try:
                    # 조건 준비
                    conditions = {
                        'task_token': torch.tensor([TaskController.TASK_TOKENS[task]]).to(self.device),
                        'lyrics': None,
                        'style_prompt': torch.randn(1, 512).to(self.device),  # 랜덤 스타일
                        'icl_reference': None
                    }
                    
                    # 가사 추가 (SONG 태스크)
                    if task == 'SONG':
                        sample_lyrics = "Walking down the street tonight\nEverything feels so right"
                        lyrics_tokens = self.tokenizer.encode_text(sample_lyrics)
                        conditions['lyrics'] = torch.tensor([lyrics_tokens]).to(self.device)
                        
                    # Flow Matching 생성
                    shape = (1, 8, 512)  # (batch, channels, time) - 약 24초
                    generated_latent, _ = self.flow_matching.generate(
                        shape=shape,
                        conditions=conditions,
                        num_steps=10,  # ✅ 수정: steps -> num_steps
                        cfg_scale=1.5
                    )
                    
                    # DCAE 디코딩 (더미 skip_features 생성)
                    dummy_skip_features = [torch.zeros_like(generated_latent) for _ in range(5)]
                    audio = self.dcae_model.decode(generated_latent, dummy_skip_features)
                    
                    generated_samples.append({
                        'task': task,
                        'audio': audio.cpu(),
                        'latent': generated_latent.cpu()
                    })
                    
                except Exception as e:
                    print(f"Sample generation error for {task}: {e}")
                    continue
                
        # 저장 및 로깅
        if self.args.save_samples and generated_samples:
            sample_dir = self.checkpoint_dir / f'samples_epoch_{epoch}'
            sample_dir.mkdir(exist_ok=True)
            
            for i, sample in enumerate(generated_samples):
                try:
                    # 오디오 저장
                    audio_path = sample_dir / f'{sample["task"]}_{i}.wav'
                    torchaudio.save(
                        audio_path,
                        sample['audio'][0],
                        sample_rate=44100
                    )
                      # Wandb 로깅
                    if self.args.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            f'samples/{sample["task"]}_{i}': wandb.Audio(
                                sample['audio'][0].numpy(),
                                sample_rate=44100,
                                caption=f'{sample["task"]} sample {i}'
                            )
                        })
                        
                except Exception as e:
                    print(f"Sample saving error: {e}")
                    continue
                    
    def save_checkpoint(self, epoch, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'stage': self.stage,
            'model_state_dict': self.ssm_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'args': self.args
        }
        
        # Adaptive weights 저장 (있는 경우)
        if hasattr(self, 'task_weights'):
            checkpoint['task_weights'] = self.task_weights
            
        # 일반 체크포인트
        checkpoint_path = self.checkpoint_dir / f'checkpoint_stage_{self.stage}_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # 베스트 모델
        if is_best:
            best_path = self.checkpoint_dir / f'best_model_stage_{self.stage}.pt'
            torch.save(checkpoint, best_path)
            
        # Stage 완료 체크포인트
        if epoch == self.args.epochs - 1:
            final_path = self.checkpoint_dir / f'final_stage_{self.stage}.pt'
            torch.save(checkpoint, final_path)
            print(f"Stage {self.stage} training completed! Saved to {final_path}")
            
    def load_checkpoint(self, checkpoint_path):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.ssm_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'task_weights' in checkpoint:
            self.task_weights = checkpoint['task_weights']
            
        return checkpoint['epoch']
    
    def _load_prev_stage(self, checkpoint_path):
        """이전 스테이지 체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 모델 가중치만 로드 (optimizer는 초기화)
        self.ssm_model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded previous stage checkpoint from {checkpoint_path}")
        
    def train(self):
        """전체 학습 루프"""
        best_val_loss = float('inf')
        start_epoch = 0
        
        # 체크포인트 로드 (있는 경우)
        if self.args.resume:
            start_epoch = self.load_checkpoint(self.args.resume)
            print(f"Resumed from epoch {start_epoch}")
            
        print(f"\nStarting Stage {self.stage} training")
        print(f"{'='*50}")
        
        # 학습 시작
        for epoch in range(start_epoch, self.args.epochs):
            print(f"\n{'='*50}")
            print(f"Stage {self.stage} - Epoch {epoch}/{self.args.epochs}")
            print(f"{'='*50}")
            
            # 학습
            train_loss, task_losses, task_counts = self.train_epoch(epoch)
            
            # Task별 손실 출력
            print(f"Train - Total Loss: {train_loss:.4f}")
            for task in ['SONG', 'INST', 'COVER']:
                if task_counts[task] > 0:
                    avg_loss = task_losses[task] / task_counts[task]
                    print(f"  {task}: {avg_loss:.4f} (count: {task_counts[task]})")
                    
            # 검증
            val_loss, task_val_losses = self.validate(epoch)
            print(f"Val - Total Loss: {val_loss:.4f}")
            for task, loss in task_val_losses.items():
                if loss > 0:
                    print(f"  {task}: {loss:.4f}")
                    
            # 샘플 생성 (주기적으로)
            if epoch % self.args.sample_interval == 0:
                self.generate_samples(epoch)
                
            # 체크포인트 저장
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                
            if epoch % self.args.save_interval == 0 or epoch == self.args.epochs - 1:
                self.save_checkpoint(epoch, is_best)
                
        print(f"\nStage {self.stage} training completed!")
        
        # 다음 스테이지 안내
        next_stage_map = {'B': 'C', 'C': 'D', 'D': 'E'}
        if self.stage in next_stage_map:
            next_stage = next_stage_map[self.stage]
            print(f"\nTo continue with Stage {next_stage}, run:")
            print(f"python ssm/train_lyro.py --stage {next_stage} \\")
            print(f"  --prev_stage_checkpoint {self.checkpoint_dir}/final_stage_{self.stage}.pt")


def main():
    parser = argparse.ArgumentParser(description='LYRO SSM Training')
    
    # Stage 설정
    parser.add_argument('--stage', type=str, required=True, choices=['B', 'C', 'D', 'E'],
                        help='Training stage')
    
    # 데이터 관련
    parser.add_argument('--train_metadata', type=str, default='dataset/metadata/train_metadata.jsonl',
                        help='Training metadata path')
    parser.add_argument('--val_metadata', type=str, default='dataset/metadata/val_metadata.jsonl',
                        help='Validation metadata path')
    parser.add_argument('--gold_metadata', type=str, default='dataset/metadata/gold_metadata.jsonl',
                        help='Gold metadata for Stage E')
    parser.add_argument('--dataset_root', type=str, default='dataset/',
                        help='Dataset root directory')
    
    # 모델 관련
    parser.add_argument('--max_seq_len', type=int, default=8192,  # ✅ 수정: 6144 -> 8192
                        help='Maximum sequence length')
    parser.add_argument('--dcae_checkpoint', type=str, required=True,
                        help='DCAE checkpoint path')
    parser.add_argument('--prev_stage_checkpoint', type=str, default=None,
                        help='Previous stage checkpoint')
    parser.add_argument('--text_tokenizer_path', type=str, default=None,
                        help='Text tokenizer model path')
    
    # 학습 관련
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (stage-specific default)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Warmup steps')
    
    # 데이터 처리
    parser.add_argument('--max_audio_length', type=int, default=441000,  # ✅ 수정: 10초로 증가
                        help='Max audio length in samples (10s at 44.1kHz)')
    parser.add_argument('--max_text_length', type=int, default=512,
                        help='Max text token length')
    
    # 기타
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='ssm/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--exp_name', type=str, default='lyro_ssm',
                        help='Experiment name')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Checkpoint save interval')
    parser.add_argument('--sample_interval', type=int, default=10,
                        help='Sample generation interval')
    parser.add_argument('--save_samples', action='store_true',
                        help='Save generated samples')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 학습 시작
    trainer = LyroTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()