#!/usr/bin/env python3
"""
Multimodal LYRO Training Pipeline
오디오-가사 정렬 + TTS 보조 학습 + 기존 생성 모델 통합 훈련
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import wandb
from tqdm import tqdm
import os

from ..models.multimodal_lyro import MultimodalLyroSystem
from ..models.generator import GeneratorConfig
from ..models.losses import TrainingScheduler
from .config import LyroConfig


class MultimodalTrainer:
    """
    멀티모달 LYRO 훈련기
    
    훈련 단계:
    1. Pre-training: 각 컴포넌트 개별 훈련
    2. Alignment: 오디오-가사 정렬 훈련  
    3. Joint: 전체 시스템 통합 훈련
    4. Fine-tuning: TTS 보조 학습 강화
    """
    
    def __init__(
        self,
        config: LyroConfig,
        model: MultimodalLyroSystem = None,
        device: str = "auto"
    ):
        self.config = config
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 모델 초기화
        if model is None:
            generator_config = GeneratorConfig(
                d_model=config.generator.d_model,
                n_layers=config.generator.n_layers,
                n_heads=config.generator.n_heads,
                d_ff=config.generator.d_ff,
                latent_channels=config.generator.latent_channels,
                latent_time_steps=config.generator.latent_time_steps,
                max_lyrics_length=config.generator.max_lyrics_length,
                max_style_length=getattr(config.generator, 'max_style_length', 128)
            )
            
            self.model = MultimodalLyroSystem(
                generator_config=generator_config,
                training_mode="joint"
            )
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # 옵티마이저 설정 (컴포넌트별)
        self._setup_optimizers()
        
        # 훈련 상태
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Training scheduler placeholder (실제 데이터로더에서 초기화)
        self.training_scheduler = None
        self._scheduler_initialized = False
        
        # 로깅
        self.use_wandb = config.training.use_wandb
        if self.use_wandb:
            wandb.init(
                project=config.training.wandb_project,
                config=config.__dict__
            )
        
        print(f"🎵 MultimodalTrainer initialized")
        print(f"  - Device: {self.device}")
        print(f"  - Model parameters: {self._count_parameters():,}")
        print(f"  - Training stages: pre-training → alignment → joint → fine-tuning")
    
    def _setup_optimizers(self):
        """컴포넌트별 옵티마이저 설정"""
        
        # 1. Generator 옵티마이저 (기존 LYRO)
        self.generator_optimizer = optim.AdamW(
            self.model.generator.parameters(),
            lr=self.config.generator.learning_rate,
            weight_decay=self.config.generator.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # 2. Text Encoder 옵티마이저
        self.text_optimizer = optim.AdamW(
            self.model.text_encoder.parameters(),
            lr=self.config.generator.learning_rate * 0.5,  # 더 작은 학습률
            weight_decay=self.config.generator.weight_decay
        )
        
        # 3. Aligner 옵티마이저
        self.aligner_optimizer = optim.AdamW(
            self.model.aligner.parameters(),
            lr=1e-4,  # 정렬은 더 세밀하게
            weight_decay=0.01
        )
        
        # 4. Fusion Layers 옵티마이저  
        fusion_params = list(self.model.tts_condition_proj.parameters()) + \
                      list(self.model.tts_weight_adapter.parameters()) + \
                      list(self.model.pronunciation_predictor.parameters()) + \
                      list(self.model.condition_fusion.parameters()) + \
                      list(self.model.task_heads.parameters()) + \
                      [self.model.task_weights]
        
        self.fusion_optimizer = optim.AdamW(
            fusion_params,
            lr=self.config.generator.learning_rate,
            weight_decay=self.config.generator.weight_decay
        )
        
        # 스케줄러들
        self.schedulers = {
            'generator': optim.lr_scheduler.CosineAnnealingLR(
                self.generator_optimizer,
                T_max=self.config.generator.epochs,
                eta_min=1e-6
            ),
            'text': optim.lr_scheduler.CosineAnnealingLR(
                self.text_optimizer,
                T_max=self.config.generator.epochs,
                eta_min=1e-6
            ),
            'aligner': optim.lr_scheduler.CosineAnnealingLR(
                self.aligner_optimizer,
                T_max=self.config.generator.epochs,
                eta_min=1e-6
            ),
            'fusion': optim.lr_scheduler.CosineAnnealingLR(
                self.fusion_optimizer,
                T_max=self.config.generator.epochs,
                eta_min=1e-6
            )
        }
    
    def _count_parameters(self) -> int:
        """전체 파라미터 수 계산"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _initialize_scheduler_with_dataloader(self, train_loader: DataLoader, num_epochs: int):
        """실제 데이터로더 기반으로 스케줄러 초기화"""
        if not self._scheduler_initialized:
            actual_total_steps = len(train_loader) * num_epochs
            
            self.training_scheduler = TrainingScheduler(
                total_steps=actual_total_steps,
                weight_decay_start=int(actual_total_steps * 0.9)
            )
            
            self._scheduler_initialized = True
            
            print(f"📊 Scheduler initialized:")
            print(f"  - Total steps: {actual_total_steps}")
            print(f"  - Weight decay starts at step: {self.training_scheduler.weight_decay_start}")
            print(f"  - Decay duration: {actual_total_steps - self.training_scheduler.weight_decay_start} steps")
    
    def train_stage(
        self,
        stage: str,  # 'pre-training', 'alignment', 'joint', 'fine-tuning'
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        num_epochs: int = None,
        save_interval: int = 5
    ):
        """훈련 단계별 실행"""
        
        if num_epochs is None:
            num_epochs = self.config.generator.epochs
        
        # 실제 데이터로더 기반으로 스케줄러 초기화
        self._initialize_scheduler_with_dataloader(train_loader, num_epochs)
        
        print(f"\n🚀 Starting '{stage}' training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.current_epoch += 1
            
            # 훈련
            train_metrics = self._train_epoch(stage, train_loader)
            
            # 검증
            if val_loader is not None:
                val_metrics = self._validate_epoch(stage, val_loader)
                
                # 모델 저장 (최고 성능)
                if val_metrics['total_loss'] < self.best_loss:
                    self.best_loss = val_metrics['total_loss']
                    self._save_checkpoint(f'best_{stage}')
            
            # 주기적 저장
            if epoch % save_interval == 0:
                self._save_checkpoint(f'{stage}_epoch_{epoch}')
            
            # 로깅
            self._log_metrics(stage, epoch, train_metrics, val_metrics)
            
            # 스케줄러 업데이트
            self._update_schedulers(stage)
        
        print(f"✅ '{stage}' training completed!")
    
    def _train_epoch(self, stage: str, train_loader: DataLoader) -> Dict[str, float]:
        """에포크 단위 훈련"""
        
        self.model.train()
        
        # 훈련할 컴포넌트 결정
        if stage == 'pre-training':
            # Generator만 훈련
            self._freeze_components(['text_encoder', 'aligner', 'fusion'])
        elif stage == 'alignment':
            # Aligner + Text Encoder 훈련
            self._freeze_components(['generator'])
            self._unfreeze_components(['text_encoder', 'aligner', 'fusion'])
        elif stage == 'joint':
            # 전체 훈련
            self._unfreeze_components(['generator', 'text_encoder', 'aligner', 'fusion'])
        elif stage == 'fine-tuning':
            # TTS 관련만 세밀 조정
            self._freeze_components(['generator', 'text_encoder'])
            self._unfreeze_components(['aligner', 'fusion'])
        
        total_loss = 0.0
        num_batches = 0
        loss_components = {}
        
        progress_bar = tqdm(train_loader, desc=f"Training {stage}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 배치 준비
            batch = self._prepare_batch(batch)
            
            # Forward pass
            results = self.model(
                lyrics=batch.get('lyrics'),
                style=batch.get('style'),
                reference_latents=batch.get('reference_latents'),
                target_latents=batch.get('target_latents'),
                audio=batch.get('audio'),
                lyrics_timestamps=batch.get('lyrics_timestamps'),
                training_stage=self._get_training_stage_name(stage),
                return_alignment=batch.get('audio') is not None,
                return_tts=True
            )
            
            # 손실 계산 (current_step 전달)
            losses = self.model.compute_losses(
                predictions=results,
                targets=batch,
                training_stage=self._get_training_stage_name(stage),
                current_step=self.global_step
            )
            
            total_loss_batch = losses['total']
            
            # Backward pass
            self._backward_pass(stage, total_loss_batch)
            
            # 통계 업데이트
            total_loss += total_loss_batch.item()
            num_batches += 1
            
            for key, value in losses.items():
                if key != 'total':
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value.item()
            
            # 진행률 업데이트 (mHuBERT weight 포함)
            current_mhubert_weight = self.training_scheduler.get_mhubert_weight(self.global_step)
            progress_bar.set_postfix({
                'loss': f"{total_loss_batch.item():.4f}",
                'avg_loss': f"{total_loss / num_batches:.4f}",
                'mhub_w': f"{current_mhubert_weight:.3f}"
            })
            
            self.global_step += 1
        
        # 평균 계산
        avg_metrics = {'total_loss': total_loss / num_batches}
        for key, value in loss_components.items():
            avg_metrics[key] = value / num_batches
        
        return avg_metrics
    
    def _validate_epoch(self, stage: str, val_loader: DataLoader) -> Dict[str, float]:
        """검증"""
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        loss_components = {}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating {stage}"):
                batch = self._prepare_batch(batch)
                
                results = self.model(
                    lyrics=batch.get('lyrics'),
                    style=batch.get('style'),
                    reference_latents=batch.get('reference_latents'),
                    target_latents=batch.get('target_latents'),
                    audio=batch.get('audio'),
                    lyrics_timestamps=batch.get('lyrics_timestamps'),
                    training_stage=self._get_training_stage_name(stage),
                    return_alignment=batch.get('audio') is not None,
                    return_tts=True
                )
                
                losses = self.model.compute_losses(
                    predictions=results,
                    targets=batch,
                    training_stage=self._get_training_stage_name(stage),
                    current_step=self.global_step
                )
                
                total_loss += losses['total'].item()
                num_batches += 1
                
                for key, value in losses.items():
                    if key != 'total':
                        if key not in loss_components:
                            loss_components[key] = 0.0
                        loss_components[key] += value.item()
        
        # 평균 계산
        avg_metrics = {'total_loss': total_loss / num_batches}
        for key, value in loss_components.items():
            avg_metrics[key] = value / num_batches
        
        return avg_metrics
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """배치 데이터 준비"""
        prepared = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device)
            else:
                prepared[key] = value  # 텍스트 등은 그대로
        
        return prepared
    
    def _get_training_stage_name(self, stage: str) -> str:
        """훈련 단계 이름 변환"""
        mapping = {
            'pre-training': 'generation_only',
            'alignment': 'alignment_only', 
            'joint': 'joint',
            'fine-tuning': 'joint'
        }
        return mapping.get(stage, 'joint')
    
    def _freeze_components(self, components: List[str]):
        """컴포넌트 동결"""
        for comp in components:
            if comp == 'generator':
                for param in self.model.generator.parameters():
                    param.requires_grad = False
            elif comp == 'text_encoder':
                for param in self.model.text_encoder.parameters():
                    param.requires_grad = False
            elif comp == 'aligner':
                for param in self.model.aligner.parameters():
                    param.requires_grad = False
            elif comp == 'fusion':
                for param in self.model.tts_condition_proj.parameters():
                    param.requires_grad = False
                for param in self.model.condition_fusion.parameters():
                    param.requires_grad = False
    
    def _unfreeze_components(self, components: List[str]):
        """컴포넌트 해제"""
        for comp in components:
            if comp == 'generator':
                for param in self.model.generator.parameters():
                    param.requires_grad = True
            elif comp == 'text_encoder':
                for param in self.model.text_encoder.parameters():
                    param.requires_grad = True
            elif comp == 'aligner':
                for param in self.model.aligner.parameters():
                    param.requires_grad = True
            elif comp == 'fusion':
                for param in self.model.tts_condition_proj.parameters():
                    param.requires_grad = True
                for param in self.model.condition_fusion.parameters():
                    param.requires_grad = True
    
    def _backward_pass(self, stage: str, loss: torch.Tensor):
        """단계별 역전파"""
        
        # 그라디언트 초기화
        if stage in ['pre-training', 'joint']:
            self.generator_optimizer.zero_grad()
        if stage in ['alignment', 'joint', 'fine-tuning']:
            self.text_optimizer.zero_grad()
            self.aligner_optimizer.zero_grad()
        if stage in ['alignment', 'joint', 'fine-tuning']:
            self.fusion_optimizer.zero_grad()
        
        # 역전파
        loss.backward()
        
        # 그라디언트 클리핑
        if stage in ['pre-training', 'joint']:
            torch.nn.utils.clip_grad_norm_(
                self.model.generator.parameters(),
                self.config.generator.grad_clip
            )
        
        # 옵티마이저 스텝
        if stage in ['pre-training', 'joint']:
            self.generator_optimizer.step()
        if stage in ['alignment', 'joint', 'fine-tuning']:
            self.text_optimizer.step()
            self.aligner_optimizer.step()
            self.fusion_optimizer.step()
    
    def _update_schedulers(self, stage: str):
        """스케줄러 업데이트"""
        if stage in ['pre-training', 'joint']:
            self.schedulers['generator'].step()
        if stage in ['alignment', 'joint', 'fine-tuning']:
            self.schedulers['text'].step()
            self.schedulers['aligner'].step()
            self.schedulers['fusion'].step()
    
    def _log_metrics(self, stage: str, epoch: int, train_metrics: Dict, val_metrics: Dict = None):
        """메트릭 로깅"""
        
        print(f"\nEpoch {epoch} ({stage}):")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        
        if val_metrics:
            print(f"  Val Loss: {val_metrics['total_loss']:.4f}")
        
        # 세부 손실들
        for key, value in train_metrics.items():
            if key != 'total_loss':
                print(f"  Train {key}: {value:.4f}")
        
        # Wandb 로깅
        if self.use_wandb:
            log_dict = {
                f'train/{key}': value for key, value in train_metrics.items()
            }
            
            if val_metrics:
                log_dict.update({
                    f'val/{key}': value for key, value in val_metrics.items()
                })
            
            log_dict.update({
                'epoch': epoch,
                'stage': stage,
                'global_step': self.global_step
            })
            
            wandb.log(log_dict)
    
    def _save_checkpoint(self, name: str):
        """체크포인트 저장"""
        
        checkpoint_dir = self.config.training.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'text_optimizer': self.text_optimizer.state_dict(),
            'aligner_optimizer': self.aligner_optimizer.state_dict(),
            'fusion_optimizer': self.fusion_optimizer.state_dict(),
            'schedulers': {k: v.state_dict() for k, v in self.schedulers.items()},
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def full_training_pipeline(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None
    ):
        """전체 훈련 파이프라인 실행"""
        
        print("🎵 Starting Full Multimodal LYRO Training Pipeline")
        print("=" * 60)
        
        # 1. Pre-training (Generator 기본 훈련)
        print("\n📚 Stage 1: Pre-training (Generator)")
        self.train_stage('pre-training', train_loader, val_loader, num_epochs=20)
        
        # 2. Alignment Training (정렬 시스템 훈련)
        print("\n🎯 Stage 2: Alignment Training")
        self.train_stage('alignment', train_loader, val_loader, num_epochs=15)
        
        # 3. Joint Training (전체 통합 훈련)
        print("\n🤝 Stage 3: Joint Training")
        self.train_stage('joint', train_loader, val_loader, num_epochs=30)
        
        # 4. Fine-tuning (TTS 보조 학습 강화)
        print("\n✨ Stage 4: Fine-tuning")
        self.train_stage('fine-tuning', train_loader, val_loader, num_epochs=10)
        
        print("\n🎉 Full Training Pipeline Completed!")
        
        # 최종 모델 저장
        self.model.save_pretrained(os.path.join(self.config.training.checkpoint_dir, "final_model"))


if __name__ == "__main__":
    # 테스트
    from ..training.config import LyroConfig
    
    config = LyroConfig()
    trainer = MultimodalTrainer(config)
    
    print("✅ MultimodalTrainer initialized successfully!")