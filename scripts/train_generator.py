#!/usr/bin/env python3
# lyro/scripts/train_generator.py
"""
LYRO Generator 훈련 스크립트 - 1.5B 파라미터 모델
SSM + Flow Matching + REPA Loss
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import logging
import time

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyro.models.generator import create_lyro_generator, GeneratorConfig
from lyro.models.dcae import create_dcae
from lyro.models.losses import CombinedLoss, FlowMatchingLoss, REPALoss, ReconstructionLoss, PerceptualLoss
from lyro.data.dataset import create_lyro_datasets
from lyro.data.collator import LyroCollator
from lyro.data.tokenizer import LyroTokenizer
from lyro.training.config import LyroConfig
from lyro.utils import get_audio_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LyroGeneratorTrainer:
    """LYRO Generator 트레이너 (1.5B 파라미터)"""
    
    def __init__(self, config: LyroConfig, dcae_checkpoint: str = None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # DCAE 모델 로드 (frozen)
        self.dcae_model = self._load_dcae(dcae_checkpoint)
        
        # Generator 모델 생성 (1.5B 파라미터)
        self.generator_model = self._create_generator()
        
        # 손실 함수 (REPA Loss 포함)
        self.loss_fn = self._create_loss_function()
        
        # 옵티마이저 및 스케줄러
        self.optimizer = torch.optim.AdamW(
            self.generator_model.parameters(),
            lr=config.generator.learning_rate,
            weight_decay=config.generator.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # 학습률 스케줄러
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.generator.epochs,
            eta_min=config.generator.learning_rate * 0.01
        )
        
        # 그래디언트 스케일러 (FP16)
        self.scaler = GradScaler(
            init_scale=1024.0,
            growth_factor=1.02,
            backoff_factor=0.5,
            growth_interval=100
        )
        
        logger.info(f"LYRO Generator Trainer initialized on {self.device}")
        logger.info(f"Generator parameters: {self.generator_model.count_parameters():,}")
    
    def _load_dcae(self, checkpoint_path: str = None):
        """DCAE 모델 로드"""
        dcae_model = create_dcae(
            sample_rate=self.config.dcae.sample_rate,
            latent_channels=self.config.dcae.latent_channels,
            target_compression_ratio=self.config.dcae.estimate_compression_ratio()
        ).to(self.device).eval()
        
        # 체크포인트 로드
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            dcae_model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded DCAE from {checkpoint_path}")
        else:
            logger.warning("No DCAE checkpoint provided, using random weights")
        
        # 파라미터 고정
        for param in dcae_model.parameters():
            param.requires_grad = False
        
        return dcae_model
    
    def _create_generator(self):
        """1.5B 파라미터 Generator 생성"""
        generator_config = GeneratorConfig(
            # 1.5B 파라미터 타겟
            d_model=1536,
            n_layers=24,
            n_heads=24,
            d_ff=6144,
            
            # SSM 설정
            d_state=64,
            d_conv=4,
            expand_factor=2,
            
            # 입출력 설정
            latent_channels=self.config.dcae.latent_channels,
            latent_time_steps=self.config.dcae.target_time_steps,
            
            # Flow Matching 설정
            flow_steps=50,
            sigma_min=1e-4,
            sigma_max=1.0,
            cfg_scale=7.5,
            
            # 기타
            dropout=0.1
        )
        
        generator = create_lyro_generator(generator_config).to(self.device)
        
        # FP16으로 변환
        generator = generator.half()
        
        return generator
    
    def _create_loss_function(self):
        """REPA Loss가 포함된 손실 함수 생성"""
        # 개별 손실 함수들
        flow_loss = FlowMatchingLoss(
            beta_schedule="cosine",
            num_timesteps=1000
        )
        
        # REPA Loss (ZhenYe234/hubert_base_general_audio)
        repa_loss = REPALoss(
            hubert_model="ZhenYe234/hubert_base_general_audio",
            sample_rate=self.config.dcae.sample_rate,
            layer_weights=[0.1, 0.2, 0.3, 0.3, 0.1]  # 중간 레이어 강조
        )
        
        recon_loss = ReconstructionLoss(
            l1_weight=1.0,
            l2_weight=0.1,
            use_multi_scale=True
        )
        
        perceptual_loss = PerceptualLoss(
            sample_rate=self.config.dcae.sample_rate,
            mel_weight=1.0,
            stft_weight=0.5
        )
        
        # 결합 손실
        combined_loss = CombinedLoss(
            flow_loss=flow_loss,
            repa_loss=repa_loss,
            recon_loss=recon_loss,
            perceptual_loss=perceptual_loss,
            weights={
                'flow': 1.0,        # Flow Matching Loss
                'repa': 0.1,        # REPA Loss (HuBERT 기반)
                'recon': 0.3,       # Reconstruction Loss
                'perceptual': 0.2   # Perceptual Loss
            }
        )
        
        return combined_loss
    
    def train_step(self, batch):
        """훈련 스텝"""
        # 데이터 준비
        audio = batch['audio'].to(self.device, dtype=torch.float16)
        lyrics = batch.get('lyrics')
        lyrics_mask = batch.get('lyrics_mask')
        captions = batch.get('captions')
        task_types = batch.get('task_types', ['SONG'] * audio.shape[0])
        
        # DCAE로 잠재 벡터 인코딩
        with torch.no_grad():
            try:
                latents, _ = self.dcae_model.encode(audio.float())
                latents = latents.half()
            except Exception as e:
                logger.error(f"DCAE encoding failed: {e}")
                return None
        
        # Generator 훈련
        with autocast():
            # Flow Matching 훈련 손실
            loss_dict = self.generator_model.training_loss(
                latents=latents.float(),
                lyrics=lyrics,
                lyrics_mask=lyrics_mask,
                captions=captions,
                task_type=task_types[0] if task_types else 'SONG'
            )
            
            flow_loss = loss_dict['flow_loss']
            
            # 추가 손실 계산을 위해 샘플 생성
            try:
                # 간단한 생성 (한 스텝)
                with torch.no_grad():
                    generated_latents = latents + 0.1 * loss_dict.get('predicted_v', torch.zeros_like(latents))
                    generated_audio = self.dcae_model.decode(generated_latents.float()).half()
                
                # 추가 손실들 계산
                additional_losses = self.loss_fn(
                    predicted_v=loss_dict.get('predicted_v', torch.zeros_like(latents)),
                    target_v=loss_dict.get('target_v', torch.zeros_like(latents)),
                    predicted_audio=generated_audio.float(),
                    target_audio=audio.float(),
                    t=torch.rand(latents.shape[0], device=self.device)
                )
                
                # 전체 손실
                total_loss = flow_loss + 0.1 * additional_losses['total']
                
                result_dict = {
                    'loss': total_loss,
                    'flow_loss': flow_loss,
                    'repa_loss': additional_losses.get('repa', torch.tensor(0.0)),
                    'recon_loss': additional_losses.get('recon', torch.tensor(0.0)),
                    'perceptual_loss': additional_losses.get('perceptual', torch.tensor(0.0))
                }
                
            except Exception as e:
                logger.warning(f"Additional loss calculation failed: {e}")
                result_dict = {
                    'loss': flow_loss,
                    'flow_loss': flow_loss,
                    'repa_loss': torch.tensor(0.0),
                    'recon_loss': torch.tensor(0.0),
                    'perceptual_loss': torch.tensor(0.0)
                }
        
        # 백워드 패스
        self.optimizer.zero_grad()
        self.scaler.scale(result_dict['loss']).backward()
        
        # 그래디언트 클리핑
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.generator_model.parameters(), 
            self.config.generator.grad_clip
        )
        
        # 옵티마이저 스텝
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        result_dict['grad_norm'] = grad_norm
        return result_dict
    
    def validate_step(self, batch):
        """검증 스텝"""
        audio = batch['audio'].to(self.device, dtype=torch.float16)
        lyrics = batch.get('lyrics')
        lyrics_mask = batch.get('lyrics_mask')
        captions = batch.get('captions')
        task_types = batch.get('task_types', ['SONG'] * audio.shape[0])
        
        with torch.no_grad():
            # DCAE 인코딩
            latents, _ = self.dcae_model.encode(audio.float())
            latents = latents.half()
            
            # Generator 검증
            with autocast():
                loss_dict = self.generator_model.training_loss(
                    latents=latents.float(),
                    lyrics=lyrics,
                    lyrics_mask=lyrics_mask,
                    captions=captions,
                    task_type=task_types[0] if task_types else 'SONG'
                )
        
        return {
            'loss': loss_dict['flow_loss'],
            'flow_loss': loss_dict['flow_loss']
        }


def main():
    parser = argparse.ArgumentParser(description='LYRO Generator Training (1.5B Parameters)')
    
    # 기본 설정
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--dataset_root', type=str, default='dataset', help='Dataset root directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/generator', help='Checkpoint directory')
    parser.add_argument('--dcae_checkpoint', type=str, required=True, help='DCAE checkpoint path')
    
    # 훈련 설정
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (small for 1.5B model)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--audio_duration', type=float, default=5.0, help='Audio duration in seconds')
    
    # 기술적 설정
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 설정 로드
    if args.config:
        config = LyroConfig.from_yaml(args.config)
    else:
        config = LyroConfig()
        config.generator.epochs = args.epochs
        config.generator.batch_size = args.batch_size
        config.generator.learning_rate = args.learning_rate
        config.dcae.audio_duration = args.audio_duration
        config.training.checkpoint_dir = args.checkpoint_dir
    
    logger.info("Starting LYRO Generator Training (1.5B Parameters)")
    logger.info(f"DCAE checkpoint: {args.dcae_checkpoint}")
    
    # 토크나이저
    tokenizer = LyroTokenizer(vocab_size=32000)
    
    # 데이터셋 생성
    train_dataset, val_dataset, _ = create_lyro_datasets(
        train_metadata=f"{args.dataset_root}/metadata/train_metadata.jsonl",
        val_metadata=f"{args.dataset_root}/metadata/val_metadata.jsonl",
        test_metadata=None,
        dataset_root=args.dataset_root,
        tokenizer=tokenizer,
        max_audio_length=int(config.dcae.sample_rate * config.dcae.audio_duration),
        task_ratios={'SONG': 0.6, 'INST': 0.3, 'COVER': 0.1}
    )
    
    logger.info(f"Dataset loaded: train={len(train_dataset)}, val={len(val_dataset)}")
    
    # 콜레이터
    collator = LyroCollator(
        tokenizer=tokenizer,
        max_audio_length=int(config.dcae.sample_rate * config.dcae.audio_duration),
        max_text_length=512
    )
    
    # 데이터 로더 (작은 배치 크기)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.generator.batch_size,
        shuffle=True,
        num_workers=2,  # 1.5B 모델에는 작은 워커 수
        pin_memory=True,
        collate_fn=collator,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.generator.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collator
    )
    
    # 트레이너 생성
    trainer = LyroGeneratorTrainer(config, args.dcae_checkpoint)
    
    # 체크포인트 디렉토리 생성
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 훈련 루프
    best_val_loss = float('inf')
    
    logger.info("Starting training loop...")
    
    for epoch in range(config.generator.epochs):
        # 훈련
        trainer.generator_model.train()
        train_losses = []
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                loss_dict = trainer.train_step(batch)
                if loss_dict:
                    train_losses.append(loss_dict)
                    
                    if batch_idx % 10 == 0:
                        logger.info(
                            f"Epoch {epoch}, Batch {batch_idx}: "
                            f"Loss={loss_dict['loss']:.4f}, "
                            f"Flow={loss_dict['flow_loss']:.4f}, "
                            f"REPA={loss_dict['repa_loss']:.4f}, "
                            f"Grad={loss_dict['grad_norm']:.3f}"
                        )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("OOM error, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                else:
                    logger.error(f"Training step failed: {e}")
                    continue
            except Exception as e:
                logger.error(f"Training step failed: {e}")
                continue
        
        # 검증
        trainer.generator_model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 20:  # 검증 배치 제한
                    break
                try:
                    loss_dict = trainer.validate_step(batch)
                    val_losses.append(loss_dict)
                except Exception as e:
                    logger.error(f"Validation step failed: {e}")
                    continue
        
        # 평균 계산
        epoch_time = time.time() - epoch_start_time
        
        if train_losses and val_losses:
            avg_train_loss = sum(d['loss'].item() for d in train_losses) / len(train_losses)
            avg_val_loss = sum(d['loss'].item() for d in val_losses) / len(val_losses)
            
            logger.info(
                f"Epoch {epoch} ({epoch_time:.1f}s): "
                f"Train Loss={avg_train_loss:.4f}, "
                f"Val Loss={avg_val_loss:.4f}"
            )
            
            # 체크포인트 저장
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
            
            if epoch % 5 == 0 or is_best:
                checkpoint_path = checkpoint_dir / f"generator_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.generator_model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'scaler_state_dict': trainer.scaler.state_dict(),
                    'config': config,
                    'metrics': {
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss
                    }
                }, checkpoint_path)
                
                if is_best:
                    best_path = checkpoint_dir / "generator_best.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': trainer.generator_model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'scheduler_state_dict': trainer.scheduler.state_dict(),
                        'scaler_state_dict': trainer.scaler.state_dict(),
                        'config': config,
                        'metrics': {
                            'train_loss': avg_train_loss,
                            'val_loss': avg_val_loss
                        }
                    }, best_path)
                    logger.info(f"💾 Best Generator model saved! Loss: {avg_val_loss:.4f}")
        
        # 스케줄러 스텝
        trainer.scheduler.step()
        
        # 메모리 정리
        torch.cuda.empty_cache()
    
    logger.info(f"Training completed! Best Val Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()