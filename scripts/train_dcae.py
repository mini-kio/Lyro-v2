#!/usr/bin/env python3
# lyro/scripts/train_dcae.py
"""
DCAE 훈련 스크립트 - Flow Matching 적용
SNR 성능 개선을 위한 향상된 훈련
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lyro.models.dcae import create_dcae
from lyro.models.losses import FlowMatchingLoss, ReconstructionLoss, PerceptualLoss
from lyro.data.dataset import create_lyro_datasets
from lyro.data.collator import LyroDCAECollator
from lyro.data.tokenizer import LyroTokenizer
from lyro.training.config import LyroConfig
from lyro.training.trainer import create_trainer
from lyro.utils import get_audio_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowMatchingDCAE(nn.Module):
    """Flow Matching이 적용된 DCAE"""
    
    def __init__(self, dcae_model):
        super().__init__()
        self.dcae = dcae_model
        
        # Flow Matching 컴포넌트
        self.flow_matching = FlowMatchingLoss(
            beta_schedule="cosine",
            num_timesteps=1000
        )
        
    def forward(self, audio):
        """포워드 패스"""
        # 기본 DCAE 인코딩/디코딩
        latents, quantization_loss = self.dcae.encode(audio)
        reconstructed = self.dcae.decode(latents)
        
        return reconstructed, quantization_loss, latents
    
    def flow_matching_loss(self, audio):
        """Flow Matching 손실 계산"""
        with torch.no_grad():
            latents, _ = self.dcae.encode(audio)
        
        # 노이즈 샘플링
        noise = torch.randn_like(latents)
        
        # 시간 샘플링
        batch_size = latents.shape[0]
        t = torch.rand(batch_size, device=latents.device)
        
        # Flow matching
        t_expanded = t.view(-1, 1, 1)
        noisy_latents = (1 - t_expanded) * noise + t_expanded * latents
        target_velocity = latents - noise
        
        # 속도 예측 (간단한 MLP)
        if not hasattr(self, 'velocity_predictor'):
            latent_dim = latents.shape[1] * latents.shape[2]
            self.velocity_predictor = nn.Sequential(
                nn.Linear(latent_dim + 1, latent_dim * 2),
                nn.GELU(),
                nn.Linear(latent_dim * 2, latent_dim),
            ).to(latents.device)
        
        # 입력 준비
        flat_latents = noisy_latents.flatten(1)
        time_input = t.unsqueeze(1)
        velocity_input = torch.cat([flat_latents, time_input], dim=1)
        
        # 속도 예측
        predicted_velocity = self.velocity_predictor(velocity_input)
        predicted_velocity = predicted_velocity.view_as(target_velocity)
        
        # Flow matching 손실
        flow_loss = self.flow_matching(
            predicted_v=predicted_velocity,
            target_v=target_velocity,
            t=t
        )
        
        return flow_loss


def create_enhanced_dcae(config: LyroConfig):
    """향상된 DCAE 모델 생성"""
    # 기본 DCAE 생성
    dcae = create_dcae(
        sample_rate=config.dcae.sample_rate,
        latent_channels=config.dcae.latent_channels,
        target_compression_ratio=config.dcae.estimate_compression_ratio(),
        use_quantization=config.dcae.use_quantization
    )
    
    # Flow Matching 래퍼로 감싸기
    enhanced_dcae = FlowMatchingDCAE(dcae)
    
    return enhanced_dcae


def create_enhanced_loss(config: LyroConfig):
    """향상된 손실 함수 생성"""
    class EnhancedDCAELoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.recon_loss = ReconstructionLoss(
                l1_weight=1.0,
                l2_weight=0.1,
                use_multi_scale=True
            )
            self.perceptual_loss = PerceptualLoss(
                sample_rate=config.dcae.sample_rate
            )
            
        def forward(self, model, audio, reconstructed, quantization_loss, latents):
            # 기본 재구성 손실
            recon_loss = self.recon_loss(reconstructed, audio)
            
            # 지각적 손실
            perceptual_loss = self.perceptual_loss(reconstructed, audio)
            
            # Flow Matching 손실
            flow_loss = model.flow_matching_loss(audio)
            
            # 전체 손실
            total_loss = (
                recon_loss + 
                0.2 * perceptual_loss + 
                0.1 * flow_loss
            )
            
            if quantization_loss is not None:
                total_loss = total_loss + 0.1 * quantization_loss
            
            return {
                'loss': total_loss,
                'recon_loss': recon_loss,
                'perceptual_loss': perceptual_loss,
                'flow_loss': flow_loss,
                'quantization_loss': quantization_loss.item() if quantization_loss else 0.0
            }
    
    return EnhancedDCAELoss()


class EnhancedDCAETrainer:
    """향상된 DCAE 트레이너"""
    
    def __init__(self, config: LyroConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 생성
        self.model = create_enhanced_dcae(config).to(self.device)
        
        # 손실 함수
        self.loss_fn = create_enhanced_loss(config)
        
        # 옵티마이저
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.dcae.learning_rate,
            weight_decay=config.dcae.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # 스케줄러
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.dcae.epochs,
            eta_min=config.dcae.learning_rate * 0.01
        )
        
        # 그래디언트 스케일러
        self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"Enhanced DCAE Trainer initialized on {self.device}")
    
    def train_step(self, batch):
        """훈련 스텝"""
        audio = batch['audio'].to(self.device)
        
        # 포워드 패스
        with torch.cuda.amp.autocast():
            reconstructed, quantization_loss, latents = self.model(audio)
            
            # 손실 계산
            loss_dict = self.loss_fn(
                model=self.model,
                audio=audio,
                reconstructed=reconstructed,
                quantization_loss=quantization_loss,
                latents=latents
            )
        
        # 백워드 패스
        self.optimizer.zero_grad()
        self.scaler.scale(loss_dict['loss']).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss_dict
    
    def validate_step(self, batch):
        """검증 스텝"""
        audio = batch['audio'].to(self.device)
        
        with torch.no_grad():
            reconstructed, quantization_loss, latents = self.model(audio)
            
            loss_dict = self.loss_fn(
                model=self.model,
                audio=audio,
                reconstructed=reconstructed,
                quantization_loss=quantization_loss,
                latents=latents
            )
            
            # SNR 계산
            from lyro.utils.metrics import AudioQualityMetrics
            audio_metrics = AudioQualityMetrics(self.config.dcae.sample_rate)
            snr_result = audio_metrics.compute_snr(audio, reconstructed)
            loss_dict['snr'] = snr_result.value
        
        return loss_dict


def main():
    parser = argparse.ArgumentParser(description='Enhanced DCAE Training with Flow Matching')
    
    # 기본 설정
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--dataset_root', type=str, default='dataset', help='Dataset root directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/dcae', help='Checkpoint directory')
    
    # 훈련 설정
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--audio_duration', type=float, default=2.0, help='Audio duration in seconds')
    
    # 기술적 설정
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 설정 로드
    if args.config:
        config = LyroConfig.from_yaml(args.config)
    else:
        config = LyroConfig()
        config.dcae.epochs = args.epochs
        config.dcae.batch_size = args.batch_size
        config.dcae.learning_rate = args.learning_rate
        config.dcae.audio_duration = args.audio_duration
        config.training.checkpoint_dir = args.checkpoint_dir
    
    logger.info("Starting Enhanced DCAE Training")
    logger.info(f"Configuration: {config.dcae}")
    
    # 토크나이저 (DCAE는 텍스트 불필요하지만 데이터셋 호환성을 위해)
    tokenizer = LyroTokenizer()
    
    # 데이터셋 생성
    train_dataset, val_dataset, _ = create_lyro_datasets(
        train_metadata=f"{args.dataset_root}/metadata/train_metadata.jsonl",
        val_metadata=f"{args.dataset_root}/metadata/val_metadata.jsonl",
        test_metadata=None,
        dataset_root=args.dataset_root,
        tokenizer=tokenizer,
        max_audio_length=int(config.dcae.sample_rate * config.dcae.audio_duration),
        task_ratios={'SONG': 0.4, 'INST': 0.4, 'COVER': 0.2}
    )
    
    logger.info(f"Dataset loaded: train={len(train_dataset)}, val={len(val_dataset)}")
    
    # 콜레이터 (DCAE용)
    collator = LyroDCAECollator(
        max_length=int(config.dcae.sample_rate * config.dcae.audio_duration),
        ensure_stereo=True
    )
    
    # 데이터 로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dcae.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dcae.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator
    )
    
    # 트레이너 생성
    trainer = EnhancedDCAETrainer(config)
    
    # 체크포인트 디렉토리 생성
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 훈련 루프
    best_snr = 0.0
    
    for epoch in range(config.dcae.epochs):
        # 훈련
        trainer.model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                loss_dict = trainer.train_step(batch)
                train_losses.append(loss_dict)
                
                if batch_idx % 50 == 0:
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}: "
                        f"Loss={loss_dict['loss']:.4f}, "
                        f"Recon={loss_dict['recon_loss']:.4f}, "
                        f"Flow={loss_dict['flow_loss']:.4f}"
                    )
            except Exception as e:
                logger.error(f"Training step failed: {e}")
                continue
        
        # 검증
        trainer.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    loss_dict = trainer.validate_step(batch)
                    val_losses.append(loss_dict)
                except Exception as e:
                    logger.error(f"Validation step failed: {e}")
                    continue
        
        # 평균 계산
        if train_losses and val_losses:
            avg_train_loss = sum(d['loss'].item() for d in train_losses) / len(train_losses)
            avg_val_loss = sum(d['loss'].item() for d in val_losses) / len(val_losses)
            avg_val_snr = sum(d['snr'] for d in val_losses) / len(val_losses)
            
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={avg_train_loss:.4f}, "
                f"Val Loss={avg_val_loss:.4f}, "
                f"Val SNR={avg_val_snr:.2f}dB"
            )
            
            # 체크포인트 저장
            is_best = avg_val_snr > best_snr
            if is_best:
                best_snr = avg_val_snr
            
            if epoch % 10 == 0 or is_best:
                checkpoint_path = checkpoint_dir / f"dcae_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'config': config,
                    'metrics': {
                        'train_loss': avg_train_loss,
                        'val_loss': avg_val_loss,
                        'val_snr': avg_val_snr
                    }
                }, checkpoint_path)
                
                if is_best:
                    best_path = checkpoint_dir / "dcae_best.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': trainer.model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'scheduler_state_dict': trainer.scheduler.state_dict(),
                        'config': config,
                        'metrics': {
                            'train_loss': avg_train_loss,
                            'val_loss': avg_val_loss,
                            'val_snr': avg_val_snr
                        }
                    }, best_path)
                    logger.info(f"💾 Best model saved! SNR: {avg_val_snr:.2f}dB")
        
        # 스케줄러 스텝
        trainer.scheduler.step()
    
    logger.info(f"Training completed! Best SNR: {best_snr:.2f}dB")


if __name__ == '__main__':
    main()