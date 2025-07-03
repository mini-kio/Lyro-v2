"""
LYRO Generator 트레이너 (올바른 DCAE + Vocoder 파이프라인)
프리트레인된 DCAE + Vocoder를 사용한 Generator 훈련
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import gc
import warnings

from .config import LyroConfig
from .utils import (
    TrainingMetrics, MemoryManager, TensorValidator, 
    LearningRateScheduler, CheckpointManager, LoggingManager
)
from utils.metrics import MetricCalculator
from utils.audio import AudioProcessor
from models.losses import CombinedLoss
from models.dcae import PretrainedDCAE, AdvancedVocoder

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class TrainerState:
    """트레이너 상태"""
    epoch: int = 0
    step: int = 0
    best_metric: float = float('inf')
    is_best: bool = False
    last_checkpoint_path: Optional[Path] = None


class GeneratorTrainer:
    """
    LYRO Generator 트레이너 (올바른 DCAE + Vocoder 파이프라인)
    오디오 -> 멜 -> DCAE latent -> Generator 훈련 -> DCAE mel -> Vocoder -> 오디오
    """
    
    def __init__(
        self,
        config: LyroConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dcae_model: PretrainedDCAE,
        vocoder_model: AdvancedVocoder,
        device: torch.device = None
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dcae_model = dcae_model
        self.vocoder_model = vocoder_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 상태
        self.state = TrainerState()
        
        # 유틸리티 초기화
        self.memory_manager = MemoryManager(self.device)
        self.tensor_validator = TensorValidator()
        self.metrics_calculator = MetricCalculator(config.data.sample_rate)
        self.audio_processor = AudioProcessor()
        
        # 메트릭 추적
        self.train_metrics = TrainingMetrics()
        self.val_metrics = TrainingMetrics()
        
        # 모델들을 디바이스로 이동
        self.model = self.model.to(self.device)
        self.dcae_model = self.dcae_model.to(self.device).eval()
        self.vocoder_model = self.vocoder_model.to(self.device).eval()
        
        # DCAE 및 Vocoder 파라미터 고정
        for param in self.dcae_model.parameters():
            param.requires_grad = False
        for param in self.vocoder_model.parameters():
            param.requires_grad = False
        
        # 최적화 설정
        self._setup_optimization()
        
        # 체크포인트 관리
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(config.training.checkpoint_dir),
            keep_best=config.training.keep_best,
            metric_mode="min"
        )
        
        # 로깅
        self.logging_manager = LoggingManager(
            log_dir=Path(config.training.checkpoint_dir) / "logs",
            use_wandb=config.training.use_wandb
        )
        
        # 손실 함수
        self._setup_loss_function()
        
        logger.info(f"Initialized GeneratorTrainer with corrected DCAE+Vocoder pipeline")
        logger.info(f"Generator: {type(model).__name__}")
        logger.info(f"DCAE: {type(dcae_model).__name__} (frozen)")
        logger.info(f"Vocoder: {type(vocoder_model).__name__} (frozen)")
        logger.info(f"Pipeline: audio->mel->dcae_latent->generator->dcae_mel->vocoder->audio")
    
    def _setup_optimization(self):
        """최적화 설정"""
        # 옵티마이저 (Generator만)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.generator.learning_rate,
            weight_decay=self.config.generator.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # 스케줄러
        total_steps = len(self.train_loader) * self.config.generator.epochs
        warmup_steps = len(self.train_loader) * self.config.generator.warmup_epochs
        
        self.scheduler = LearningRateScheduler(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=0.01,
            schedule="cosine"
        )
        
        # 그래디언트 스케일러 (mixed precision)
        if self.config.generator.mixed_precision == "fp16":
            self.scaler = GradScaler()
        else:
            self.scaler = None
    
    def _setup_loss_function(self):
        """Generator 손실 함수 설정"""
        from models.losses import CombinedLoss, FlowMatchingLoss, REPALoss, ReconstructionLoss, PerceptualLoss
        
        # 개별 손실 함수들
        flow_loss = FlowMatchingLoss()
        repa_loss = REPALoss(hubert_model=self.config.loss.hubert_model)
        recon_loss = ReconstructionLoss()
        perceptual_loss = PerceptualLoss()
        
        # 결합 손실
        self.loss_fn = CombinedLoss(
            flow_loss=flow_loss,
            repa_loss=repa_loss,
            recon_loss=recon_loss,
            perceptual_loss=perceptual_loss,
            weights={
                'flow': self.config.loss.flow_matching_weight,
                'repa': self.config.loss.repa_weight,
                'recon': self.config.loss.reconstruction_weight,
                'perceptual': self.config.loss.perceptual_weight
            }
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """훈련 에폭"""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            step_start_time = time.time()
            
            try:
                # 메모리 모니터링
                with self.memory_manager.monitoring(f"train_step_{batch_idx}"):
                    step_result = self.training_step(batch)
                
                # 메트릭 업데이트
                if step_result:
                    self.train_metrics.update(**step_result)
                    
                    # 로깅
                    if batch_idx % self.config.training.log_interval == 0:
                        self._log_training_step(batch_idx, step_result, time.time() - step_start_time)
                
                self.state.step += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.memory_manager.log_oom()
                    continue
                else:
                    logger.error(f"Training step failed: {e}")
                    continue
            except Exception as e:
                logger.error(f"Unexpected error in training step: {e}")
                continue
        
        # 에폭 메트릭 계산
        epoch_metrics = self.train_metrics.epoch_metrics
        epoch_time = time.time() - epoch_start_time
        
        logger.info(
            f"Epoch {self.state.epoch} completed in {epoch_time:.2f}s, "
            f"avg_loss: {epoch_metrics.get('loss', 0):.4f}"
        )
        
        return epoch_metrics
    
    def training_step(self, batch: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Generator 훈련 스텝 (올바른 DCAE + Vocoder 파이프라인)
        """
        # 배치를 디바이스로 이동
        batch = self._move_batch_to_device(batch)
        
        # 오디오 데이터 가져오기
        audio = batch.get('audio')
        if audio is None:
            return None
        
        # 올바른 파이프라인: 오디오 -> 멜 -> DCAE latent
        with torch.no_grad():
            try:
                # Step 1: 오디오 -> 멜 스펙트로그램
                mel_target = self.dcae_model.audio_to_mel(audio.float())
                
                # Step 2: 멜 -> DCAE latent (target latents)
                latents_target, _ = self.dcae_model.encode_mel(mel_target.float())
                latents_target = latents_target.half() if self.scaler else latents_target
                
            except Exception as e:
                logger.error(f"DCAE encoding failed: {e}")
                return None
        
        # 조건들 준비
        lyrics = batch.get('lyrics')
        lyrics_mask = batch.get('lyrics_mask')
        captions = batch.get('captions')
        reference_audio = batch.get('reference_audio')
        task_types = batch.get('task_types', ['SONG'] * latents_target.shape[0])
        
        # 참조 오디오 처리 (올바른 파이프라인)
        reference_latents = None
        if reference_audio is not None:
            with torch.no_grad():
                try:
                    ref_mel = self.dcae_model.audio_to_mel(reference_audio.float())
                    reference_latents, _ = self.dcae_model.encode_mel(ref_mel.float())
                    reference_latents = reference_latents.half() if self.scaler else reference_latents
                except Exception as e:
                    logger.warning(f"Reference audio processing failed: {e}")
                    reference_latents = None
        
        # 그래디언트 누적 여부 확인
        accumulate_grad = (
            self.state.step % self.config.generator.gradient_accumulation_steps != 0
        )
        
        # Generator 훈련 (Flow Matching)
        if self.scaler:
            with autocast():
                loss_dict = self.model.training_loss(
                    latents=latents_target.float(),
                    lyrics=lyrics,
                    lyrics_mask=lyrics_mask,
                    captions=captions,
                    reference_audio=reference_latents.float() if reference_latents is not None else None,
                    task_type=task_types[0] if task_types else 'SONG'
                )
        else:
            loss_dict = self.model.training_loss(
                latents=latents_target.float(),
                lyrics=lyrics,
                lyrics_mask=lyrics_mask,
                captions=captions,
                reference_audio=reference_latents.float() if reference_latents is not None else None,
                task_type=task_types[0] if task_types else 'SONG'
            )
        
        if not loss_dict or 'flow_loss' not in loss_dict:
            return None
        
        # 기본 flow loss
        flow_loss = loss_dict['flow_loss']
        
        # 추가 손실 계산 (REPA, perceptual 등) - 올바른 파이프라인 사용
        additional_losses = {}
        try:
            predicted_v = loss_dict.get('predicted_v')
            target_v = loss_dict.get('target_v')
            
            if predicted_v is not None and target_v is not None:
                # 생성된 latent로부터 오디오 복원 (올바른 파이프라인)
                with torch.no_grad():
                    # 한 스텝 Flow Matching 근사
                    generated_latents = latents_target + 0.1 * predicted_v
                    
                    # latent -> mel (DCAE 디코딩)
                    generated_mel = self.dcae_model.decode_to_mel(generated_latents.float())
                    
                    # mel -> audio (Vocoder)
                    generated_audio = self.vocoder_model.mel_to_audio(generated_mel)
                
                # 추가 손실들 계산
                additional_losses = self.loss_fn(
                    predicted_v=predicted_v,
                    target_v=target_v,
                    predicted_audio=generated_audio.float(),
                    target_audio=audio.float(),
                    t=torch.rand(latents_target.shape[0], device=self.device)
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
            else:
                result_dict = {
                    'loss': flow_loss,
                    'flow_loss': flow_loss,
                    'repa_loss': torch.tensor(0.0),
                    'recon_loss': torch.tensor(0.0),
                    'perceptual_loss': torch.tensor(0.0)
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
        
        # 손실 정규화 (그래디언트 누적)
        loss = result_dict['loss'] / self.config.generator.gradient_accumulation_steps
        
        # 백워드 패스
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 옵티마이저 스텝 (그래디언트 누적 완료시)
        if not accumulate_grad:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.generator.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.generator.grad_clip
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            # 그래디언트 노름 추가
            result_dict['grad_norm'] = grad_norm
        
        # 결과 반환 (실제 손실로 복원)
        final_result = {k: v.item() if torch.is_tensor(v) else v for k, v in result_dict.items()}
        final_result['loss'] = final_result['loss'] * self.config.generator.gradient_accumulation_steps
        final_result['learning_rate'] = self.scheduler.get_lr()[0]
        
        return final_result
    
    def validate(self) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        self.val_metrics.reset()
        
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    # 메모리 제한으로 일부만 검증
                    if batch_idx >= 50:  # 최대 50 배치
                        break
                    
                    # 배치 처리
                    batch = self._move_batch_to_device(batch)
                    
                    # 검증 스텝
                    result = self.validation_step(batch)
                    
                    if result:
                        self.val_metrics.update(**result)
                
                except Exception as e:
                    logger.warning(f"Validation step {batch_idx} failed: {e}")
                    continue
        
        val_metrics = self.val_metrics.epoch_metrics
        val_time = time.time() - val_start_time
        
        logger.info(
            f"Validation completed in {val_time:.2f}s, "
            f"avg_loss: {val_metrics.get('loss', 0):.4f}"
        )
        
        return val_metrics
    
    def validation_step(self, batch: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """검증 스텝 (올바른 파이프라인)"""
        audio = batch.get('audio')
        if audio is None:
            return None
        
        # 올바른 파이프라인: 오디오 -> 멜 -> DCAE latent
        try:
            # Step 1: 오디오 -> 멜
            mel_target = self.dcae_model.audio_to_mel(audio.float())
            
            # Step 2: 멜 -> DCAE latent
            latents_target, _ = self.dcae_model.encode_mel(mel_target.float())
            
        except Exception as e:
            logger.error(f"DCAE encoding failed during validation: {e}")
            return None
        
        # 조건들 준비
        lyrics = batch.get('lyrics')
        lyrics_mask = batch.get('lyrics_mask')
        captions = batch.get('captions')
        reference_audio = batch.get('reference_audio')
        task_types = batch.get('task_types', ['SONG'] * latents_target.shape[0])
        
        # 참조 오디오 처리
        reference_latents = None
        if reference_audio is not None:
            try:
                ref_mel = self.dcae_model.audio_to_mel(reference_audio.float())
                reference_latents, _ = self.dcae_model.encode_mel(ref_mel.float())
            except Exception:
                reference_latents = None
        
        # Generator 검증
        loss_dict = self.model.training_loss(
            latents=latents_target.float(),
            lyrics=lyrics,
            lyrics_mask=lyrics_mask,
            captions=captions,
            reference_audio=reference_latents.float() if reference_latents is not None else None,
            task_type=task_types[0] if task_types else 'SONG'
        )
        
        if not loss_dict or 'flow_loss' not in loss_dict:
            return None
        
        return {
            'loss': loss_dict['flow_loss'].item(),
            'flow_loss': loss_dict['flow_loss'].item()
        }
    
    def train(self):
        """메인 훈련 루프"""
        logger.info("Starting Generator training with corrected DCAE+Vocoder pipeline...")
        
        # 파이프라인 테스트
        self._test_pipeline()
        
        # 모델 정보 로깅
        self.logging_manager.log_model_info(self.model)
        
        start_epoch = self.state.epoch
        total_epochs = self.config.generator.epochs
        
        for epoch in range(start_epoch, total_epochs):
            self.state.epoch = epoch
            
            # 훈련
            train_metrics = self.train_epoch()
            
            # 검증
            if epoch % self.config.training.eval_interval == 0:
                val_metrics = self.validate()
                
                # 메트릭 결합
                combined_metrics = {
                    **{f'train_{k}': v for k, v in train_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }
                
                # 최고 모델 확인
                val_loss = val_metrics.get('loss', float('inf'))
                is_best = val_loss < self.state.best_metric
                
                if is_best:
                    self.state.best_metric = val_loss
                    self.state.is_best = True
                
                # 체크포인트 저장
                if epoch % self.config.training.save_interval == 0 or is_best:
                    checkpoint_path = self.checkpoint_manager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        step=self.state.step,
                        metrics=combined_metrics,
                        extra_data={'config': self.config}
                    )
                    self.state.last_checkpoint_path = checkpoint_path
                
                # 로깅
                self.logging_manager.log_metrics(combined_metrics, self.state.step)
                
                logger.info(f"Epoch {epoch}: train_loss={train_metrics.get('loss', 0):.4f}, val_loss={val_loss:.4f}")
            
            else:
                # 검증 없이 훈련만
                train_only_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
                self.logging_manager.log_metrics(train_only_metrics, self.state.step)
        
        logger.info("Generator training completed with corrected pipeline!")
        return self.state.last_checkpoint_path
    
    def _test_pipeline(self):
        """파이프라인 올바른 동작 테스트"""
        logger.info("Testing corrected DCAE+Vocoder pipeline...")
        
        try:
            # 테스트 오디오 생성
            test_audio = torch.randn(1, 2, 44100, device=self.device)
            
            # Step 1: 오디오 -> 멜
            mel = self.dcae_model.audio_to_mel(test_audio)
            logger.info(f"  Audio -> Mel: {test_audio.shape} -> {mel.shape}")
            
            # Step 2: 멜 -> DCAE latent
            latents, _ = self.dcae_model.encode_mel(mel)
            logger.info(f"  Mel -> Latent: {mel.shape} -> {latents.shape}")
            
            # Step 3: DCAE latent -> 멜
            reconstructed_mel = self.dcae_model.decode_to_mel(latents)
            logger.info(f"  Latent -> Mel: {latents.shape} -> {reconstructed_mel.shape}")
            
            # Step 4: 멜 -> 오디오
            reconstructed_audio = self.vocoder_model.mel_to_audio(reconstructed_mel)
            logger.info(f"  Mel -> Audio: {reconstructed_mel.shape} -> {reconstructed_audio.shape}")
            
            logger.info("✅ Pipeline test completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline test failed: {e}")
            raise RuntimeError("Pipeline test failed - check DCAE and Vocoder configuration")
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """배치를 디바이스로 이동"""
        device_batch = {}
        
        for key, value in batch.items():
            if torch.is_tensor(value):
                device_batch[key] = value.to(self.device, non_blocking=True)
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _log_training_step(self, batch_idx: int, step_result: Dict[str, float], step_time: float):
        """훈련 스텝 로깅"""
        memory_info = self.memory_manager.get_memory_info()
        
        log_str = (
            f"Epoch {self.state.epoch}, Step {batch_idx}: "
            f"loss={step_result.get('loss', 0):.4f}, "
            f"flow={step_result.get('flow_loss', 0):.4f}, "
            f"repa={step_result.get('repa_loss', 0):.4f}, "
            f"lr={step_result.get('learning_rate', 0):.2e}, "
            f"time={step_time:.2f}s"
        )
        
        if 'grad_norm' in step_result:
            log_str += f", grad_norm={step_result['grad_norm']:.3f}"
        
        if self.device.type == 'cuda':
            log_str += f", mem={memory_info['gpu_allocated_gb']:.1f}GB"
        
        logger.info(log_str)
    
    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]):
        """체크포인트에서 재개"""
        checkpoint_data = self.checkpoint_manager.load(Path(checkpoint_path))
        
        # 모델 상태 로드
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint_data and checkpoint_data['scheduler_state_dict']:
            self.scheduler.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        # 트레이너 상태 복원
        self.state.epoch = checkpoint_data['epoch'] + 1
        self.state.step = checkpoint_data['step']
        self.state.best_metric = checkpoint_data.get('best_metric', float('inf'))
        
        logger.info(f"Resumed training from epoch {self.state.epoch}")


def create_trainer(
    config: LyroConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dcae_model: PretrainedDCAE,
    vocoder_model: AdvancedVocoder,
    device: Optional[torch.device] = None
) -> GeneratorTrainer:
    """
    Generator 트레이너 생성 (올바른 DCAE + Vocoder 파이프라인)
    
    Args:
        config: 설정
        model: Generator 모델
        train_loader: 훈련 데이터 로더
        val_loader: 검증 데이터 로더
        dcae_model: 프리트레인된 DCAE 모델
        vocoder_model: 프리트레인된 Vocoder 모델
        device: 디바이스
    
    Returns:
        GeneratorTrainer 인스턴스
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return GeneratorTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dcae_model=dcae_model,
        vocoder_model=vocoder_model,
        device=device
    )
