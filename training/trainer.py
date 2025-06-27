# lyro/training/trainer.py
"""
LYRO 통합 트레이너
DCAE와 Generator 모델을 위한 통합 훈련 시스템
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

# LYRO 모듈
from .config import LyroConfig
from .utils import (
    TrainingMetrics, MemoryManager, TensorValidator, 
    LearningRateScheduler, CheckpointManager, LoggingManager
)
from ..utils.metrics import MetricCalculator
from ..utils.audio import AudioProcessor
from ..models.losses import CombinedLoss

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


class BaseTrainer(ABC):
    """기본 트레이너 추상 클래스"""
    
    def __init__(
        self,
        config: LyroConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device = None
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
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
        
        # 모델을 디바이스로 이동
        self.model = self.model.to(self.device)
        
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
        
        logger.info(f"Initialized {self.__class__.__name__} for {type(model).__name__}")
    
    def _setup_optimization(self):
        """최적화 설정"""
        # 옵티마이저
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
    
    @abstractmethod
    def _setup_loss_function(self):
        """손실 함수 설정 (서브클래스에서 구현)"""
        pass
    
    @abstractmethod
    def forward_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """포워드 스텝 (서브클래스에서 구현)"""
        pass
    
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
        """단일 훈련 스텝"""
        # 배치를 디바이스로 이동
        batch = self._move_batch_to_device(batch)
        
        # 그래디언트 누적 여부 확인
        accumulate_grad = (
            self.state.step % self.config.generator.gradient_accumulation_steps != 0
        )
        
        # 포워드 패스
        if self.scaler:
            with autocast():
                loss_dict = self.forward_step(batch)
        else:
            loss_dict = self.forward_step(batch)
        
        if not loss_dict or 'loss' not in loss_dict:
            return None
        
        # 손실 정규화 (그래디언트 누적)
        loss = loss_dict['loss'] / self.config.generator.gradient_accumulation_steps
        
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
            loss_dict['grad_norm'] = grad_norm
        
        # 결과 반환 (실제 손실로 복원)
        result = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        result['loss'] = result['loss'] * self.config.generator.gradient_accumulation_steps
        result['learning_rate'] = self.scheduler.get_lr()[0]
        
        return result
    
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
                    
                    # 포워드 패스
                    if self.scaler:
                        with autocast():
                            loss_dict = self.forward_step(batch)
                    else:
                        loss_dict = self.forward_step(batch)
                    
                    if loss_dict:
                        result = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
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
    
    def train(self):
        """메인 훈련 루프"""
        logger.info("Starting training...")
        
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
        
        logger.info("Training completed!")
        return self.state.last_checkpoint_path
    
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


class DCAETrainer(BaseTrainer):
    """DCAE 트레이너"""
    
    def _setup_loss_function(self):
        """DCAE 손실 함수 설정"""
        from ..models.losses import SafeDCAELoss
        self.loss_fn = SafeDCAELoss()
    
    def forward_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """DCAE 포워드 스텝"""
        # 오디오 데이터 가져오기
        audio = batch.get('audio')
        if audio is None:
            return None
        
        # 텐서 검증
        if not self.tensor_validator.is_valid(audio, "input_audio"):
            return None
        
        # 모델 포워드
        try:
            reconstructed, quantization_loss = self.model(audio)
            
            # 재구성 손실
            recon_loss = self.loss_fn(reconstructed, audio)
            
            # 전체 손실
            total_loss = recon_loss
            if quantization_loss is not None:
                total_loss = total_loss + 0.1 * quantization_loss
            
            # 추가 메트릭 계산
            with torch.no_grad():
                audio_metrics = self.metrics_calculator.compute_all_metrics(
                    target_audio=audio,
                    generated_audio=reconstructed
                )
                
                snr = audio_metrics.get('snr', None)
                snr_value = snr.value if snr else 0.0
            
            return {
                'loss': total_loss,
                'recon_loss': recon_loss,
                'quantization_loss': quantization_loss.item() if quantization_loss else 0.0,
                'snr': snr_value
            }
            
        except Exception as e:
            logger.error(f"DCAE forward failed: {e}")
            return None


class GeneratorTrainer(BaseTrainer):
    """Generator 트레이너"""
    
    def __init__(
        self,
        config: LyroConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dcae_model: Optional[nn.Module] = None,
        device: torch.device = None
    ):
        self.dcae_model = dcae_model
        super().__init__(config, model, train_loader, val_loader, device)
        
        # DCAE 모델 설정
        if self.dcae_model:
            self.dcae_model = self.dcae_model.to(self.device).eval()
            # DCAE 파라미터 고정
            for param in self.dcae_model.parameters():
                param.requires_grad = False
    
    def _setup_loss_function(self):
        """Generator 손실 함수 설정"""
        from ..models.losses import CombinedLoss, FlowMatchingLoss, REPALoss, ReconstructionLoss, PerceptualLoss
        
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
    
    def forward_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Generator 포워드 스텝"""
        # 오디오 데이터 가져오기
        audio = batch.get('audio')
        if audio is None:
            return None
        
        # DCAE로 잠재 벡터 생성
        if self.dcae_model:
            with torch.no_grad():
                try:
                    latents, _ = self.dcae_model.encode(audio)
                except Exception as e:
                    logger.error(f"DCAE encoding failed: {e}")
                    return None
        else:
            # DCAE가 없으면 더미 잠재 벡터
            latents = torch.randn(
                audio.shape[0], 
                self.config.dcae.latent_channels,
                self.config.dcae.target_time_steps,
                device=self.device
            )
        
        # 조건들 준비
        lyrics = batch.get('lyrics')
        lyrics_mask = batch.get('lyrics_mask')
        captions = batch.get('captions')  # List[str]
        reference_audio = batch.get('reference_audio')
        task_types = batch.get('task_types', ['SONG'] * latents.shape[0])
        
        # 모델 훈련 손실 계산
        try:
            loss_dict = self.model.training_loss(
                latents=latents,
                lyrics=lyrics,
                lyrics_mask=lyrics_mask,
                captions=captions,
                reference_audio=reference_audio,
                task_type=task_types[0] if task_types else 'SONG'
            )
            
            # 예측된 벡터로 오디오 재구성 (REPA 손실용)
            if self.dcae_model and 'predicted_v' in loss_dict:
                with torch.no_grad():
                    try:
                        # 간단한 오일러 적분으로 생성된 잠재 벡터 근사
                        generated_latents = latents + 0.1 * loss_dict['predicted_v']  # 작은 스텝
                        generated_audio = self.dcae_model.decode(generated_latents)
                        
                        # REPA 및 기타 손실들 계산
                        additional_losses = self.loss_fn(
                            predicted_v=loss_dict['predicted_v'],
                            target_v=loss_dict['target_v'],
                            predicted_audio=generated_audio,
                            target_audio=audio,
                            t=torch.rand(latents.shape[0], device=self.device)
                        )
                        
                        # 손실 결합
                        total_loss = loss_dict['flow_loss'] + additional_losses['total'] * 0.1
                        
                        return {
                            'loss': total_loss,
                            'flow_loss': loss_dict['flow_loss'],
                            'repa_loss': additional_losses.get('repa', torch.tensor(0.0)),
                            'recon_loss': additional_losses.get('recon', torch.tensor(0.0)),
                            'perceptual_loss': additional_losses.get('perceptual', torch.tensor(0.0))
                        }
                    
                    except Exception as e:
                        logger.warning(f"Additional loss calculation failed: {e}")
                        return {'loss': loss_dict['flow_loss'], 'flow_loss': loss_dict['flow_loss']}
            else:
                return {'loss': loss_dict['flow_loss'], 'flow_loss': loss_dict['flow_loss']}
            
        except Exception as e:
            logger.error(f"Generator forward failed: {e}")
            return None


def create_trainer(
    model_type: str,
    config: LyroConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dcae_model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None
) -> BaseTrainer:
    """
    트레이너 팩토리 함수
    
    Args:
        model_type: 'dcae' 또는 'generator'
        config: 설정
        model: 훈련할 모델
        train_loader: 훈련 데이터 로더
        val_loader: 검증 데이터 로더
        dcae_model: DCAE 모델 (Generator 훈련시 필요)
        device: 디바이스
    
    Returns:
        해당 트레이너 인스턴스
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type.lower() == 'dcae':
        return DCAETrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
    
    elif model_type.lower() == 'generator':
        return GeneratorTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            dcae_model=dcae_model,
            device=device
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class MultiStageTrainer:
    """다단계 훈련 관리자 (DCAE -> Generator)"""
    
    def __init__(
        self,
        config: LyroConfig,
        dcae_model: nn.Module,
        generator_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.dcae_model = dcae_model
        self.generator_model = generator_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 개별 트레이너들
        self.dcae_trainer = None
        self.generator_trainer = None
    
    def train_dcae(self, dcae_checkpoint: Optional[str] = None) -> Path:
        """DCAE 훈련"""
        logger.info("Starting DCAE training...")
        
        self.dcae_trainer = DCAETrainer(
            config=self.config,
            model=self.dcae_model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device
        )
        
        if dcae_checkpoint:
            self.dcae_trainer.resume_from_checkpoint(dcae_checkpoint)
        
        dcae_checkpoint_path = self.dcae_trainer.train()
        
        logger.info(f"DCAE training completed. Checkpoint: {dcae_checkpoint_path}")
        return dcae_checkpoint_path
    
    def train_generator(self, dcae_checkpoint_path: Path, generator_checkpoint: Optional[str] = None) -> Path:
        """Generator 훈련"""
        logger.info("Starting Generator training...")
        
        # DCAE 체크포인트 로드
        if dcae_checkpoint_path and dcae_checkpoint_path.exists():
            dcae_checkpoint = torch.load(dcae_checkpoint_path, map_location=self.device)
            self.dcae_model.load_state_dict(dcae_checkpoint['model_state_dict'])
            logger.info(f"Loaded DCAE checkpoint: {dcae_checkpoint_path}")
        
        self.generator_trainer = GeneratorTrainer(
            config=self.config,
            model=self.generator_model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            dcae_model=self.dcae_model,
            device=self.device
        )
        
        if generator_checkpoint:
            self.generator_trainer.resume_from_checkpoint(generator_checkpoint)
        
        generator_checkpoint_path = self.generator_trainer.train()
        
        logger.info(f"Generator training completed. Checkpoint: {generator_checkpoint_path}")
        return generator_checkpoint_path
    
    def train_full_pipeline(
        self, 
        dcae_checkpoint: Optional[str] = None,
        generator_checkpoint: Optional[str] = None,
        skip_dcae: bool = False
    ) -> Tuple[Optional[Path], Path]:
        """전체 파이프라인 훈련"""
        dcae_checkpoint_path = None
        
        # DCAE 훈련 (필요시)
        if not skip_dcae:
            dcae_checkpoint_path = self.train_dcae(dcae_checkpoint)
        elif dcae_checkpoint:
            dcae_checkpoint_path = Path(dcae_checkpoint)
        
        # Generator 훈련
        generator_checkpoint_path = self.train_generator(dcae_checkpoint_path, generator_checkpoint)
        
        return dcae_checkpoint_path, generator_checkpoint_path
