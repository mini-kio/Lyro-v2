# lyro/training/utils.py
"""
LYRO 트레이닝 유틸리티
메모리 관리, 메트릭 계산, 체크포인트 관리 등
"""

import torch
import torch.nn as nn
import numpy as np
import gc
import time
import math
import psutil
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from contextlib import contextmanager


# 로깅 설정
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """훈련 메트릭 관리"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.epoch_metrics: Dict[str, float] = {}
        
    def update(self, **kwargs):
        """메트릭 업데이트"""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            self.metrics[key].append(float(value))
    
    def get_average(self, key: str, last_n: Optional[int] = None) -> float:
        """평균 계산"""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        
        values = self.metrics[key]
        if last_n:
            values = values[-last_n:]
        
        return sum(values) / len(values)
    
    def get_latest(self, key: str) -> float:
        """최신 값 가져오기"""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return self.metrics[key][-1]
    
    def reset(self):
        """메트릭 리셋"""
        self.metrics.clear()
        self.epoch_metrics.clear()
    
    def end_epoch(self):
        """에폭 종료 시 평균 저장"""
        for key, values in self.metrics.items():
            if values:
                self.epoch_metrics[key] = sum(values) / len(values)
        self.reset()


class MemoryManager:
    """GPU 메모리 관리"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.peak_memory = 0
        self.oom_count = 0
        
    def cleanup(self):
        """메모리 정리"""
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_info(self) -> Dict[str, float]:
        """메모리 정보 가져오기"""
        info = {
            'cpu_percent': psutil.virtual_memory().percent,
            'cpu_available_gb': psutil.virtual_memory().available / (1024**3),
        }
        
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            
            # GPU 총 메모리
            try:
                total_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
                utilization = allocated / total_memory
            except:
                total_memory = 0
                utilization = 0
            
            info.update({
                'gpu_allocated_gb': allocated,
                'gpu_reserved_gb': reserved,
                'gpu_max_allocated_gb': max_allocated,
                'gpu_total_gb': total_memory,
                'gpu_utilization': utilization,
            })
            
            self.peak_memory = max(self.peak_memory, allocated)
        
        return info
    
    def log_oom(self):
        """OOM 로깅"""
        self.oom_count += 1
        logger.warning(f"OOM event #{self.oom_count}")
        self.cleanup()
    
    @contextmanager
    def monitoring(self, name: str = "operation"):
        """메모리 모니터링 컨텍스트"""
        start_info = self.get_memory_info()
        start_time = time.time()
        
        try:
            yield
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.log_oom()
                raise
        finally:
            end_info = self.get_memory_info()
            duration = time.time() - start_time
            
            if self.device.type == 'cuda':
                memory_diff = end_info['gpu_allocated_gb'] - start_info['gpu_allocated_gb']
                logger.debug(
                    f"{name}: {duration:.2f}s, "
                    f"memory delta: {memory_diff:+.3f}GB, "
                    f"peak: {self.peak_memory:.2f}GB"
                )


class TensorValidator:
    """텐서 검증 유틸리티"""
    
    @staticmethod
    def is_valid(tensor: torch.Tensor, name: str = "tensor") -> bool:
        """텐서 유효성 검사"""
        if tensor is None:
            logger.warning(f"{name} is None")
            return False
        
        if not isinstance(tensor, torch.Tensor):
            logger.warning(f"{name} is not a tensor")
            return False
        
        if tensor.numel() == 0:
            logger.warning(f"{name} is empty")
            return False
        
        if torch.isnan(tensor).any():
            logger.warning(f"{name} contains NaN")
            return False
        
        if torch.isinf(tensor).any():
            logger.warning(f"{name} contains Inf")
            return False
        
        return True
    
    @staticmethod
    def sanitize(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """텐서 정리"""
        if not TensorValidator.is_valid(tensor, name):
            logger.warning(f"Sanitizing {name}")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return tensor
    
    @staticmethod
    def ensure_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        """디바이스 보장"""
        if tensor.device != device:
            tensor = tensor.to(device)
        return tensor
    
    @staticmethod
    def ensure_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """데이터 타입 보장"""
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor


class LearningRateScheduler:
    """학습률 스케줄러"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
        schedule: str = "cosine"
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.schedule = schedule
        
        # 초기 학습률 저장
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
    
    def step(self):
        """스케줄러 스텝"""
        self.current_step += 1
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_step <= self.warmup_steps:
                # Warmup
                lr = base_lr * self.current_step / self.warmup_steps
            else:
                # Main schedule
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                progress = min(1.0, progress)
                
                if self.schedule == "cosine":
                    lr = base_lr * (self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress)))
                elif self.schedule == "linear":
                    lr = base_lr * (1 - progress * (1 - self.min_lr_ratio))
                else:
                    lr = base_lr  # constant
            
            param_group['lr'] = lr
    
    def get_lr(self) -> List[float]:
        """현재 학습률 가져오기"""
        return [group['lr'] for group in self.optimizer.param_groups]


class CheckpointManager:
    """체크포인트 관리"""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        keep_best: int = 5,
        keep_recent: int = 3,
        metric_mode: str = "min"  # min or max
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_best = keep_best
        self.keep_recent = keep_recent
        self.metric_mode = metric_mode
        
        self.checkpoints: List[Dict] = []
        self.best_metric = float('inf') if metric_mode == "min" else float('-inf')
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        extra_data: Optional[Dict] = None
    ) -> Path:
        """체크포인트 저장"""
        
        # 메인 메트릭 확인
        main_metric = metrics.get('val_loss', metrics.get('loss', 0.0))
        is_best = self._is_best_metric(main_metric)
        
        # 체크포인트 데이터
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'best_metric': self.best_metric,
            'timestamp': time.time(),
        }
        
        if extra_data:
            checkpoint_data.update(extra_data)
        
        # 파일명 생성
        if is_best:
            filename = f"best_epoch_{epoch:03d}_step_{step}.pt"
            self.best_metric = main_metric
        else:
            filename = f"checkpoint_epoch_{epoch:03d}_step_{step}.pt"
        
        filepath = self.checkpoint_dir / filename
        
        # 저장
        torch.save(checkpoint_data, filepath)
        
        # 체크포인트 목록 업데이트
        self.checkpoints.append({
            'path': filepath,
            'epoch': epoch,
            'step': step,
            'metric': main_metric,
            'is_best': is_best,
            'timestamp': time.time(),
        })
        
        # 정리
        self._cleanup_checkpoints()
        
        logger.info(f"Saved checkpoint: {filename} (best: {is_best})")
        return filepath
    
    def load(self, checkpoint_path: Path) -> Dict[str, Any]:
        """체크포인트 로드"""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
        return checkpoint_data
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """최신 체크포인트 가져오기"""
        if not self.checkpoints:
            return None
        
        latest = max(self.checkpoints, key=lambda x: x['timestamp'])
        return latest['path']
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """최고 체크포인트 가져오기"""
        best_checkpoints = [cp for cp in self.checkpoints if cp['is_best']]
        if not best_checkpoints:
            return None
        
        if self.metric_mode == "min":
            best = min(best_checkpoints, key=lambda x: x['metric'])
        else:
            best = max(best_checkpoints, key=lambda x: x['metric'])
        
        return best['path']
    
    def _is_best_metric(self, metric: float) -> bool:
        """최고 메트릭인지 확인"""
        if self.metric_mode == "min":
            return metric < self.best_metric
        else:
            return metric > self.best_metric
    
    def _cleanup_checkpoints(self):
        """오래된 체크포인트 정리"""
        # 최신 순으로 정렬
        self.checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Best 체크포인트와 Recent 체크포인트 분리
        best_checkpoints = [cp for cp in self.checkpoints if cp['is_best']]
        recent_checkpoints = [cp for cp in self.checkpoints if not cp['is_best']]
        
        # Best 체크포인트 정리
        if len(best_checkpoints) > self.keep_best:
            to_remove = best_checkpoints[self.keep_best:]
            for cp in to_remove:
                if cp['path'].exists():
                    cp['path'].unlink()
                self.checkpoints.remove(cp)
        
        # Recent 체크포인트 정리
        if len(recent_checkpoints) > self.keep_recent:
            to_remove = recent_checkpoints[self.keep_recent:]
            for cp in to_remove:
                if cp['path'].exists():
                    cp['path'].unlink()
                self.checkpoints.remove(cp)


class GradientClipping:
    """그래디언트 클리핑 유틸리티"""
    
    @staticmethod
    def clip_grad_norm(
        parameters: Union[torch.Tensor, List[torch.Tensor]],
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False
    ) -> float:
        """그래디언트 노름 클리핑"""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        # 파라미터들에서 그래디언트가 있는 것들만 선택
        grads = [p.grad for p in parameters if p.grad is not None]
        
        if not grads:
            return 0.0
        
        # 전체 노름 계산
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type) for g in grads]),
            norm_type
        )
        
        # 클리핑
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef = torch.clamp(clip_coef, max=1.0)
        
        for g in grads:
            g.detach().mul_(clip_coef)
        
        return total_norm.item()
    
    @staticmethod
    def clip_grad_value(
        parameters: Union[torch.Tensor, List[torch.Tensor]],
        clip_value: float
    ):
        """그래디언트 값 클리핑"""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        
        for p in parameters:
            if p.grad is not None:
                p.grad.data.clamp_(-clip_value, clip_value)


class LoggingManager:
    """로깅 관리"""
    
    def __init__(self, log_dir: Path, use_wandb: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # 파일 로거 설정
        self.setup_file_logging()
        
        # Wandb 설정
        if use_wandb:
            self.setup_wandb()
    
    def setup_file_logging(self):
        """파일 로깅 설정"""
        log_file = self.log_dir / "training.log"
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    def setup_wandb(self):
        """Wandb 설정"""
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            logger.warning("Wandb not available")
            self.use_wandb = False
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """메트릭 로깅"""
        # 파일 로깅
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {step} - {metrics_str}")
        
        # Wandb 로깅
        if self.use_wandb:
            self.wandb.log(metrics, step=step)
    
    def log_model_info(self, model: nn.Module):
        """모델 정보 로깅"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        if self.use_wandb:
            self.wandb.log({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
            })


def safe_load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """안전한 체크포인트 로딩"""
    
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
    
    # 모델 로딩
    try:
        model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
        logger.info("Model state loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load model state: {e}")
        if strict:
            raise
    
    # 옵티마이저 로딩
    if optimizer and 'optimizer_state_dict' in checkpoint_data:
        try:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            logger.info("Optimizer state loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
    
    # 스케줄러 로딩
    if scheduler and 'scheduler_state_dict' in checkpoint_data:
        try:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            logger.info("Scheduler state loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load scheduler state: {e}")
    
    return checkpoint_data


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """모델 파라미터 수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
    }


def get_model_size_mb(model: nn.Module) -> float:
    """모델 크기 계산 (MB)"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    return total_size / (1024 ** 2)  # Convert to MB


def format_time(seconds: float) -> str:
    """시간 포맷팅"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_number(num: Union[int, float]) -> str:
    """숫자 포맷팅"""
    if isinstance(num, float):
        if abs(num) < 1e-3:
            return f"{num:.2e}"
        else:
            return f"{num:.4f}"
    else:
        if num >= 1_000_000_000:
            return f"{num/1_000_000_000:.1f}B"
        elif num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return str(num)
