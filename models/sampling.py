import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Callable


class ReferenceEncoder(nn.Module):
    def __init__(self, input_channels: int = 16, output_dim: int = 512):
        super().__init__()
        self.processor = nn.Sequential(
            nn.Conv1d(input_channels, input_channels * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(input_channels * 2, input_channels * 4, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_channels * 4, output_dim)
        )
    
    def forward(self, reference: Optional[torch.Tensor]) -> torch.Tensor:
        if reference is None:
            device = next(self.parameters()).device
            return torch.zeros(1, self.processor[-1].out_features, device=device)
        return self.processor(reference)


class CFG:
    """
    간단·안정형 Classifier-Free Guidance (MVP 버전).
    Args:
        scale (float): 초기 guidance scale.
        min_scale (float): 활성 구간 끝에서의 최저 scale.
        active_ratio (float): 총 스텝 중 CFG를 적용할 비율(0~1).
        clip (float): L2 노름 클리핑 임계값.
    """
    def __init__(
        self,
        scale: float = 15.0,
        min_scale: float = 3.0,
        active_ratio: float = 0.5,
        clip: float = 2.5,
    ):
        self.scale = scale
        self.min_scale = min_scale
        self.active_ratio = active_ratio
        self.clip = clip
        
    def apply(self, cond, uncond, idx: int, total: int):
        """
        cond/uncond 예측을 섞어 one-step guidance 출력을 반환.
        Args:
            cond (Tensor): 조건부 예측.
            uncond (Tensor): 무조건 예측.
            idx (int): 현재 스텝 인덱스(0-based).
            total (int): 전체 스텝 수.
        """
        # 1) CFG 활성 구간 계산
        active_len = int(total * self.active_ratio + 0.5)   # round-half-up
        start = (total - active_len) // 2
        end = start + active_len                            # exclusive
        if not (start <= idx < end):
            return cond  # 비활성 구간은 cond 그대로
            
        # 2) 스케일 선형 감소
        prog = (idx - start) / max(active_len - 1, 1)
        cur_scale = self.scale - (self.scale - self.min_scale) * prog
        
        # 3) 노름 클리핑 & 합성
        diff = cond - uncond
        p_norm = diff.norm(p=2, dim=list(range(1, diff.ndim)), keepdim=True)
        scale = torch.clamp(self.clip / (p_norm + 1e-12), max=1.0)
        diff = diff * scale
        return uncond + cur_scale * diff


class EulerSch:
    """
    CFG 구간을 고려한 3-Phase Euler Scheduler (MVP 버전).
    Args:
        steps (int): 총 스텝 수.
        active_ratio (float): CFG 활성 구간 비율(0~1).
        device (str or torch.device): 타임스텝 텐서가 올라갈 디바이스.
    """
    def __init__(self, steps: int = 50, active_ratio: float = 0.5, device="cpu"):
        self.steps = steps
        # ── CFG 구간 인덱스 계산 ──
        active_len = int(steps * active_ratio + 0.5)
        self.cfg_start = (steps - active_len) // 2
        self.cfg_end = self.cfg_start + active_len          # exclusive
        
        # ── 3-Phase time schedule (double precision) ──
        p1 = torch.linspace(1.0, 0.6, self.cfg_start, device=device)
        p2 = torch.linspace(0.6, 0.2, active_len, device=device)
        p3 = torch.linspace(0.2, 0.0, steps - self.cfg_end, device=device)
        self.t = torch.cat([p1, p2, p3]).to(torch.float64)
        
    def step(self, model_out, idx: int, x):
        """
        Euler-Maruyama 1-step 업데이트.
        Args:
            model_out (Tensor): 모델 예측(이미 CFG 적용된 값).
            idx (int): 현재 스텝 인덱스.
            x (Tensor): 현재 샘플.
        Returns:
            Tensor: 업데이트된 샘플.
        """
        # α 값: 3-Phase 적응
        if idx < self.cfg_start:
            alpha = 1.2
        elif idx < self.cfg_end:
            alpha = 1.0
        else:
            alpha = 0.8
            
        # dt 계산 (마지막 스텝은 t 자체)
        dt = self.t[idx] - self.t[idx + 1] if idx + 1 < self.steps else self.t[idx]
        
        # Ensure dt is scalar
        if hasattr(dt, 'item'):
            dt = dt.item()
        
        # Ensure all operations are on tensors
        return x - alpha * float(dt) * model_out


class FlowMatchingSampler:
    """
    CFG + Euler 스케줄러를 통합한 Flow Matching 샘플러
    """
    
    def __init__(
        self,
        cfg_scale: float = 15.0,
        cfg_min_scale: float = 3.0,
        cfg_active_ratio: float = 0.5,
        cfg_clip: float = 2.5,
        steps: int = 50,
        device: str = "cpu"
    ):
        self.cfg = CFG(
            scale=cfg_scale,
            min_scale=cfg_min_scale,
            active_ratio=cfg_active_ratio,
            clip=cfg_clip
        )
        self.scheduler = EulerSch(
            steps=steps,
            active_ratio=cfg_active_ratio,
            device=device
        )
        self.steps = steps
        
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        conditions: dict,
        device: torch.device = None,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Flow Matching 샘플링 실행
        
        Args:
            model: 생성 모델
            shape: 생성할 텐서 형태
            conditions: 조건 딕셔너리
            device: 디바이스
            verbose: 진행상황 출력
            
        Returns:
            생성된 샘플
        """
        if device is None:
            device = next(model.parameters()).device
            
        # 초기 노이즈
        x = torch.randn(shape, device=device)
        
        if verbose:
            print(f"Starting Flow Matching sampling with {self.steps} steps...")
        
        for i in range(self.steps):
            # 현재 시간스텝
            t = torch.full((shape[0],), self.scheduler.t[i], device=device)
            
            # 조건부 예측
            cond_output = model(x, t, **conditions)
            cond_pred = cond_output[0] if isinstance(cond_output, tuple) else cond_output
            
            # 무조건 예측 (조건 제거)
            uncond_conditions = self._create_uncond_conditions(conditions)
            uncond_output = model(x, t, **uncond_conditions)
            uncond_pred = uncond_output[0] if isinstance(uncond_output, tuple) else uncond_output
            
            # CFG 적용
            guided_pred = self.cfg.apply(cond_pred, uncond_pred, i, self.steps)
            
            # Euler 스텝
            x = self.scheduler.step(guided_pred, i, x)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Step {i + 1}/{self.steps} completed")
        
        return x
    
    def _create_uncond_conditions(self, conditions: dict) -> dict:
        """무조건 조건 생성"""
        uncond = {}
        for key, value in conditions.items():
            if value is None:
                uncond[key] = None
            elif isinstance(value, torch.Tensor):
                uncond[key] = torch.zeros_like(value)
            elif isinstance(value, list):
                uncond[key] = [None] * len(value) if value else []
            else:
                uncond[key] = None
        return uncond


class DPMSolverSampler:
    """
    DPM-Solver 기반 빠른 샘플러 (선택적)
    """
    
    def __init__(self, steps: int = 20, device: str = "cpu"):
        self.steps = steps
        self.device = device
        
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        conditions: dict,
        device: torch.device = None
    ) -> torch.Tensor:
        """DPM-Solver 샘플링 (간단한 구현)"""
        if device is None:
            device = next(model.parameters()).device
            
        x = torch.randn(shape, device=device)
        
        # 시간 스케줄
        timesteps = torch.linspace(1.0, 0.0, self.steps + 1, device=device)
        
        for i in range(self.steps):
            t = torch.full((shape[0],), timesteps[i], device=device)
            
            # 1차 예측
            pred = model(x, t, **conditions)
            
            # DPM 업데이트 (간소화)
            dt = timesteps[i] - timesteps[i + 1]
            x = x - dt * pred
            
        return x


def create_sampler(
    sampler_type: str = "flow_matching",
    cfg_scale: float = 15.0,
    steps: int = 50,
    device: str = "cpu",
    **kwargs
):
    """
    샘플러 팩토리 함수
    
    Args:
        sampler_type: 샘플러 종류 ("flow_matching", "dpm_solver")
        cfg_scale: CFG 스케일
        steps: 샘플링 스텝 수
        device: 디바이스
        **kwargs: 추가 파라미터
        
    Returns:
        초기화된 샘플러
    """
    if sampler_type == "flow_matching":
        return FlowMatchingSampler(
            cfg_scale=cfg_scale,
            steps=steps,
            device=device,
            **kwargs
        )
    elif sampler_type == "dpm_solver":
        return DPMSolverSampler(
            steps=steps,
            device=device
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")