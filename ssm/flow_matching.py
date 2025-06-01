# lyro/ssm/flow_matching.py
"""
Advanced Flow Matching for Lyro
Ultra-fast music generation with 4-16 steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # ✅ 추가: 누락된 import
import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Callable  # ✅ 추가: 누락된 import
import math  # ✅ 추가: 누락된 import
from abc import ABC, abstractmethod  # ✅ 추가: 누락된 import


# ==================== Flow Schedulers ====================

class FlowScheduler(ABC):
    """Abstract flow scheduler"""
    
    @abstractmethod
    def get_timesteps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Get timestep schedule"""
        pass
    
    @abstractmethod
    def scale_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Scale noise for given timestep"""
        pass


class LinearFlowScheduler(FlowScheduler):
    """Linear timestep scheduler"""
    
    def get_timesteps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        return torch.linspace(0, 1, num_steps + 1, device=device)
    
    def scale_noise(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)


class CosineFlowScheduler(FlowScheduler):
    """Cosine scheduler for smoother generation"""
    
    def get_timesteps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        steps = torch.linspace(0, 1, num_steps + 1, device=device)
        return 0.5 * (1 - torch.cos(math.pi * steps))
    
    def scale_noise(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1 - torch.cos(math.pi * t))


class AdvancedFlowScheduler(FlowScheduler):
    """Advanced scheduler with sway sampling for better quality"""
    
    def __init__(self, sway_coeff: float = 0.3):
        self.sway_coeff = sway_coeff
    
    def get_timesteps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        base_steps = torch.linspace(0, 1, num_steps + 1, device=device)
        # Apply sway sampling
        sway_steps = base_steps + self.sway_coeff * (
            torch.cos(math.pi / 2 * base_steps) - 1 + base_steps
        )
        return torch.clamp(sway_steps, 0, 1)
    
    def scale_noise(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)


# ==================== ODE Solvers ====================

class ODESolver(ABC):
    """Abstract ODE solver"""
    
    @abstractmethod
    def step(
        self,
        velocity_fn: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        conditions: Dict,
        **kwargs
    ) -> torch.Tensor:
        """Single integration step"""
        pass


class EulerSolver(ODESolver):
    """Euler method (1st order)"""
    
    def step(
        self,
        velocity_fn: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        conditions: Dict,
        **kwargs
    ) -> torch.Tensor:
        velocity = velocity_fn(x, t, conditions, **kwargs)
        return x + dt * velocity


class HeunSolver(ODESolver):
    """Heun's method (2nd order)"""
    
    def step(
        self,
        velocity_fn: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        conditions: Dict,
        **kwargs
    ) -> torch.Tensor:
        # First evaluation
        v1 = velocity_fn(x, t, conditions, **kwargs)
        
        # Predictor step
        x_pred = x + dt * v1
        
        # Second evaluation
        v2 = velocity_fn(x_pred, t + dt, conditions, **kwargs)
        
        # Corrector step
        return x + dt * 0.5 * (v1 + v2)


class RK4Solver(ODESolver):
    """4th-order Runge-Kutta (high quality)"""
    
    def step(
        self,
        velocity_fn: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        conditions: Dict,
        **kwargs
    ) -> torch.Tensor:
        k1 = velocity_fn(x, t, conditions, **kwargs)
        k2 = velocity_fn(x + dt * k1 / 2, t + dt / 2, conditions, **kwargs)
        k3 = velocity_fn(x + dt * k2 / 2, t + dt / 2, conditions, **kwargs)
        k4 = velocity_fn(x + dt * k3, t + dt, conditions, **kwargs)
        
        return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


class AdaptiveSolver(ODESolver):
    """Adaptive step size solver"""
    
    def __init__(self, base_solver: ODESolver = None, tolerance: float = 1e-3):
        self.base_solver = base_solver or HeunSolver()
        self.tolerance = tolerance
    
    def step(
        self,
        velocity_fn: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        conditions: Dict,
        **kwargs
    ) -> torch.Tensor:
        # Try full step
        x_full = self.base_solver.step(velocity_fn, x, t, dt, conditions, **kwargs)
        
        # Try two half steps
        x_half = self.base_solver.step(velocity_fn, x, t, dt/2, conditions, **kwargs)
        x_double = self.base_solver.step(velocity_fn, x_half, t + dt/2, dt/2, conditions, **kwargs)
        
        # Estimate error
        error = torch.norm(x_full - x_double) / (torch.norm(x_full) + 1e-8)
        
        if error < self.tolerance:
            return x_full
        else:
            # Use more accurate two-step result
            return x_double


# ==================== Velocity Networks ====================

class VelocityPredictor(nn.Module):
    """Advanced velocity prediction with multiple features"""
    
    def __init__(
        self,
        model: nn.Module,
        use_cfg: bool = True,
        use_self_conditioning: bool = True,
        cfg_scale_range: Tuple[float, float] = (1.0, 3.0),
    ):
        super().__init__()
        self.model = model
        self.use_cfg = use_cfg
        self.use_self_conditioning = use_self_conditioning
        self.cfg_scale_range = cfg_scale_range        # Self-conditioning projection
        if use_self_conditioning:
            # Self-conditioning receives channel-wise mean, so input_dim is the number of channels
            input_channels = getattr(model, 'input_channels', 8)
            model_dim = getattr(model, 'hidden_dims', [128])[0] if hasattr(model, 'hidden_dims') else 512
            self.self_cond_proj = nn.Linear(input_channels, model_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditions: Dict,
        cfg_scale: float = 1.5,
        self_cond: Optional[torch.Tensor] = None,
        return_raw: bool = False
    ) -> torch.Tensor:
        """
        Predict velocity field
        
        Args:
            x: (B, C, T) noisy latent
            t: (B,) timesteps
            conditions: conditioning dict
            cfg_scale: classifier-free guidance scale
            self_cond: self-conditioning from previous step
            return_raw: return raw prediction without CFG
        """
        # Add self-conditioning to conditions if available
        if self_cond is not None and self.use_self_conditioning:
            # Project self-conditioning - ensure device compatibility
            device = x.device
            self_cond_input = self_cond.mean(dim=-1).to(device)
            self_cond_proj = self.self_cond_proj.to(device)(self_cond_input)
            conditions = conditions.copy()
            conditions['self_cond'] = self_cond_proj
        
        # Forward pass
        velocity = self.model(x, t, conditions)
        
        if return_raw or not self.use_cfg or cfg_scale == 1.0:
            return velocity
        
        # Classifier-free guidance
        empty_conditions = self._create_empty_conditions(conditions)
        velocity_uncond = self.model(x, t, empty_conditions)
        
        # CFG interpolation
        velocity_cfg = velocity_uncond + cfg_scale * (velocity - velocity_uncond)
        
        return velocity_cfg
    
    def _create_empty_conditions(self, conditions: Dict) -> Dict:
        """Create empty conditions for CFG"""
        empty = {}
        for key, value in conditions.items():
            if key == 'self_cond':
                continue
            elif value is None:
                empty[key] = None
            elif isinstance(value, torch.Tensor):
                if value.dtype in [torch.long, torch.int]:
                    # For token embeddings, use a special "null" token
                    empty[key] = torch.zeros_like(value)
                else:
                    empty[key] = torch.zeros_like(value)
            else:
                empty[key] = None
        return empty


# ==================== Advanced Flow Matching ====================

class LyroFlowMatching(nn.Module):
    """
    Complete Flow Matching implementation for Lyro
    Supports multiple flow types and advanced features
    """
    
    def __init__(
        self,
        model: nn.Module,
        scheduler_type: str = "cosine",
        solver_type: str = "heun",
        sigma: float = 1e-4,
        use_cfg: bool = True,
        use_self_conditioning: bool = True,
        flow_type: str = "rectified",  # "rectified" or "cfm"
    ):
        super().__init__()
        
        # Store flow_steps for compatibility
        self.flow_steps = 10  # Default value
        
        # Velocity predictor
        self.velocity_predictor = VelocityPredictor(
            model=model,
            use_cfg=use_cfg,
            use_self_conditioning=use_self_conditioning,
        )
        
        # Scheduler
        if scheduler_type == "linear":
            self.scheduler = LinearFlowScheduler()
        elif scheduler_type == "cosine":
            self.scheduler = CosineFlowScheduler()
        elif scheduler_type == "advanced":
            self.scheduler = AdvancedFlowScheduler()
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        # Solver
        if solver_type == "euler":
            self.solver = EulerSolver()
        elif solver_type == "heun":
            self.solver = HeunSolver()
        elif solver_type == "rk4":
            self.solver = RK4Solver()
        elif solver_type == "adaptive":
            self.solver = AdaptiveSolver()
        else:
            raise ValueError(f"Unknown solver: {solver_type}")
        
        self.sigma = sigma
        self.flow_type = flow_type
    
    def compute_flow_path(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute flow path and target velocity"""
        t_expanded = t.view(-1, 1, 1)
        
        if self.flow_type == "rectified":
            # Rectified flow: straight line paths
            xt = (1 - t_expanded) * x0 + t_expanded * x1
            target_velocity = x1 - x0
        elif self.flow_type == "cfm":
            # Conditional flow matching with small noise
            noise = torch.randn_like(x0) * self.sigma
            xt = (1 - t_expanded) * x0 + t_expanded * x1 + noise
            target_velocity = x1 - x0
        else:
            raise ValueError(f"Unknown flow type: {self.flow_type}")
        
        return xt, target_velocity
    
    def training_loss(
        self,
        x1: torch.Tensor,
        conditions: Dict,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute flow matching training loss with numerical stability
        
        Args:
            x1: (B, C, T) clean data
            conditions: conditioning dict
            mask: (B, T) optional mask for sequence lengths
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # ✅ 개선: Numerical stability - 극값 방지
        t = torch.rand(batch_size, device=device)
        t = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)  # 극값 방지
        
        # ✅ 개선: 더 안정적인 노이즈 생성
        x0 = torch.randn_like(x1)
        x0 = x0 * (0.5 + 0.5 * torch.rand_like(x0[:1]))  # 스케일 다양화
        
        # Compute flow path
        xt, target_velocity = self.compute_flow_path(x0, x1, t)
        
        # ✅ 개선: NaN 체크
        if torch.isnan(xt).any() or torch.isnan(target_velocity).any():
            print("Warning: NaN detected in flow path computation")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Predict velocity
        predicted_velocity = self.velocity_predictor(
            xt, t, conditions, return_raw=True
        )
        
        # ✅ 개선: Loss with clipping
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction='none')
        loss = torch.clamp(loss, max=100.0)  # 큰 loss 방지
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)  # (B, 1, T)
            loss = loss * mask_expanded
            loss = loss.sum() / (mask_expanded.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        shape: Tuple[int, int, int],
        conditions: Dict,
        num_steps: int = 10,
        cfg_scale: float = 1.5,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate samples using flow matching
        
        Args:
            shape: (B, C, T) output shape
            conditions: conditioning dict
            num_steps: number of integration steps
            cfg_scale: classifier-free guidance scale
            seed: random seed
            progress_callback: optional progress callback
        
        Returns:
            generated: (B, C, T) generated samples
            trajectory: list of intermediate states
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # ✅ 개선: Shape 검증
        B, C, T = shape
        if C != 8:
            raise ValueError(f"Expected 8 latent channels, got {C}")
        
        device = next(self.velocity_predictor.parameters()).device
        
        # Initial noise
        x = torch.randn(shape, device=device)
        
        # Get timestep schedule
        timesteps = self.scheduler.get_timesteps(num_steps, device)
        
        # Store trajectory
        trajectory = [x.clone()]
        
        # Self-conditioning state
        self_cond = None
        
        # Integration loop
        for i in range(num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr
            
            # Broadcast timestep
            t_batch = t_curr.expand(shape[0])
            dt_batch = dt.expand(shape[0])
            
            # Velocity function for solver
            def velocity_fn(x_in, t_in, cond, **kwargs):
                return self.velocity_predictor(
                    x_in, t_in, cond, 
                    cfg_scale=cfg_scale,
                    self_cond=self_cond,
                    **kwargs
                )
            
            # Integration step
            x_next = self.solver.step(
                velocity_fn, x, t_batch, dt_batch, conditions
            )
            
            # ✅ 개선: NaN/Inf 체크
            if torch.isnan(x_next).any() or torch.isinf(x_next).any():
                print(f"Warning: Invalid values at step {i}, using previous state")
                x_next = x  # 이전 상태 유지
            
            # Update self-conditioning
            if self.velocity_predictor.use_self_conditioning:
                with torch.no_grad():
                    self_cond = self.velocity_predictor(
                        x, t_batch, conditions, 
                        cfg_scale=1.0,
                        return_raw=True
                    ).detach()
            
            x = x_next
            trajectory.append(x.clone())
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, num_steps)
        
        return x, trajectory
    
    @torch.no_grad()
    def flow_edit(
        self,
        original: torch.Tensor,
        mask: torch.Tensor,
        new_conditions: Dict,
        edit_steps: int = 8,
        edit_strength: float = 0.8,
        blend_method: str = "smooth"
    ) -> torch.Tensor:
        """
        Flow-based editing
        
        Args:
            original: (B, C, T) original latent
            mask: (B, 1, T) edit mask
            new_conditions: new conditioning for edited regions
            edit_steps: number of edit steps
            edit_strength: editing strength
            blend_method: blending method
        """
        device = original.device
        batch_size = original.shape[0]
        
        # Create edit schedule
        edit_timesteps = self.scheduler.get_timesteps(edit_steps, device)
        
        # Start from original
        x = original.clone()
        
        # Forward process with new conditions in masked regions
        for i in range(edit_steps):
            t_curr = edit_timesteps[i]
            t_next = edit_timesteps[i + 1] if i < edit_steps - 1 else torch.tensor(1.0, device=device)
            dt = t_next - t_curr
            
            t_batch = t_curr.expand(batch_size)
              # Get velocities
            empty_conditions = self.velocity_predictor._create_empty_conditions(new_conditions)
            v_original = self.velocity_predictor(x, t_batch, empty_conditions, return_raw=True)
            v_edited = self.velocity_predictor(x, t_batch, new_conditions, return_raw=True)
            
            # Blend velocities based on mask and strength
            v_blend = v_original + mask * edit_strength * (v_edited - v_original)
            
            # Integration step
            x = x + dt * v_blend
        
        # Final blending
        if blend_method == "smooth":
            # Gaussian blur on mask boundaries
            kernel_size = 5
            blur_kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
            mask_blurred = F.conv1d(
                mask.float(), 
                blur_kernel, 
                padding=kernel_size//2
            )
            final_mask = mask_blurred
        else:
            final_mask = mask
        
        # Blend original and edited
        result = original * (1 - final_mask) + x * final_mask
        
        return result
    
    @torch.no_grad()
    def flow_extend(
        self,
        original_latent: torch.Tensor,  # ✅ 수정: 파라미터명 통일
        extend_length: int,
        conditions: Dict,
        context_length: int = 100,
        extend_steps: int = 12
    ) -> torch.Tensor:
        """
        Flow-based extension
        
        Args:
            original_latent: (B, C, T) original latent
            extend_length: number of frames to extend
            conditions: conditioning for extension
            context_length: context length for conditioning
            extend_steps: number of extension steps
        """
        B, C, T = original_latent.shape
        device = original_latent.device
        
        # Extract context
        context = original_latent[:, :, -context_length:]
        
        # Initialize extension with noise
        extension = torch.randn(B, C, extend_length, device=device)
        
        # Extend schedule
        extend_timesteps = self.scheduler.get_timesteps(extend_steps, device)
        
        for i in range(extend_steps):
            t_curr = extend_timesteps[i]
            t_batch = t_curr.expand(B)
            
            # Concatenate context and extension for conditioning
            full_sequence = torch.cat([context, extension], dim=-1)
            
            # Predict velocity for extension part only
            velocity = self.velocity_predictor(
                full_sequence, t_batch, conditions, return_raw=True
            )
            extension_velocity = velocity[:, :, -extend_length:]
            
            # Update extension
            dt = extend_timesteps[i+1] - t_curr if i < extend_steps - 1 else 0.1
            extension = extension + dt * extension_velocity
        
        # Concatenate with original
        result = torch.cat([original_latent, extension], dim=-1)
        
        return result


# ==================== Flow Matching Utilities ====================

class FlowEditUtils:
    """Utilities for flow-based editing"""
    
    @staticmethod
    def create_time_mask(
        length: int,
        start_time: float,
        end_time: float,
        device: torch.device,
        fade_length: int = 10
    ) -> torch.Tensor:
        """Create time-based edit mask with smooth fades"""
        mask = torch.zeros(1, 1, length, device=device)
        
        start_idx = int(start_time * length)
        end_idx = int(end_time * length)
        
        # Core region
        mask[:, :, start_idx:end_idx] = 1.0
        
        # Fade in
        if fade_length > 0 and start_idx > fade_length:
            fade_in = torch.linspace(0, 1, fade_length, device=device)
            mask[:, :, start_idx-fade_length:start_idx] = fade_in
        
        # Fade out
        if fade_length > 0 and end_idx + fade_length < length:
            fade_out = torch.linspace(1, 0, fade_length, device=device)
            mask[:, :, end_idx:end_idx+fade_length] = fade_out
        
        return mask
    
    @staticmethod
    def create_frequency_mask(
        shape: Tuple[int, int, int],
        freq_range: Tuple[int, int],
        device: torch.device
    ) -> torch.Tensor:
        """Create frequency-based edit mask"""
        B, C, T = shape
        mask = torch.zeros(B, C, T, device=device)
        
        freq_start, freq_end = freq_range
        freq_start = max(0, freq_start)
        freq_end = min(C, freq_end)
        
        mask[:, freq_start:freq_end, :] = 1.0
        
        return mask


# ==================== Configuration Classes ====================

class FlowConfig:
    """Flow matching configuration"""
    
    def __init__(self):
        # Flow settings
        self.flow_type = "rectified"  # "rectified" or "cfm"
        self.scheduler_type = "cosine"  # "linear", "cosine", "advanced"
        self.solver_type = "heun"  # "euler", "heun", "rk4", "adaptive"
        self.sigma = 1e-4
        
        # Generation settings  
        self.flow_steps = 10  # Much fewer than diffusion
        self.cfg_scale = 1.5
        self.use_self_conditioning = True
        
        # ✅ 수정: integration_method 추가 (호환성)
        self.integration_method = self.solver_type
        
        # Quality presets
        self.quality_presets = {
            "fast": {
                "flow_steps": 4,
                "solver_type": "euler",
                "cfg_scale": 1.2,
            },
            "standard": {
                "flow_steps": 10,
                "solver_type": "heun", 
                "cfg_scale": 1.5,
            },
            "premium": {
                "flow_steps": 20,
                "solver_type": "rk4",
                "cfg_scale": 2.0,
            }
        }
    
    def apply_preset(self, preset: str):
        """Apply quality preset"""
        if preset in self.quality_presets:
            for key, value in self.quality_presets[preset].items():
                setattr(self, key, value)


# ==================== Factory Functions ====================

def create_flow_matching(
    model: nn.Module,
    config: Optional[FlowConfig] = None
) -> LyroFlowMatching:
    """Create flow matching model with config"""
    
    if config is None:
        config = FlowConfig()
    
    return LyroFlowMatching(
        model=model,
        scheduler_type=config.scheduler_type,
        solver_type=config.solver_type,
        sigma=config.sigma,
        flow_type=config.flow_type,
    )


def benchmark_flow_matching(
    flow_matching: LyroFlowMatching,
    shape: Tuple[int, int, int],
    conditions: Dict,
    steps_list: List[int] = [4, 8, 12, 16, 20]
) -> Dict[str, float]:
    """Benchmark flow matching with different step counts"""
    
    import time
    
    results = {}
    
    for steps in steps_list:
        start_time = time.time()
        
        with torch.no_grad():
            generated, _ = flow_matching.generate(
                shape=shape,
                conditions=conditions,
                num_steps=steps,
                cfg_scale=1.5
            )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Calculate RTF (Real-Time Factor)
        # ✅ 수정: 32x 압축율 반영
        audio_duration = shape[2] * 512 * 32 / 44100  
        rtf = generation_time / audio_duration
        
        results[f"{steps}_steps"] = {
            "time": generation_time,
            "rtf": rtf,
            "steps": steps
        }
    
    return results