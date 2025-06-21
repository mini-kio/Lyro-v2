# lyro/ssm/flow_matching.py - Refactored Large Model Only
"""
S6-Enhanced Flow Matching - Large Model Optimized
Simplified Architecture + FP16 Ready + DCAE Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Callable
import math
from abc import ABC, abstractmethod
from einops import rearrange, repeat


# ==================== Core Flow Schedulers ====================

class S6FlowScheduler(ABC):
    """Abstract S6 flow scheduler"""
    
    @abstractmethod
    def get_timesteps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        pass
    
    @abstractmethod
    def scale_noise(self, t: torch.Tensor) -> torch.Tensor:
        pass


class LinearFlowScheduler(S6FlowScheduler):
    """Simple linear scheduler for large model"""
    
    def get_timesteps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        return torch.linspace(0, 1, num_steps + 1, device=device)
    
    def scale_noise(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t) * 0.9


class CosineFlowScheduler(S6FlowScheduler):
    """Cosine scheduler for large model"""
    
    def __init__(self, warmup_ratio: float = 0.1):
        self.warmup_ratio = warmup_ratio
    
    def get_timesteps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        steps = torch.linspace(0, 1, num_steps + 1, device=device)
        warmup_steps = int(num_steps * self.warmup_ratio)
        
        if warmup_steps > 0:
            warmup = torch.linspace(0, 0.1, warmup_steps, device=device)
            main_steps = torch.linspace(0.1, 1, num_steps + 1 - warmup_steps, device=device)
            cosine_main = 0.1 + 0.9 * 0.5 * (1 - torch.cos(math.pi * main_steps / main_steps[-1]))
            steps = torch.cat([warmup, cosine_main])
        else:
            steps = 0.5 * (1 - torch.cos(math.pi * steps))
        
        return steps
    
    def scale_noise(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1 - torch.cos(math.pi * t)) * 0.8 + 0.2


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
        pass


class EulerSolver(ODESolver):
    """Simple Euler method"""
    
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
        dt_expanded = dt.view(-1, 1, 1)
        return x + dt_expanded * velocity


class HeunSolver(ODESolver):
    """Heun's method for better accuracy"""
    
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
        dt_expanded = dt.view(-1, 1, 1)
        
        # Predictor step
        x_pred = x + dt_expanded * v1
        
        # Second evaluation
        v2 = velocity_fn(x_pred, t + dt, conditions, **kwargs)
        
        # Corrector step
        return x + dt_expanded * 0.5 * (v1 + v2)


# ==================== Simplified Velocity Predictor ====================

class S6VelocityPredictor(nn.Module):
    """Simplified S6 velocity prediction for large model only"""
    
    def __init__(
        self,
        model: nn.Module,
        use_cfg: bool = True,
        cfg_scale_range: Tuple[float, float] = (1.0, 3.0),
        latent_channels: int = 16,  # Large model setting
    ):
        super().__init__()
        self.model = model
        self.use_cfg = use_cfg
        self.cfg_scale_range = cfg_scale_range
        self.latent_channels = latent_channels
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditions: Dict,
        cfg_scale: float = 1.5,
        **kwargs
    ) -> torch.Tensor:
        """
        Simplified velocity prediction for large model
        
        Args:
            x: (B, 16, H, W) latent for large model
            t: (B,) timesteps
            conditions: conditioning dict
            cfg_scale: guidance scale
        """
        
        # Validate input dimensions for large model
        if x.shape[1] != self.latent_channels:
            print(f"Warning: Expected {self.latent_channels} channels, got {x.shape[1]}")
        
        # Simple forward without CFG if scale is 1.0
        if not self.use_cfg or cfg_scale <= 1.0:
            return self._safe_model_forward(x, t, conditions)
        
        # CFG computation
        return self._cfg_forward(x, t, conditions, cfg_scale)
    
    def _safe_model_forward(self, x: torch.Tensor, t: torch.Tensor, conditions: Dict) -> torch.Tensor:
        """Safe model forward with error handling"""
        try:
            return self.model(x, t, conditions)
        except Exception as e:
            print(f"Model forward failed: {e}")
            return torch.zeros_like(x)
    
    def _cfg_forward(self, x: torch.Tensor, t: torch.Tensor, conditions: Dict, cfg_scale: float) -> torch.Tensor:
        """Simplified CFG computation"""
        try:
            # Create empty conditions
            empty_conditions = self._create_empty_conditions(conditions)
            
            # Batch processing
            batch_size = x.shape[0]
            x_doubled = torch.cat([x, x], dim=0)
            t_doubled = torch.cat([t, t], dim=0)
            
            # Prepare doubled conditions
            doubled_conditions = {}
            for key, value in conditions.items():
                if value is not None and isinstance(value, torch.Tensor):
                    empty_value = empty_conditions.get(key, torch.zeros_like(value))
                    doubled_conditions[key] = torch.cat([value, empty_value], dim=0)
                else:
                    doubled_conditions[key] = value
            
            # Single forward pass
            doubled_output = self._safe_model_forward(x_doubled, t_doubled, doubled_conditions)
            
            # Split and apply CFG
            velocity_cond, velocity_uncond = doubled_output.chunk(2, dim=0)
            cfg_scale = torch.clamp(torch.tensor(cfg_scale), 0.1, 3.0).item()
            
            return velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
            
        except Exception as e:
            print(f"CFG forward failed: {e}")
            return self._safe_model_forward(x, t, conditions)
    
    def _create_empty_conditions(self, conditions: Dict) -> Dict:
        """Create empty conditions for CFG"""
        empty = {}
        for key, value in conditions.items():
            if value is None:
                empty[key] = None
            elif isinstance(value, torch.Tensor):
                empty[key] = torch.zeros_like(value)
            else:
                empty[key] = None
        return empty


# ==================== Main Flow Matching Class ====================

class LyroS6FlowMatching(nn.Module):
    """
    Simplified S6 Flow Matching for Large Model Only
    Compatible with Large DCAE (16 latent channels)
    """
    
    def __init__(
        self,
        model: nn.Module,
        scheduler_type: str = "cosine",
        solver_type: str = "heun",
        sigma: float = 1e-4,
        use_cfg: bool = True,
        flow_type: str = "rectified",
        latent_channels: int = 16,  # Large model setting
    ):
        super().__init__()
        
        self.flow_steps = 10
        self.latent_channels = latent_channels
        
        # Velocity predictor
        self.velocity_predictor = S6VelocityPredictor(
            model=model,
            use_cfg=use_cfg,
            latent_channels=latent_channels,
        )
        
        # Scheduler
        if scheduler_type == "linear":
            self.scheduler = LinearFlowScheduler()
        elif scheduler_type == "cosine":
            self.scheduler = CosineFlowScheduler()
        else:
            self.scheduler = CosineFlowScheduler()
        
        # Solver
        if solver_type == "euler":
            self.solver = EulerSolver()
        elif solver_type == "heun":
            self.solver = HeunSolver()
        else:
            self.solver = HeunSolver()
        
        self.sigma = sigma
        self.flow_type = flow_type
    
    def compute_flow_path(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute flow path for large model"""
        t_expanded = t.view(-1, 1, 1, 1)  # For 4D tensors (B, C, H, W)
        
        if self.flow_type == "rectified":
            xt = (1 - t_expanded) * x0 + t_expanded * x1
            target_velocity = x1 - x0
        else:
            noise = torch.randn_like(x0) * self.sigma
            xt = (1 - t_expanded) * x0 + t_expanded * x1 + noise
            target_velocity = x1 - x0
        
        return xt, target_velocity
    
    def training_loss(
        self,
        x1: torch.Tensor,
        conditions: Dict,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Flow matching training loss for large model
        
        Args:
            x1: (B, 16, H, W) clean latents from Large DCAE
            conditions: conditioning dict
            mask: optional mask
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # Validate large model input
        if x1.shape[1] != self.latent_channels:
            print(f"Warning: Expected {self.latent_channels} channels, got {x1.shape[1]}")
        
        # Force FP16
        if x1.dtype != torch.float16:
            x1 = x1.half()
        
        # Timestep sampling
        t = torch.rand(batch_size, device=device)
        t = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
        
        # Noise generation
        x0 = torch.randn_like(x1)
        
        # Flow path
        xt, target_velocity = self.compute_flow_path(x0, x1, t)
        
        # Velocity prediction
        try:
            predicted_velocity = self.velocity_predictor(xt, t, conditions)
        except Exception as e:
            print(f"Velocity prediction failed: {e}")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Loss computation
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.view(batch_size, 1, 1, 1)
            loss = loss * mask_expanded
            loss = loss.sum() / (mask_expanded.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        shape: Tuple[int, int, int, int],
        conditions: Dict,
        num_steps: int = 10,
        cfg_scale: float = 1.5,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate samples for large model
        
        Args:
            shape: (B, 16, H, W) output shape for large model
            conditions: conditioning dict
            num_steps: integration steps
            cfg_scale: guidance scale
            seed: random seed
            progress_callback: progress callback
        
        Returns:
            generated: (B, 16, H, W) generated latents
            trajectory: intermediate states
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        B, C, H, W = shape
        
        # Validate large model shape
        if C != self.latent_channels:
            print(f"Warning: Expected {self.latent_channels} latent channels, got {C}")
        
        device = next(self.velocity_predictor.parameters()).device
        
        # Initial noise (FP16)
        x = torch.randn(shape, device=device, dtype=torch.float16)
        
        # Timestep schedule
        timesteps = self.scheduler.get_timesteps(num_steps, device)
        
        # Store trajectory
        trajectory = [x.clone()]
        
        # Integration loop
        for i in range(num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr
            
            # Broadcast timestep
            t_batch = t_curr.expand(B)
            dt_batch = dt.expand(B)
            
            # Velocity function
            def velocity_fn(x_in, t_in, cond, **kwargs):
                return self.velocity_predictor(x_in, t_in, cond, cfg_scale=cfg_scale, **kwargs)
            
            # Integration step
            try:
                x_next = self.solver.step(velocity_fn, x, t_batch, dt_batch, conditions)
                
                # Stability check
                if torch.isnan(x_next).any() or torch.isinf(x_next).any():
                    print(f"Warning: Invalid values at step {i}")
                    x_next = x
                
                x = x_next.half()  # Ensure FP16
                trajectory.append(x.clone())
                
            except Exception as e:
                print(f"Integration step {i} failed: {e}")
                trajectory.append(x.clone())
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, num_steps)
        
        return x, trajectory


# ==================== Configuration ====================

class FlowConfig:
    """Simplified flow configuration for large model"""
    
    def __init__(self):
        # Flow settings
        self.flow_type = "rectified"
        self.scheduler_type = "cosine"
        self.solver_type = "heun"
        self.sigma = 1e-4
        
        # Large model settings
        self.latent_channels = 16
        
        # Generation settings
        self.flow_steps = 10
        self.cfg_scale = 1.5
        
        # Quality presets for large model
        self.quality_presets = {
            "fast": {
                "flow_steps": 6,
                "solver_type": "euler",
                "cfg_scale": 1.2,
            },
            "standard": {
                "flow_steps": 10,
                "solver_type": "heun", 
                "cfg_scale": 1.5,
            },
            "high": {
                "flow_steps": 16,
                "solver_type": "heun",
                "cfg_scale": 2.0,
            }
        }
    
    def apply_preset(self, preset: str):
        """Apply quality preset"""
        if preset in self.quality_presets:
            for key, value in self.quality_presets[preset].items():
                setattr(self, key, value)


# ==================== Factory Function ====================

def create_s6_flow_matching(
    model: nn.Module,
    config: Optional[FlowConfig] = None,
) -> LyroS6FlowMatching:
    """
    Create S6 flow matching for large model
    
    Args:
        model: S6-based neural network model (large)
        config: flow matching configuration
        
    Returns:
        LyroS6FlowMatching model for large DCAE
    """
    
    if config is None:
        config = FlowConfig()
    
    # Create flow matching model
    flow_matching = LyroS6FlowMatching(
        model=model,
        scheduler_type=config.scheduler_type,
        solver_type=config.solver_type,
        sigma=config.sigma,
        flow_type=config.flow_type,
        use_cfg=True,
        latent_channels=config.latent_channels,
    )
    
    print(f"✅ S6 FlowMatching Created for Large Model:")
    print(f"   - Latent Channels: {config.latent_channels}")
    print(f"   - Scheduler: {config.scheduler_type}")
    print(f"   - Solver: {config.solver_type}")
    print(f"   - CFG Enabled: True")
    
    return flow_matching


# ==================== Backward Compatibility ====================

# Legacy aliases
LyroFlowMatching = LyroS6FlowMatching
create_flow_matching = create_s6_flow_matching

print("✅ S6 Flow Matching - Large Model Ready!")
print("Key features:")
print("- Large model optimized (16 latent channels)")
print("- FP16 enforced throughout pipeline")
print("- Simplified architecture for stability")
print("- DCAE integration ready")