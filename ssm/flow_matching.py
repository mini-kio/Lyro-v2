# lyro/ssm/flow_matching.py
"""
S6-Optimized Flow Matching for Lyro
Ultra-fast music generation with enhanced S6 State Space processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Callable
import math
from abc import ABC, abstractmethod
import time
from einops import rearrange, repeat
import triton
import triton.language as tl


# ==================== S6-Optimized Flow Schedulers ====================

class S6FlowScheduler(ABC):
    """Abstract S6-optimized flow scheduler"""
    
    @abstractmethod
    def get_timesteps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Get timestep schedule optimized for S6"""
        pass
    
    @abstractmethod
    def scale_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Scale noise for given timestep with S6 considerations"""
        pass
    
    @abstractmethod
    def get_s6_chunk_schedule(self, seq_len: int, chunk_size: int) -> List[int]:
        """Get S6 chunk processing schedule"""
        pass


class S6LinearFlowScheduler(S6FlowScheduler):
    """S6-optimized linear timestep scheduler"""
    
    def __init__(self, chunk_size: int = 256):
        self.chunk_size = chunk_size
    
    def get_timesteps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        return torch.linspace(0, 1, num_steps + 1, device=device)
    
    def scale_noise(self, t: torch.Tensor) -> torch.Tensor:
        # S6 works better with slightly scaled noise
        return torch.ones_like(t) * 0.95
    
    def get_s6_chunk_schedule(self, seq_len: int, chunk_size: int) -> List[int]:
        """Optimize chunk schedule for S6 processing"""
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        return [min(chunk_size, seq_len - i * chunk_size) for i in range(num_chunks)]


class S6CosineFlowScheduler(S6FlowScheduler):
    """S6-optimized cosine scheduler for smoother generation"""
    
    def __init__(self, chunk_size: int = 256, warmup_ratio: float = 0.1):
        self.chunk_size = chunk_size
        self.warmup_ratio = warmup_ratio
    
    def get_timesteps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        steps = torch.linspace(0, 1, num_steps + 1, device=device)
        # S6-optimized cosine schedule with warmup
        warmup_steps = int(num_steps * self.warmup_ratio)
        
        if warmup_steps > 0:
            # Linear warmup
            warmup = torch.linspace(0, 0.1, warmup_steps, device=device)
            # Cosine main schedule
            main_steps = torch.linspace(0.1, 1, num_steps + 1 - warmup_steps, device=device)
            cosine_main = 0.1 + 0.9 * 0.5 * (1 - torch.cos(math.pi * main_steps / main_steps[-1]))
            steps = torch.cat([warmup, cosine_main])
        else:
            steps = 0.5 * (1 - torch.cos(math.pi * steps))
            
        return steps
    
    def scale_noise(self, t: torch.Tensor) -> torch.Tensor:
        # S6-optimized noise scaling
        return 0.5 * (1 - torch.cos(math.pi * t)) * 0.9 + 0.1
    
    def get_s6_chunk_schedule(self, seq_len: int, chunk_size: int) -> List[int]:
        """Adaptive chunk schedule for S6 cosine processing"""
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        chunks = []
        
        for i in range(num_chunks):
            # Adaptive chunk size based on position
            if i == 0 or i == num_chunks - 1:
                # Smaller chunks at boundaries for better S6 processing
                adaptive_size = min(chunk_size // 2, seq_len - i * chunk_size)
            else:
                adaptive_size = min(chunk_size, seq_len - i * chunk_size)
            chunks.append(max(adaptive_size, 1))
        
        return chunks


class S6AdvancedFlowScheduler(S6FlowScheduler):
    """Advanced S6 scheduler with sway sampling and chunk optimization"""
    
    def __init__(self, chunk_size: int = 256, sway_coeff: float = 0.2, adaptive_chunks: bool = True):
        self.chunk_size = chunk_size
        self.sway_coeff = sway_coeff
        self.adaptive_chunks = adaptive_chunks
    
    def get_timesteps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        base_steps = torch.linspace(0, 1, num_steps + 1, device=device)
        # S6-optimized sway sampling (reduced coefficient for stability)
        sway_steps = base_steps + self.sway_coeff * (
            torch.cos(math.pi / 2 * base_steps) - 1 + base_steps
        )
        return torch.clamp(sway_steps, 0, 1)
    
    def scale_noise(self, t: torch.Tensor) -> torch.Tensor:
        # Advanced S6 noise scaling with temporal dependency
        return 0.8 + 0.2 * torch.cos(2 * math.pi * t)
    
    def get_s6_chunk_schedule(self, seq_len: int, chunk_size: int) -> List[int]:
        """Advanced adaptive chunk schedule for S6"""
        if not self.adaptive_chunks:
            return S6LinearFlowScheduler.get_s6_chunk_schedule(self, seq_len, chunk_size)
        
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        chunks = []
        
        for i in range(num_chunks):
            # Progressive chunk sizing for S6 efficiency
            progress = i / max(num_chunks - 1, 1)
            
            # Larger chunks in the middle, smaller at edges
            size_multiplier = 0.5 + 0.5 * (1 - abs(0.5 - progress) * 2)
            adaptive_size = int(chunk_size * size_multiplier)
            
            actual_size = min(adaptive_size, seq_len - i * chunk_size)
            chunks.append(max(actual_size, 16))  # Minimum chunk size for S6
        
        return chunks


# ==================== S6-Optimized ODE Solvers ====================

class S6ODESolver(ABC):
    """Abstract S6-optimized ODE solver"""
    
    @abstractmethod
    def step(
        self,
        velocity_fn: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        conditions: Dict,
        chunk_schedule: Optional[List[int]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Single integration step optimized for S6"""
        pass


class S6EulerSolver(S6ODESolver):
    """S6-optimized Euler method with chunk processing"""
    
    def __init__(self, use_chunked_processing: bool = True):
        self.use_chunked_processing = use_chunked_processing
    
    def step(
        self,
        velocity_fn: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        conditions: Dict,
        chunk_schedule: Optional[List[int]] = None,
        **kwargs
    ) -> torch.Tensor:
        
        if self.use_chunked_processing and chunk_schedule and x.shape[-1] > 512:
            return self._chunked_step(velocity_fn, x, t, dt, conditions, chunk_schedule, **kwargs)
        else:
            return self._standard_step(velocity_fn, x, t, dt, conditions, **kwargs)
    
    def _standard_step(self, velocity_fn, x, t, dt, conditions, **kwargs):
        """Standard Euler step"""
        velocity = velocity_fn(x, t, conditions, **kwargs)
        dt_expanded = dt.view(-1, 1, 1)
        return x + dt_expanded * velocity
    
    def _chunked_step(self, velocity_fn, x, t, dt, conditions, chunk_schedule, **kwargs):
        """S6-optimized chunked Euler step"""
        B, C, T = x.shape
        chunks = []
        start_idx = 0
        
        for chunk_size in chunk_schedule:
            end_idx = min(start_idx + chunk_size, T)
            if start_idx >= T:
                break
                
            # Process chunk
            x_chunk = x[:, :, start_idx:end_idx]
            velocity_chunk = velocity_fn(x_chunk, t, conditions, **kwargs)
            
            dt_expanded = dt.view(-1, 1, 1)
            x_next_chunk = x_chunk + dt_expanded * velocity_chunk
            
            chunks.append(x_next_chunk)
            start_idx = end_idx
        
        return torch.cat(chunks, dim=-1)


class S6HeunSolver(S6ODESolver):
    """S6-optimized Heun's method"""
    
    def __init__(self, use_chunked_processing: bool = True):
        self.use_chunked_processing = use_chunked_processing
    
    def step(
        self,
        velocity_fn: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        conditions: Dict,
        chunk_schedule: Optional[List[int]] = None,
        **kwargs
    ) -> torch.Tensor:
        
        if self.use_chunked_processing and chunk_schedule and x.shape[-1] > 512:
            return self._chunked_heun_step(velocity_fn, x, t, dt, conditions, chunk_schedule, **kwargs)
        else:
            return self._standard_heun_step(velocity_fn, x, t, dt, conditions, **kwargs)
    
    def _standard_heun_step(self, velocity_fn, x, t, dt, conditions, **kwargs):
        """Standard Heun step"""
        # First evaluation
        v1 = velocity_fn(x, t, conditions, **kwargs)
        dt_expanded = dt.view(-1, 1, 1)
        
        # Predictor step
        x_pred = x + dt_expanded * v1
        
        # Second evaluation
        v2 = velocity_fn(x_pred, t + dt, conditions, **kwargs)
        
        # Corrector step
        return x + dt_expanded * 0.5 * (v1 + v2)
    
    def _chunked_heun_step(self, velocity_fn, x, t, dt, conditions, chunk_schedule, **kwargs):
        """S6-optimized chunked Heun step"""
        B, C, T = x.shape
        chunks = []
        start_idx = 0
        
        for chunk_size in chunk_schedule:
            end_idx = min(start_idx + chunk_size, T)
            if start_idx >= T:
                break
                
            # Process chunk with Heun's method
            x_chunk = x[:, :, start_idx:end_idx]
            
            # First evaluation
            v1_chunk = velocity_fn(x_chunk, t, conditions, **kwargs)
            dt_expanded = dt.view(-1, 1, 1)
            
            # Predictor step
            x_pred_chunk = x_chunk + dt_expanded * v1_chunk
            
            # Second evaluation
            v2_chunk = velocity_fn(x_pred_chunk, t + dt, conditions, **kwargs)
            
            # Corrector step
            x_next_chunk = x_chunk + dt_expanded * 0.5 * (v1_chunk + v2_chunk)
            
            chunks.append(x_next_chunk)
            start_idx = end_idx
        
        return torch.cat(chunks, dim=-1)


class S6AdaptiveSolver(S6ODESolver):
    """S6-optimized adaptive step size solver"""
    
    def __init__(self, base_solver: S6ODESolver = None, tolerance: float = 1e-3, max_chunk_size: int = 512):
        self.base_solver = base_solver or S6HeunSolver()
        self.tolerance = tolerance
        self.max_chunk_size = max_chunk_size
    
    def step(
        self,
        velocity_fn: Callable,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        conditions: Dict,
        chunk_schedule: Optional[List[int]] = None,
        **kwargs
    ) -> torch.Tensor:
        
        # Adaptive chunk schedule based on sequence length
        if chunk_schedule is None:
            seq_len = x.shape[-1]
            chunk_schedule = self._generate_adaptive_schedule(seq_len)
        
        # Try full step
        x_full = self.base_solver.step(velocity_fn, x, t, dt, conditions, chunk_schedule, **kwargs)
        
        # Try two half steps for error estimation
        x_half = self.base_solver.step(velocity_fn, x, t, dt/2, conditions, chunk_schedule, **kwargs)
        x_double = self.base_solver.step(velocity_fn, x_half, t + dt/2, dt/2, conditions, chunk_schedule, **kwargs)
        
        # Estimate error
        error = torch.norm(x_full - x_double) / (torch.norm(x_full) + 1e-8)
        
        if error < self.tolerance:
            return x_full
        else:
            return x_double  # More accurate result
    
    def _generate_adaptive_schedule(self, seq_len: int) -> List[int]:
        """Generate adaptive chunk schedule for S6"""
        if seq_len <= self.max_chunk_size:
            return [seq_len]
        
        num_chunks = (seq_len + self.max_chunk_size - 1) // self.max_chunk_size
        base_size = seq_len // num_chunks
        remainder = seq_len % num_chunks
        
        schedule = [base_size] * num_chunks
        # Distribute remainder
        for i in range(remainder):
            schedule[i] += 1
        
        return schedule


# ==================== S6-Enhanced Velocity Networks ====================

class S6VelocityPredictor(nn.Module):
    """S6-enhanced velocity prediction with chunked processing"""
    
    def __init__(
        self,
        model: nn.Module,
        use_cfg: bool = True,
        use_self_conditioning: bool = True,
        cfg_scale_range: Tuple[float, float] = (1.0, 3.0),
        chunk_size: int = 256,
        enable_s6_optimizations: bool = True,
    ):
        super().__init__()
        self.model = model
        self.use_cfg = use_cfg
        self.use_self_conditioning = use_self_conditioning
        self.cfg_scale_range = cfg_scale_range
        self.chunk_size = chunk_size
        self.enable_s6_optimizations = enable_s6_optimizations
        
        # S6-optimized self-conditioning
        if use_self_conditioning:
            input_channels = getattr(model, 'input_channels', 8)
            model_dim = getattr(model, 'hidden_dims', [128])[0] if hasattr(model, 'hidden_dims') else 512
            
            self.self_cond_proj = nn.Sequential(
                nn.Linear(input_channels, model_dim // 2),
                nn.LayerNorm(model_dim // 2),
                nn.GELU(),
                nn.Linear(model_dim // 2, model_dim),
                nn.Dropout(0.1)
            )
        
        # S6 chunk processing cache
        self._chunk_cache = {}
        self._cache_size_limit = 100
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditions: Dict,
        cfg_scale: float = 1.5,
        self_cond: Optional[torch.Tensor] = None,
        return_raw: bool = False,
        use_chunked_processing: bool = None,
    ) -> torch.Tensor:
        """
        S6-optimized velocity prediction
        
        Args:
            x: (B, C, T) noisy latent
            t: (B,) timesteps
            conditions: conditioning dict
            cfg_scale: classifier-free guidance scale
            self_cond: self-conditioning from previous step
            return_raw: return raw prediction without CFG
            use_chunked_processing: override chunked processing
        """
        
        # Determine if chunked processing should be used
        if use_chunked_processing is None:
            use_chunked_processing = (
                self.enable_s6_optimizations and 
                x.shape[-1] > self.chunk_size * 2
            )
        
        # Enhanced self-conditioning for S6
        if self_cond is not None and self.use_self_conditioning:
            device = x.device
            # More sophisticated self-conditioning for S6
            self_cond_input = self_cond.mean(dim=-1).to(device)  # (B, C)
            self_cond_proj = self.self_cond_proj.to(device)(self_cond_input)
            
            conditions = conditions.copy()
            conditions['self_cond'] = self_cond_proj
        
        # S6-optimized forward pass
        if use_chunked_processing:
            velocity = self._chunked_forward(x, t, conditions)
        else:
            velocity = self.model(x, t, conditions)
        
        if return_raw or not self.use_cfg or cfg_scale == 1.0:
            return velocity
        
        # S6-optimized classifier-free guidance
        empty_conditions = self._create_empty_conditions(conditions)
        
        if use_chunked_processing:
            velocity_uncond = self._chunked_forward(x, t, empty_conditions)
        else:
            velocity_uncond = self.model(x, t, empty_conditions)
        
        # Enhanced CFG interpolation for S6
        velocity_cfg = velocity_uncond + cfg_scale * (velocity - velocity_uncond)
        
        return velocity_cfg
    
    def _chunked_forward(self, x: torch.Tensor, t: torch.Tensor, conditions: Dict) -> torch.Tensor:
        """S6-optimized chunked forward pass"""
        B, C, T = x.shape
        
        if T <= self.chunk_size:
            return self.model(x, t, conditions)
        
        # Generate chunk schedule
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        chunks = []
        
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, T)
            
            # Extract chunk with overlap for continuity
            overlap = min(16, self.chunk_size // 8)  # Small overlap
            chunk_start = max(0, start_idx - overlap)
            chunk_end = min(T, end_idx + overlap)
            
            x_chunk = x[:, :, chunk_start:chunk_end]
            
            # Process chunk
            with torch.cuda.amp.autocast(enabled=True):
                velocity_chunk = self.model(x_chunk, t, conditions)
            
            # Extract the relevant part (removing overlap)
            if chunk_start < start_idx:
                velocity_chunk = velocity_chunk[:, :, overlap:]
            if chunk_end > end_idx:
                velocity_chunk = velocity_chunk[:, :, :-(chunk_end - end_idx)]
            
            chunks.append(velocity_chunk)
        
        return torch.cat(chunks, dim=-1)
    
    def _create_empty_conditions(self, conditions: Dict) -> Dict:
        """Create S6-optimized empty conditions for CFG"""
        empty = {}
        for key, value in conditions.items():
            if key == 'self_cond':
                continue
            elif value is None:
                empty[key] = None
            elif isinstance(value, torch.Tensor):
                if value.dtype in [torch.long, torch.int]:
                    # Use special null tokens for S6
                    empty[key] = torch.zeros_like(value)
                else:
                    # Use small noise for continuous values
                    empty[key] = torch.randn_like(value) * 0.01
            else:
                empty[key] = None
        return empty


# ==================== S6-Enhanced Flow Matching ====================

class LyroS6FlowMatching(nn.Module):
    """
    S6-Enhanced Flow Matching for Lyro
    Optimized for S6 State Space Models with chunk processing
    """
    
    def __init__(
        self,
        model: nn.Module,
        scheduler_type: str = "s6_cosine",
        solver_type: str = "s6_heun",
        sigma: float = 1e-4,
        use_cfg: bool = True,
        use_self_conditioning: bool = True,
        flow_type: str = "rectified",
        # S6 specific parameters
        chunk_size: int = 256,
        enable_s6_optimizations: bool = True,
        use_chunked_solver: bool = True,
    ):
        super().__init__()
        
        self.flow_steps = 10  # Default for compatibility
        self.chunk_size = chunk_size
        self.enable_s6_optimizations = enable_s6_optimizations
        
        # S6-enhanced velocity predictor
        self.velocity_predictor = S6VelocityPredictor(
            model=model,
            use_cfg=use_cfg,
            use_self_conditioning=use_self_conditioning,
            chunk_size=chunk_size,
            enable_s6_optimizations=enable_s6_optimizations,
        )
        
        # S6-optimized scheduler
        if scheduler_type == "s6_linear":
            self.scheduler = S6LinearFlowScheduler(chunk_size)
        elif scheduler_type == "s6_cosine":
            self.scheduler = S6CosineFlowScheduler(chunk_size)
        elif scheduler_type == "s6_advanced":
            self.scheduler = S6AdvancedFlowScheduler(chunk_size)
        else:
            # Fallback to regular schedulers
            if scheduler_type == "linear":
                self.scheduler = S6LinearFlowScheduler(chunk_size)
            else:
                self.scheduler = S6CosineFlowScheduler(chunk_size)
        
        # S6-optimized solver
        if solver_type == "s6_euler":
            self.solver = S6EulerSolver(use_chunked_solver)
        elif solver_type == "s6_heun":
            self.solver = S6HeunSolver(use_chunked_solver)
        elif solver_type == "s6_adaptive":
            self.solver = S6AdaptiveSolver(max_chunk_size=chunk_size)
        else:
            # Fallback
            self.solver = S6HeunSolver(use_chunked_solver)
        
        self.sigma = sigma
        self.flow_type = flow_type
    
    def compute_flow_path(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """S6-optimized flow path computation"""
        t_expanded = t.view(-1, 1, 1)
        
        if self.flow_type == "rectified":
            # Rectified flow optimized for S6
            xt = (1 - t_expanded) * x0 + t_expanded * x1
            target_velocity = x1 - x0
        elif self.flow_type == "cfm":
            # S6-optimized conditional flow matching
            noise = torch.randn_like(x0) * self.sigma * 0.8  # Reduced noise for S6 stability
            xt = (1 - t_expanded) * x0 + t_expanded * x1 + noise
            target_velocity = x1 - x0
        else:
            raise ValueError(f"Unknown flow type: {self.flow_type}")
        
        return xt, target_velocity
    
    def training_loss(
        self,
        x1: torch.Tensor,
        conditions: Dict,
        mask: Optional[torch.Tensor] = None,
        use_s6_chunking: bool = None,
    ) -> torch.Tensor:
        """
        S6-optimized flow matching training loss
        
        Args:
            x1: (B, C, T) clean data
            conditions: conditioning dict
            mask: (B, T) optional mask for sequence lengths
            use_s6_chunking: enable S6 chunked processing
        """
        batch_size = x1.shape[0]
        device = x1.device
        seq_len = x1.shape[-1]
        
        # S6-optimized timestep sampling
        t = torch.rand(batch_size, device=device)
        t = torch.clamp(t, min=1e-4, max=1.0 - 1e-4)
        
        # S6-enhanced noise generation
        x0 = torch.randn_like(x1)
        # Add slight structure to noise for S6 efficiency
        if seq_len > self.chunk_size:
            # Structured noise for long sequences
            chunk_noise = torch.randn(x1.shape[0], x1.shape[1], self.chunk_size, device=device)
            x0 = chunk_noise.repeat(1, 1, (seq_len + self.chunk_size - 1) // self.chunk_size)[:, :, :seq_len]
        
        x0 = x0 * (0.8 + 0.4 * torch.rand_like(x0[:1]))  # Variable scaling
        
        # Compute S6-optimized flow path
        xt, target_velocity = self.compute_flow_path(x0, x1, t)
        
        # Numerical stability checks
        if torch.isnan(xt).any() or torch.isnan(target_velocity).any():
            print("Warning: NaN detected in S6 flow path computation")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # S6-optimized velocity prediction
        if use_s6_chunking is None:
            use_s6_chunking = self.enable_s6_optimizations and seq_len > self.chunk_size
        
        predicted_velocity = self.velocity_predictor(
            xt, t, conditions, 
            return_raw=True,
            use_chunked_processing=use_s6_chunking
        )
        
        # S6-enhanced loss computation
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction='none')
        
        # S6-specific loss weighting (emphasize chunk boundaries less)
        if use_s6_chunking and seq_len > self.chunk_size:
            weight = torch.ones_like(loss)
            # Slightly reduce weight at chunk boundaries
            for i in range(self.chunk_size, seq_len, self.chunk_size):
                boundary_start = max(0, i - 4)
                boundary_end = min(seq_len, i + 4)
                weight[:, :, boundary_start:boundary_end] *= 0.9
            loss = loss * weight
        
        # Apply mask and clipping
        loss = torch.clamp(loss, max=100.0)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
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
        progress_callback: Optional[Callable] = None,
        use_s6_optimizations: bool = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        S6-optimized sample generation
        
        Args:
            shape: (B, C, T) output shape
            conditions: conditioning dict
            num_steps: number of integration steps
            cfg_scale: classifier-free guidance scale
            seed: random seed
            progress_callback: optional progress callback
            use_s6_optimizations: enable S6 optimizations
        
        Returns:
            generated: (B, C, T) generated samples
            trajectory: list of intermediate states
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        B, C, T = shape
        if C != 8:
            raise ValueError(f"Expected 8 latent channels for S6, got {C}")
        
        device = next(self.velocity_predictor.parameters()).device
        
        # Determine S6 optimization usage
        if use_s6_optimizations is None:
            use_s6_optimizations = self.enable_s6_optimizations and T > self.chunk_size
        
        # S6-enhanced initial noise
        if use_s6_optimizations and T > self.chunk_size:
            # Structured initial noise for better S6 processing
            chunk_noise = torch.randn(B, C, self.chunk_size, device=device)
            x = chunk_noise.repeat(1, 1, (T + self.chunk_size - 1) // self.chunk_size)[:, :, :T]
            # Add some randomness
            x = x + torch.randn_like(x) * 0.1
        else:
            x = torch.randn(shape, device=device)
        
        # Get S6-optimized timestep schedule
        timesteps = self.scheduler.get_timesteps(num_steps, device)
        
        # Get S6 chunk schedule if needed
        chunk_schedule = None
        if use_s6_optimizations:
            chunk_schedule = self.scheduler.get_s6_chunk_schedule(T, self.chunk_size)
        
        # Store trajectory
        trajectory = [x.clone()]
        
        # S6-enhanced self-conditioning
        self_cond = None
        
        # S6-optimized integration loop
        for i in range(num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr
            
            # Broadcast timestep
            t_batch = t_curr.expand(B)
            dt_batch = dt.expand(B)
            
            # S6-optimized velocity function
            def velocity_fn(x_in, t_in, cond, **kwargs):
                return self.velocity_predictor(
                    x_in, t_in, cond, 
                    cfg_scale=cfg_scale,
                    self_cond=self_cond,
                    use_chunked_processing=use_s6_optimizations,
                    **kwargs
                )
            
            # S6-enhanced integration step
            x_next = self.solver.step(
                velocity_fn, x, t_batch, dt_batch, conditions,
                chunk_schedule=chunk_schedule
            )
            
            # Enhanced stability checks for S6
            if torch.isnan(x_next).any() or torch.isinf(x_next).any():
                print(f"Warning: Invalid values at S6 step {i}, using previous state")
                x_next = x
            
            # S6-optimized self-conditioning update
            if self.velocity_predictor.use_self_conditioning and i % 2 == 0:  # Every other step
                with torch.no_grad():
                    self_cond = self.velocity_predictor(
                        x, t_batch, conditions, 
                        cfg_scale=1.0,
                        return_raw=True,
                        use_chunked_processing=use_s6_optimizations
                    ).detach()
            
            x = x_next
            trajectory.append(x.clone())
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, num_steps)
        
        return x, trajectory
    
    @torch.no_grad()
    def s6_flow_edit(
        self,
        original: torch.Tensor,
        mask: torch.Tensor,
        new_conditions: Dict,
        edit_steps: int = 8,
        edit_strength: float = 0.8,
        blend_method: str = "s6_smooth"
    ) -> torch.Tensor:
        """
        S6-optimized flow-based editing
        """
        device = original.device
        batch_size = original.shape[0]
        seq_len = original.shape[-1]
        
        # S6-optimized edit schedule
        edit_timesteps = self.scheduler.get_timesteps(edit_steps, device)
        
        # S6 chunk schedule for editing
        chunk_schedule = None
        if self.enable_s6_optimizations and seq_len > self.chunk_size:
            chunk_schedule = self.scheduler.get_s6_chunk_schedule(seq_len, self.chunk_size)
        
        x = original.clone()
        
        # S6-enhanced forward process
        for i in range(edit_steps):
            t_curr = edit_timesteps[i]
            t_next = edit_timesteps[i + 1] if i < edit_steps - 1 else torch.tensor(1.0, device=device)
            dt = t_next - t_curr
            
            t_batch = t_curr.expand(batch_size)
            
            # Get velocities with S6 optimization
            empty_conditions = self.velocity_predictor._create_empty_conditions(new_conditions)
            
            if chunk_schedule:
                v_original = self.velocity_predictor._chunked_forward(x, t_batch, empty_conditions)
                v_edited = self.velocity_predictor._chunked_forward(x, t_batch, new_conditions)
            else:
                v_original = self.velocity_predictor(x, t_batch, empty_conditions, return_raw=True)
                v_edited = self.velocity_predictor(x, t_batch, new_conditions, return_raw=True)
            
            # S6-enhanced velocity blending
            v_blend = v_original + mask * edit_strength * (v_edited - v_original)
            
            # Apply temporal smoothing for S6
            if blend_method == "s6_smooth" and i > 0:
                # Smooth across chunk boundaries
                if chunk_schedule and len(chunk_schedule) > 1:
                    v_blend = self._smooth_chunk_boundaries(v_blend, chunk_schedule)
            
            # Integration step
            x = x + dt * v_blend
        
        # S6-optimized final blending
        if blend_method == "s6_smooth":
            # Gaussian smoothing optimized for S6 chunks
            final_mask = self._s6_smooth_mask(mask, chunk_schedule)
        else:
            final_mask = mask
        
        result = original * (1 - final_mask) + x * final_mask
        return result
    
    def _smooth_chunk_boundaries(self, velocity: torch.Tensor, chunk_schedule: List[int]) -> torch.Tensor:
        """Smooth velocity across S6 chunk boundaries"""
        if not chunk_schedule or len(chunk_schedule) <= 1:
            return velocity
        
        smoothed = velocity.clone()
        start_idx = 0
        
        for i, chunk_size in enumerate(chunk_schedule[:-1]):
            boundary_idx = start_idx + chunk_size
            if boundary_idx < velocity.shape[-1]:
                # Apply smoothing around boundary
                smooth_width = min(8, chunk_size // 8)
                start_smooth = max(0, boundary_idx - smooth_width)
                end_smooth = min(velocity.shape[-1], boundary_idx + smooth_width)
                
                # Simple moving average
                for j in range(start_smooth, end_smooth):
                    weight = 1.0 - abs(j - boundary_idx) / smooth_width
                    if j > 0 and j < velocity.shape[-1] - 1:
                        smoothed[:, :, j] = (
                            weight * velocity[:, :, j] +
                            (1 - weight) * 0.5 * (velocity[:, :, j-1] + velocity[:, :, j+1])
                        )
            
            start_idx += chunk_size
        
        return smoothed
    
    def _s6_smooth_mask(self, mask: torch.Tensor, chunk_schedule: Optional[List[int]]) -> torch.Tensor:
        """S6-optimized mask smoothing"""
        if chunk_schedule is None:
            return mask
        
        # Apply Gaussian smoothing that respects chunk boundaries
        kernel_size = 9
        sigma = 2.0
        
        # Create Gaussian kernel
        kernel = torch.exp(-torch.arange(kernel_size, dtype=torch.float32).pow(2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.to(mask.device).view(1, 1, -1)
        
        # Apply smoothing
        padding = kernel_size // 2
        mask_smooth = F.conv1d(
            mask.float(), 
            kernel, 
            padding=padding
        )
        
        return mask_smooth


# ==================== S6 Flow Configuration ====================

class S6FlowConfig:
    """S6-optimized flow matching configuration"""
    
    def __init__(self):
        # S6-optimized flow settings
        self.flow_type = "rectified"
        self.scheduler_type = "s6_cosine"
        self.solver_type = "s6_heun"
        self.sigma = 1e-4
        
        # S6-specific parameters
        self.chunk_size = 256
        self.enable_s6_optimizations = True
        self.use_chunked_solver = True
        self.adaptive_chunking = True
        
        # Generation settings optimized for S6
        self.flow_steps = 10
        self.cfg_scale = 1.5
        self.use_self_conditioning = True
        
        # S6 memory optimization
        self.max_sequence_length = 8192
        self.chunk_overlap = 16
        self.gradient_checkpointing = True
        
        # S6 quality presets
        self.quality_presets = {
            "s6_fast": {
                "flow_steps": 6,
                "solver_type": "s6_euler",
                "cfg_scale": 1.2,
                "chunk_size": 128,
            },
            "s6_standard": {
                "flow_steps": 10,
                "solver_type": "s6_heun", 
                "cfg_scale": 1.5,
                "chunk_size": 256,
            },
            "s6_premium": {
                "flow_steps": 16,
                "solver_type": "s6_adaptive",
                "cfg_scale": 2.0,
                "chunk_size": 512,
            }
        }
    
    def apply_preset(self, preset: str):
        """Apply S6-optimized quality preset"""
        if preset in self.quality_presets:
            for key, value in self.quality_presets[preset].items():
                setattr(self, key, value)
        else:
            print(f"Warning: Unknown S6 preset '{preset}', using standard")


# ==================== S6 Factory Functions ====================

def create_s6_flow_matching(
    model: nn.Module,
    config: Optional[S6FlowConfig] = None,
    use_torch_compile: bool = True,
    compile_mode: str = "default"
) -> LyroS6FlowMatching:
    """
    Create S6-optimized flow matching model
    
    Args:
        model: S6-based neural network model
        config: S6 flow matching configuration
        use_torch_compile: Enable torch.compile() optimization
        compile_mode: Compilation mode
        
    Returns:
        Optimized LyroS6FlowMatching model
    """
    
    if config is None:
        config = S6FlowConfig()
    
    # Create S6 flow matching model
    flow_matching = LyroS6FlowMatching(
        model=model,
        scheduler_type=config.scheduler_type,
        solver_type=config.solver_type,
        sigma=config.sigma,
        flow_type=config.flow_type,
        chunk_size=config.chunk_size,
        enable_s6_optimizations=config.enable_s6_optimizations,
        use_chunked_solver=config.use_chunked_solver,
    )
    
    # Apply torch.compile() if requested
    if use_torch_compile and hasattr(torch, 'compile'):
        try:
            print(f"Applying torch.compile() to S6 FlowMatching with mode: {compile_mode}")
            
            # Platform-specific optimization
            import platform
            if platform.system() == "Windows":
                safe_compile_mode = "reduce-overhead" if compile_mode == "default" else compile_mode
                print(f"Windows detected, using safer compile mode: {safe_compile_mode}")
            else:
                safe_compile_mode = compile_mode
            
            # Compile S6 velocity model with special considerations
            flow_matching.model = torch.compile(
                flow_matching.model, 
                mode=safe_compile_mode,
                options={
                    "triton.cudagraphs": False,  # Disable for S6 dynamic shapes
                    "shape_padding": True,       # Help with S6 variable chunk sizes
                }
            )
            
            # Store compilation info
            flow_matching._use_torch_compile = True
            flow_matching._compile_mode = safe_compile_mode
            
            print("✓ torch.compile() applied successfully to S6 FlowMatching")
            
        except Exception as e:
            print(f"Warning: torch.compile() failed for S6: {e}")
            print("Continuing without compilation...")
            flow_matching._use_torch_compile = False
    else:
        flow_matching._use_torch_compile = False
    
    return flow_matching


def benchmark_s6_flow_matching(
    flow_matching: LyroS6FlowMatching,
    shape: Tuple[int, int, int],
    conditions: Dict,
    steps_list: List[int] = [4, 6, 8, 10, 12, 16]
) -> Dict[str, float]:
    """Benchmark S6 flow matching performance"""
    
    import time
    
    results = {}
    
    for steps in steps_list:
        start_time = time.time()
        
        with torch.no_grad():
            generated, _ = flow_matching.generate(
                shape=shape,
                conditions=conditions,
                num_steps=steps,
                cfg_scale=1.5,
                use_s6_optimizations=True
            )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Calculate S6-specific metrics
        audio_duration = shape[2] * 512 * 32 / 44100  # Accounting for compression
        rtf = generation_time / audio_duration
        
        # S6 efficiency metrics
        s6_efficiency = 1.0 / (rtf * steps)  # Higher is better
        throughput = shape[2] / generation_time  # Tokens per second
        
        results[f"s6_{steps}_steps"] = {
            "time": generation_time,
            "rtf": rtf,
            "steps": steps,
            "s6_efficiency": s6_efficiency,
            "throughput_tokens_per_sec": throughput,
            "chunk_size": flow_matching.chunk_size,
            "s6_optimizations": flow_matching.enable_s6_optimizations
        }
    
    return results


# ==================== Backward Compatibility ====================

# Alias for backward compatibility
LyroFlowMatching = LyroS6FlowMatching
FlowConfig = S6FlowConfig
create_flow_matching = create_s6_flow_matching
benchmark_flow_matching = benchmark_s6_flow_matching

print("S6-optimized Flow Matching implementation completed!")
print("Key S6 enhancements:")
print("- Chunked processing for long sequences")
print("- S6-aware schedulers and solvers")
print("- Memory-efficient velocity prediction")
print("- Enhanced self-conditioning for S6")
print("- Chunk boundary smoothing")
print("- Adaptive chunking strategies")