# lyro/ssm/config.py - Simplified Large S6 Model Configuration
"""
S6-Enhanced Model Configuration - Large Model Only
Focus: Flow Matching integration, numerical stability, FP16 enforcement
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import torch
import os

# Disable torch compile globally
import torch._dynamo
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'


@dataclass
class S6ModelConfig:
    """Simplified S6 Model Configuration for Large Model (16 input channels)"""
    
    # ==================== Model Architecture ====================
    # Fixed for large model compatibility with DCAE
    input_channels: int = 16  # Large DCAE latent channels
    
    # S6 U-Net architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 768])
    s6_layers: List[int] = field(default_factory=lambda: [3, 4, 4])
    d_state: int = 64
    max_seq_len: int = 2048
    
    # S6 State Space parameters
    d_head: int = 64
    d_conv: int = 4
    expand: int = 2
    
    # ==================== Conditioning Settings ====================
    # Task and conditional embedding
    task_embedding_dim: int = 256
    time_embedding_dim: int = 256
    style_embedding_dim: int = 512
    
    # Text conditioning
    max_text_length: int = 512
    text_embedding_dim: int = 256
    
    # ==================== Training Settings ====================
    # Optimizer
    learning_rate: float = 8e-5  # Conservative for large model
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    
    # Training dynamics
    batch_size: int = 2  # Small for large model
    epochs: int = 100
    grad_clip: float = 1.0
    
    # Scheduler
    min_lr: float = 1e-6
    warmup_steps: int = 1000
    
    # ==================== Stability Settings ====================
    # Critical for large model stability
    disable_torch_compile: bool = True
    disable_dynamo_tracing: bool = True
    enable_numerical_stability: bool = True
    gradient_checkpointing: bool = False  # Disabled to prevent NaN
    
    # Mixed precision
    mixed_precision: str = "fp16"
    force_fp16: bool = True
    
    # ==================== Data Settings ====================
    # Audio processing
    sample_rate: int = 44100
    max_audio_length: int = 441000  # 10 seconds
    
    # Data loading (DDP safe)
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    
    # ==================== Checkpoint Settings ====================
    save_interval: int = 10
    keep_last_n: int = 3
    save_best_only: bool = True
    
    # ==================== Logging Settings ====================
    log_interval: int = 50
    sample_interval: int = 20
    val_check_interval: int = 5
    
    def validate(self) -> Dict[str, Any]:
        """Validate S6 model configuration"""
        issues = []
        warnings = []
        
        # Model architecture validation
        if self.input_channels != 16:
            issues.append(f"Large model requires 16 input channels, got {self.input_channels}")
        
        if len(self.hidden_dims) != len(self.s6_layers):
            issues.append("hidden_dims and s6_layers must have same length")
        
        # Memory validation
        estimated_memory = self._estimate_memory_usage()
        if estimated_memory > 22:  # 22GB limit for V100
            warnings.append(f"Estimated memory {estimated_memory:.1f}GB may exceed V100 capacity")
        
        # Training stability validation
        if self.learning_rate > 1e-4:
            warnings.append(f"High learning rate {self.learning_rate} may cause instability")
        
        if self.batch_size > 4:
            warnings.append(f"Large batch size {self.batch_size} may cause OOM")
        
        if not self.disable_torch_compile:
            issues.append("torch.compile must be disabled for DDP compatibility")
        
        if self.gradient_checkpointing:
            warnings.append("Gradient checkpointing may cause NaN issues")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'estimated_memory_gb': estimated_memory,
            'estimated_parameters': self._estimate_parameters()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate GPU memory usage for large S6 model"""
        # Model memory (large S6 U-Net)
        model_memory = 3.5  # ~3.5GB for large S6 model
        
        # Batch memory (per sample)
        sequence_length = self.max_seq_len
        hidden_dim = max(self.hidden_dims)
        batch_memory = self.batch_size * sequence_length * hidden_dim * 4 / (1024**3)  # FP32 bytes to GB
        
        # Flow matching overhead
        flow_memory = 1.0
        
        # System overhead
        overhead = 1.5
        
        return model_memory + batch_memory + flow_memory + overhead
    
    def _estimate_parameters(self) -> int:
        """Estimate S6 model parameters"""
        # S6 U-Net parameters estimation
        base_params = 30_000_000  # Base U-Net parameters
        
        # S6 layers parameters
        s6_params = 0
        for i, (hidden_dim, num_layers) in enumerate(zip(self.hidden_dims, self.s6_layers)):
            layer_params = hidden_dim * self.d_state * 4 * num_layers  # Rough estimation
            s6_params += layer_params
        
        # Conditional embedding parameters
        conditioning_params = (
            self.task_embedding_dim * 16 +  # Task embeddings
            self.time_embedding_dim * 2 +   # Time embeddings
            self.style_embedding_dim * 2    # Style embeddings
        )
        
        return base_params + s6_params + conditioning_params
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get core model configuration"""
        return {
            'input_channels': self.input_channels,
            'hidden_dims': self.hidden_dims,
            's6_layers': self.s6_layers,
            'd_state': self.d_state,
            'max_seq_len': self.max_seq_len,
            'force_fp16': True,
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'grad_clip': self.grad_clip,
            'mixed_precision': self.mixed_precision,
            'disable_torch_compile': self.disable_torch_compile,
        }


@dataclass
class FlowMatchingConfig:
    """Simplified Flow Matching Configuration for Large S6 Model"""
    
    # ==================== Flow Matching Core Settings ====================
    # Flow configuration
    flow_type: str = "rectified"  # "rectified" or "standard"
    scheduler_type: str = "cosine"  # "linear" or "cosine"
    solver_type: str = "heun"  # "euler" or "heun"
    sigma: float = 1e-4
    
    # Generation settings
    flow_steps: int = 10
    cfg_scale: float = 1.5
    use_cfg: bool = True
    
    # ==================== Quality Presets ====================
    quality_presets: Dict[str, Dict] = field(default_factory=lambda: {
        'fast': {
            'flow_steps': 6,
            'solver_type': 'euler',
            'cfg_scale': 1.2,
        },
        'standard': {
            'flow_steps': 10,
            'solver_type': 'heun',
            'cfg_scale': 1.5,
        },
        'high': {
            'flow_steps': 16,
            'solver_type': 'heun',
            'cfg_scale': 2.0,
        }
    })
    
    # ==================== Training Settings ====================
    # Loss settings
    flow_matching_weight: float = 1.0
    
    # ==================== Stability Settings ====================
    enable_numerical_stability: bool = True
    clamp_values: bool = True
    max_velocity_norm: float = 10.0
    
    def apply_preset(self, preset: str):
        """Apply quality preset"""
        if preset in self.quality_presets:
            for key, value in self.quality_presets[preset].items():
                if hasattr(self, key):
                    setattr(self, key, value)
        else:
            print(f"Warning: Unknown preset '{preset}'")
    
    def validate(self) -> Dict[str, Any]:
        """Validate flow matching configuration"""
        issues = []
        warnings = []
        
        if self.flow_steps < 1:
            issues.append("flow_steps must be >= 1")
        
        if self.flow_steps > 20:
            warnings.append(f"High flow_steps ({self.flow_steps}) may be slow")
        
        if self.cfg_scale < 1.0:
            warnings.append(f"CFG scale {self.cfg_scale} < 1.0 may reduce quality")
        
        if self.cfg_scale > 3.0:
            warnings.append(f"High CFG scale {self.cfg_scale} may cause instability")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }


@dataclass
class S6TrainingConfig:
    """Simplified S6 Training Configuration"""
    
    # ==================== Paths ====================
    train_metadata: str = "dataset/metadata/train_metadata.jsonl"
    val_metadata: str = "dataset/metadata/val_metadata.jsonl"
    dataset_root: str = "dataset/"
    checkpoint_dir: str = "ssm/checkpoints_large_s6"
    
    # ==================== Model Checkpoints ====================
    dcae_checkpoint: str = ""  # Required: path to DCAE checkpoint
    resume: Optional[str] = None
    
    # ==================== Training Control ====================
    use_wandb: bool = False
    save_samples: bool = True
    
    # ==================== Hardware ====================
    device: str = "cuda"
    
    def __post_init__(self):
        """Post-initialization validation"""
        if not self.dcae_checkpoint:
            print("Warning: dcae_checkpoint path is required")
        
        if not torch.cuda.is_available() and self.device == "cuda":
            print("Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"


@dataclass
class TaskConfig:
    """Task-specific configuration for music generation"""
    
    # ==================== Task Settings ====================
    # Task ratios for training
    task_ratios: Dict[str, float] = field(default_factory=lambda: {
        'SONG': 0.6,
        'INST': 0.2, 
        'COVER': 0.2
    })
    
    # ==================== Generation Settings ====================
    # Default generation parameters
    default_duration: float = 10.0  # seconds
    max_duration: float = 30.0
    
    # Task-specific settings
    song_settings: Dict[str, Any] = field(default_factory=lambda: {
        'require_lyrics': True,
        'default_style': 'pop',
        'cfg_scale': 1.5
    })
    
    instrumental_settings: Dict[str, Any] = field(default_factory=lambda: {
        'require_lyrics': False,
        'default_style': 'instrumental',
        'cfg_scale': 1.2
    })
    
    cover_settings: Dict[str, Any] = field(default_factory=lambda: {
        'require_reference': True,
        'style_transfer_strength': 0.7,
        'cfg_scale': 1.8
    })
    
    def validate_task_ratios(self) -> bool:
        """Validate that task ratios sum to 1.0"""
        total = sum(self.task_ratios.values())
        if abs(total - 1.0) > 1e-6:
            print(f"Warning: Task ratios sum to {total}, should be 1.0")
            return False
        return True


# ==================== Factory Functions ====================

def create_large_s6_config(
    input_channels: int = 16,
    batch_size: int = 2,
    learning_rate: float = 8e-5,
    flow_steps: int = 10,
    **kwargs
) -> Tuple[S6ModelConfig, FlowMatchingConfig]:
    """
    Create configuration for Large S6 model with Flow Matching
    
    Args:
        input_channels: Input channels (must be 16 for large DCAE)
        batch_size: Training batch size
        learning_rate: Learning rate
        flow_steps: Flow matching steps
        **kwargs: Additional config overrides
    
    Returns:
        Tuple of (S6ModelConfig, FlowMatchingConfig)
    """
    # Model config
    model_config = S6ModelConfig(
        input_channels=input_channels,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    # Flow config
    flow_config = FlowMatchingConfig(
        flow_steps=flow_steps,
    )
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
        elif hasattr(flow_config, key):
            setattr(flow_config, key, value)
        else:
            print(f"Warning: Unknown config parameter: {key}")
    
    # Validate configurations
    model_validation = model_config.validate()
    flow_validation = flow_config.validate()
    
    # Check for issues
    all_issues = model_validation['issues'] + flow_validation['issues']
    if all_issues:
        print("❌ Configuration validation failed:")
        for issue in all_issues:
            print(f"   - {issue}")
        raise ValueError("Configuration validation failed")
    
    # Report warnings
    all_warnings = model_validation['warnings'] + flow_validation['warnings']
    if all_warnings:
        print("⚠️ Configuration warnings:")
        for warning in all_warnings:
            print(f"   - {warning}")
    
    print(f"✅ Large S6 Config Created:")
    print(f"   - Parameters: ~{model_validation['estimated_parameters']:,}")
    print(f"   - Memory: ~{model_validation['estimated_memory_gb']:.1f}GB")
    print(f"   - Input Channels: {input_channels}")
    print(f"   - Flow Steps: {flow_steps}")
    
    return model_config, flow_config


def create_stable_s6_config(**kwargs) -> Tuple[S6ModelConfig, FlowMatchingConfig]:
    """Create maximally stable S6 configuration"""
    return create_large_s6_config(
        batch_size=1,  # Very conservative
        learning_rate=5e-5,  # Lower learning rate
        grad_clip=0.5,  # Conservative gradient clipping
        flow_steps=8,  # Fewer steps for stability
        cfg_scale=1.2,  # Lower CFG scale
        **kwargs
    )


def create_fast_s6_config(**kwargs) -> Tuple[S6ModelConfig, FlowMatchingConfig]:
    """Create fast training S6 configuration"""
    return create_large_s6_config(
        batch_size=4,  # Larger batch if memory allows
        learning_rate=1e-4,  # Higher learning rate
        flow_steps=6,  # Fewer steps for speed
        solver_type='euler',  # Faster solver
        **kwargs
    )


def create_high_quality_s6_config(**kwargs) -> Tuple[S6ModelConfig, FlowMatchingConfig]:
    """Create high quality S6 configuration"""
    return create_large_s6_config(
        learning_rate=6e-5,  # Lower for quality
        flow_steps=16,  # More steps for quality
        cfg_scale=2.0,  # Higher CFG scale
        solver_type='heun',  # More accurate solver
        **kwargs
    )


# ==================== Preset Classes ====================

class S6Presets:
    """Predefined S6 configuration presets"""
    
    @staticmethod
    def development() -> Tuple[S6ModelConfig, FlowMatchingConfig]:
        """Development preset - fast iteration"""
        return create_large_s6_config(
            batch_size=1,
            epochs=50,
            flow_steps=6,
            save_interval=5,
            sample_interval=10,
        )
    
    @staticmethod
    def production() -> Tuple[S6ModelConfig, FlowMatchingConfig]:
        """Production preset - balanced quality/speed"""
        return create_large_s6_config(
            batch_size=2,
            epochs=200,
            flow_steps=10,
            learning_rate=8e-5,
        )
    
    @staticmethod
    def research() -> Tuple[S6ModelConfig, FlowMatchingConfig]:
        """Research preset - comprehensive logging"""
        return create_large_s6_config(
            batch_size=2,
            save_interval=5,
            log_interval=25,
            sample_interval=10,
            flow_steps=12,
        )


# ==================== Unified Configuration ====================

@dataclass
class LyroS6Config:
    """Unified configuration for LYRO S6 system"""
    
    model: S6ModelConfig
    flow: FlowMatchingConfig
    training: S6TrainingConfig
    task: TaskConfig
    
    @classmethod
    def create_large_model_config(
        cls,
        dcae_checkpoint: str,
        preset: str = "production",
        **kwargs
    ) -> "LyroS6Config":
        """
        Create unified configuration for large S6 model
        
        Args:
            dcae_checkpoint: Path to DCAE checkpoint
            preset: Configuration preset ("development", "production", "research")
            **kwargs: Additional overrides
        """
        # Get preset configurations
        if preset == "development":
            model_config, flow_config = S6Presets.development()
        elif preset == "production":
            model_config, flow_config = S6Presets.production()
        elif preset == "research":
            model_config, flow_config = S6Presets.research()
        else:
            model_config, flow_config = S6Presets.production()
            print(f"Warning: Unknown preset '{preset}', using 'production'")
        
        # Training config
        training_config = S6TrainingConfig(dcae_checkpoint=dcae_checkpoint)
        
        # Task config
        task_config = TaskConfig()
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
            elif hasattr(flow_config, key):
                setattr(flow_config, key, value)
            elif hasattr(training_config, key):
                setattr(training_config, key, value)
            elif hasattr(task_config, key):
                setattr(task_config, key, value)
            else:
                print(f"Warning: Unknown config parameter: {key}")
        
        return cls(
            model=model_config,
            flow=flow_config,
            training=training_config,
            task=task_config
        )
    
    def validate_full_config(self) -> Dict[str, Any]:
        """Validate complete configuration"""
        model_validation = self.model.validate()
        flow_validation = self.flow.validate()
        task_validation = {'valid': self.task.validate_task_ratios()}
        
        all_issues = (
            model_validation['issues'] + 
            flow_validation['issues'] +
            ([] if task_validation['valid'] else ['Invalid task ratios'])
        )
        
        all_warnings = model_validation['warnings'] + flow_validation['warnings']
        
        return {
            'valid': len(all_issues) == 0,
            'issues': all_issues,
            'warnings': all_warnings,
            'model_validation': model_validation,
            'flow_validation': flow_validation,
            'task_validation': task_validation
        }


print("✅ S6 Configuration - Large Model Simplified")
print("Key features:")
print("- Large S6 model focused (16 input channels)")
print("- Flow Matching integration")
print("- Numerical stability enforced")
print("- DDP compatibility ensured")
print("- Comprehensive validation system")
print("- Unified configuration management")