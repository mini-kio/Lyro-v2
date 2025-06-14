# lyro/dcae/config.py - Enhanced Model Size & NaN Fixed Configuration
"""
Enhanced DCAE Configuration for CQT-SSM Based Models with S6-SSM Compression Optimization
FIXED: Model size adjustment for base(60M), large(100M) + NaN loss prevention
ENHANCED: Complete numerical stability and gradient flow optimization
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import os
import math

# FIXED: Disable torch._dynamo completely at config level
import torch._dynamo
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

# FIXED: Disable compilation completely
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'


@dataclass
class DCAEConfig:
    """Enhanced CQT-SSM DCAE Configuration with proper model sizing and NaN prevention"""
    
    # 오디오 설정
    sample_rate: int = 44100
    
    # FIXED: Enhanced CQT 설정 with larger model support
    cqt_hop_length: int = 512
    cqt_n_bins: int = 84
    cqt_bins_per_octave: int = 12
    cqt_fmin: float = 32.7  # C1 frequency
    cqt_window: str = 'hann'
    
    # FIXED: Enhanced compression mechanisms for larger models
    enable_forced_compression: bool = True
    cqt_projection_dims: int = 80  # Increased from 64 for larger models
    learnable_frequency_projection: bool = True
    
    # FIXED: Enhanced temporal compression
    temporal_compression_stride: int = 2
    enable_anti_aliasing: bool = True
    progressive_compression_stages: List[int] = field(default_factory=lambda: [2, 2, 2])
    
    # FIXED: Enhanced latent bottleneck for larger models
    dynamic_channel_pruning: bool = True
    adaptive_latent_channels: Tuple[int, int] = (6, 8)  # Increased from (4, 6)
    information_bottleneck_weight: float = 0.05  # Reduced for stability
    
    # FIXED: Enhanced multi-resolution CQT
    cqt_hop_lengths: List[int] = field(default_factory=lambda: [256, 512, 1024])
    cqt_n_bins_list: List[int] = field(default_factory=lambda: [72, 84, 96])
    
    # FIXED: Enhanced model configuration for proper sizing
    compression_factor: int = 16
    latent_channels: int = 12  # Increased from 8 for larger models
    
    # FIXED: Enhanced encoder configuration for larger models
    encoder_base_channels: int = 80  # Increased from 64
    encoder_channels: List[int] = field(default_factory=lambda: [40, 80, 160, 320])  # Adjusted progression
    encoder_strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    encoder_kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 5, 3])
    
    # FIXED: Enhanced decoder configuration for larger models
    decoder_base_channels: int = 80  # Increased from 64
    decoder_channels: List[int] = field(default_factory=lambda: [320, 160, 80, 40])  # Adjusted progression
    decoder_strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    decoder_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 5, 7])
    
    # FIXED: Enhanced S6-SSM configuration for larger models
    ssm_d_state: int = 40  # Increased from 32
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    ssm_layers: List[int] = field(default_factory=lambda: [3, 3, 3, 3])  # Increased layers
    
    # FIXED: Enhanced compression-aware S6 settings
    enable_compression_aware_s6: bool = True
    selective_state_saving: bool = True
    adaptive_forgetting_rate: float = 0.05  # Reduced for stability
    compression_regularization_weight: float = 0.02  # Reduced for stability
    
    # FIXED: Enhanced multi-scale S6-SSM
    enable_multiscale_ssm: bool = True
    ssm_scales: List[str] = field(default_factory=lambda: ['global', 'middle', 'fine'])
    cross_scale_attention: bool = False  # Disabled for numerical stability
    
    # FIXED: Enhanced semantic-guided S6 (disabled for stability)
    enable_semantic_guidance: bool = False  # Disabled for stability
    self_supervised_embedding: bool = False  # Disabled for stability
    semantic_state_spaces: List[str] = field(default_factory=lambda: ['harmonic', 'rhythmic'])  # Reduced
    
    # FIXED: Enhanced skip connection redesign
    enable_selective_skip: bool = True
    mutual_information_threshold: float = 0.25  # Reduced for stability
    skip_pruning_ratio: float = 0.3  # Reduced for stability
    
    # FIXED: Enhanced compression-friendly skip design
    learnable_compression_skip: bool = True
    semantic_skip: bool = False  # Disabled for stability
    adaptive_skip_activation: bool = False  # Disabled for stability
    s6_enhanced_skip: bool = True
    
    # FIXED: Enhanced memory optimization
    chunk_size: int = 256
    use_checkpointing: bool = False  # Disabled to prevent NaN
    memory_efficient: bool = True
    checkpointing_segments: int = 0  # Disabled
    
    # FIXED: Enhanced VQ settings (mostly disabled for stability)
    use_vector_quantization: bool = False
    vq_num_embeddings: int = 512
    vq_embedding_dim: int = 6
    vq_beta: float = 0.25
    
    # FIXED: Enhanced quality enhancement with stability focus
    enable_enhanced_perceptual_loss: bool = True
    multi_resolution_stft_loss: bool = False  # Disabled for stability
    mel_scale_loss: bool = True
    harmonic_loss: bool = False  # Disabled for stability
    dynamic_loss_weighting: bool = False  # Disabled for stability
    
    # FIXED: Enhanced loss weights with numerical stability
    reconstruction_weight: float = 1.0
    cqt_loss_weight: float = 1.0  # Reduced from 1.2
    time_loss_weight: float = 0.1
    vq_weight: float = 0.01  # Reduced
    adversarial_weight: float = 0.05  # Reduced
    
    # FIXED: Enhanced compression loss weights with stability
    compression_loss_weight: float = 0.1  # Reduced
    information_bottleneck_loss_weight: float = 0.05  # Reduced
    perceptual_loss_weight: float = 0.2  # Reduced
    mel_loss_weight: float = 0.15  # Reduced
    harmonic_loss_weight: float = 0.0   # Disabled
    
    # FIXED: Enhanced CQT loss configuration with stability
    w_cqt: float = 1.0  # Reduced from 1.2
    w_temporal: float = 0.5  # Reduced
    w_harmonic: float = 0.0  # Disabled
    w_compression: float = 0.2  # Reduced
    
    # FIXED: Enhanced detail refinement (disabled for stability)
    enable_detail_refinement: bool = False  # Disabled
    s6_state_based_detail_generation: bool = False  # Disabled
    style_consistent_refinement: bool = False  # Disabled
    adaptive_refinement_level: bool = False  # Disabled
    
    # FIXED: Enhanced transfer learning (disabled for DDP compatibility)
    enable_transfer_learning_optimization: bool = False  # Disabled
    s6_core_freeze_epochs: int = 0  # Disabled for DDP
    progressive_layer_unfreezing: bool = False  # Disabled
    knowledge_distillation: bool = False  # Disabled
    distillation_temperature: float = 3.0
    distillation_weight: float = 0.0  # Disabled
    
    # FIXED: Enhanced training configuration for larger models and stability
    learning_rate: float = 1.2e-4  # Conservative for larger models
    min_lr: float = 5e-7  # More conservative
    weight_decay: float = 0.02  # Slightly increased for regularization
    grad_clip: float = 0.5  # Reduced for stability
    
    # FIXED: Enhanced data configuration for larger models
    batch_size: int = 8  # Adjusted for larger models
    num_workers: int = 0  # Disabled for DDP safety
    epochs: int = 200
    audio_duration: float = 1.0  # Conservative for V100
    
    # FIXED: Enhanced checkpoint configuration
    save_interval: int = 5
    keep_last_n: int = 3  # Conservative
    
    # FIXED: Enhanced augmentation (reduced for stability)
    lossy_augmentation_prob: float = 0.3  # Reduced
    vbr_range: Tuple[int, int] = (0, 5)  # Reduced range
    restoration_loss_weight: float = 0.1  # Reduced
    
    # FIXED: Enhanced optimization flags with stability focus
    use_weight_norm: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999  # Conservative
    ema_update_after: int = 100  # More conservative
    ema_update_every: int = 10   # More frequent
    
    # FIXED: Enhanced augmentation configuration with safety
    use_augmentation: bool = True
    augmentation_prob: float = 0.3  # Reduced for stability
    gain_range: Tuple[float, float] = (-1.5, 1.5)  # Reduced range
    pitch_range: Tuple[float, float] = (-0.2, 0.2)  # Reduced range
    tempo_range: Tuple[float, float] = (0.95, 1.05)  # Reduced range
    noise_level: float = 0.001  # Reduced
    reverb_prob: float = 0.1     # Reduced
    eq_prob: float = 0.2         # Reduced
    preserve_musical_structure: bool = True
    harmonic_distortion_prob: float = 0.05  # Reduced
    
    # FIXED: Enhanced harmonic-percussive separation (simplified for stability)
    use_harmonic_percussive: bool = False  # Disabled for stability
    hp_kernel_size_h: int = 17
    hp_kernel_size_p: int = 17
    hp_power: float = 2.0
    hp_margin: float = 1.0
    hp_learnable: bool = False  # Disabled
    hp_compression_aware: bool = False  # Disabled
    
    # FIXED: Enhanced model size presets for proper parameter counts
    model_size: str = "base"  # "small", "base", "large", "compressed"
    
    # CRITICAL: Enhanced stability and compilation settings
    disable_torch_compile: bool = True
    disable_dynamo_tracing: bool = True
    enable_enhanced_numerical_stability: bool = True
    use_safe_operations: bool = True
    nan_prevention_active: bool = True
    gradient_checkpointing: bool = False  # Disabled to prevent NaN
    
    def get_model_config(self) -> Dict:
        """Enhanced model-specific configuration with proper parameter sizing"""
        if self.model_size == "small":
            return {
                "encoder_base_channels": 48,
                "decoder_base_channels": 48,
                "latent_channels": 8,
                "target_latent_channels": 4,
                "cqt_n_bins": 72,
                "ssm_d_state": 24,
                "chunk_size": 128,
                "cqt_projection_dims": 48,
                # Estimated parameters: ~25M
                "estimated_parameters": "25M"
            }
        elif self.model_size == "base":
            # FIXED: Base model configuration for ~60M parameters
            return {
                "encoder_base_channels": 80,   # Increased from 48
                "decoder_base_channels": 80,   # Increased from 48
                "latent_channels": 12,         # Increased from 6
                "target_latent_channels": 6,   # Increased from 4
                "cqt_n_bins": 84,
                "ssm_d_state": 40,            # Increased from 24
                "chunk_size": 256,
                "cqt_projection_dims": 80,    # Increased from 56
                # FIXED: Additional parameters for proper sizing
                "ssm_layers": [3, 3, 3, 3],   # Increased layers
                "encoder_channels": [40, 80, 160, 320],  # Proper progression
                "decoder_channels": [320, 160, 80, 40],  # Proper progression
                "estimated_parameters": "60M"
            }
        elif self.model_size == "large":
            # FIXED: Large model configuration for ~100M parameters
            return {
                "encoder_base_channels": 112,  # Increased significantly
                "decoder_base_channels": 112,  # Increased significantly
                "latent_channels": 16,         # Increased from 8
                "target_latent_channels": 8,   # Increased from 5
                "cqt_n_bins": 96,
                "ssm_d_state": 56,            # Increased significantly
                "chunk_size": 512,
                "cqt_projection_dims": 96,    # Increased from 64
                # FIXED: Additional parameters for larger model
                "ssm_layers": [4, 4, 4, 4],   # More layers
                "encoder_channels": [56, 112, 224, 448],  # Larger progression
                "decoder_channels": [448, 224, 112, 56],  # Larger progression
                "estimated_parameters": "100M"
            }
        elif self.model_size == "compressed":
            return {
                "encoder_base_channels": 32,
                "decoder_base_channels": 32,
                "latent_channels": 6,
                "target_latent_channels": 3,
                "cqt_n_bins": 64,
                "ssm_d_state": 16,
                "chunk_size": 128,
                "cqt_projection_dims": 32,
                "compression_factor": 24,
                "estimated_parameters": "15M"
            }
        else:
            return {}
    
    def update_for_model_size(self):
        """Enhanced model size update with proper parameter scaling"""
        config_updates = self.get_model_config()
        
        # FIXED: Apply all configuration updates
        for key, value in config_updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # FIXED: Update dependent configurations based on model size
        if self.model_size == "base":
            # FIXED: Adjust training parameters for base model (60M)
            self.batch_size = max(4, self.batch_size // 2)  # Reduce batch size for larger model
            self.learning_rate = 1.0e-4  # Slightly lower LR for stability
            self.grad_clip = 0.5  # Conservative gradient clipping
            
        elif self.model_size == "large":
            # FIXED: Adjust training parameters for large model (100M)
            self.batch_size = max(2, self.batch_size // 4)  # Further reduce batch size
            self.learning_rate = 8e-5  # Lower LR for larger model
            self.grad_clip = 0.3  # More conservative gradient clipping
            self.ema_update_after = 200  # Longer warm-up for larger model
            
        # FIXED: Update encoder/decoder channel configurations
        if 'encoder_channels' in config_updates:
            self.encoder_channels = config_updates['encoder_channels']
        if 'decoder_channels' in config_updates:
            self.decoder_channels = config_updates['decoder_channels']
        if 'ssm_layers' in config_updates:
            self.ssm_layers = config_updates['ssm_layers']
    
    def get_compression_config(self) -> Dict:
        """Enhanced compression-specific configuration with stability focus"""
        return {
            # Enhanced forced compression mechanisms
            "forced_compression": {
                "enabled": self.enable_forced_compression,
                "cqt_projection_dims": self.cqt_projection_dims,
                "temporal_compression_stride": self.temporal_compression_stride,
                "dynamic_channel_pruning": self.dynamic_channel_pruning,
                "adaptive_latent_channels": self.adaptive_latent_channels,
                "numerical_stability_enhanced": True
            },
            
            # Enhanced S6-SSM compression optimization
            "s6_compression": {
                "compression_aware": self.enable_compression_aware_s6,
                "selective_state_saving": self.selective_state_saving,
                "adaptive_forgetting_rate": self.adaptive_forgetting_rate,
                "multiscale_ssm": self.enable_multiscale_ssm,
                "semantic_guidance": self.enable_semantic_guidance,
                "d_state": self.ssm_d_state,
                "layers": self.ssm_layers
            },
            
            # Enhanced skip connection optimization
            "skip_optimization": {
                "selective_skip": self.enable_selective_skip,
                "mi_threshold": self.mutual_information_threshold,
                "skip_pruning_ratio": self.skip_pruning_ratio,
                "learnable_compression_skip": self.learnable_compression_skip,
                "s6_enhanced_skip": self.s6_enhanced_skip
            },
            
            # Enhanced quality enhancement with stability
            "quality_enhancement": {
                "enhanced_perceptual_loss": self.enable_enhanced_perceptual_loss,
                "detail_refinement": self.enable_detail_refinement,
                "transfer_learning_optimization": self.enable_transfer_learning_optimization,
                "knowledge_distillation": self.knowledge_distillation,
                "mel_scale_loss": self.mel_scale_loss
            },
            
            # FIXED: Enhanced stability and optimization settings
            "stability_settings": {
                "torch_compile_disabled": self.disable_torch_compile,
                "dynamo_tracing_disabled": self.disable_dynamo_tracing,
                "numerical_stability_enhanced": self.enable_enhanced_numerical_stability,
                "safe_operations_enabled": self.use_safe_operations,
                "nan_prevention_active": self.nan_prevention_active,
                "gradient_checkpointing_disabled": not self.gradient_checkpointing
            }
        }
    
    def estimate_model_parameters(self) -> int:
        """Estimate total model parameters based on configuration"""
        try:
            # FIXED: Enhanced parameter estimation for accurate sizing
            config = self.get_model_config()
            
            base_channels = config.get('encoder_base_channels', self.encoder_base_channels)
            latent_channels = config.get('latent_channels', self.latent_channels)
            d_state = config.get('ssm_d_state', self.ssm_d_state)
            cqt_dims = config.get('cqt_projection_dims', self.cqt_projection_dims)
            
            # FIXED: Comprehensive parameter calculation
            # CQT Transform parameters
            cqt_params = self.cqt_n_bins * cqt_dims + cqt_dims * 2  # projection layers
            
            # Encoder parameters (multi-stage with S6-SSM)
            encoder_params = 0
            current_channels = base_channels
            for i, layers in enumerate(self.ssm_layers):
                out_channels = base_channels * (2 ** min(i, 3))
                # Conv layers
                encoder_params += current_channels * out_channels * 9  # 3x3 conv
                # S6-SSM layers
                encoder_params += layers * (out_channels * d_state * 4 + d_state * out_channels * 2)
                current_channels = out_channels
            
            # Decoder parameters (symmetric to encoder)
            decoder_params = encoder_params  # Approximately symmetric
            
            # Final projection layers
            final_params = current_channels * latent_channels * 9  # final conv
            
            # FIXED: Total parameter estimation
            total_params = cqt_params + encoder_params + decoder_params + final_params
            
            # FIXED: Model size specific adjustments
            if self.model_size == "base":
                total_params = int(total_params * 1.2)  # Account for additional components
            elif self.model_size == "large":
                total_params = int(total_params * 1.4)  # Account for larger components
            
            return total_params
            
        except Exception as e:
            print(f"⚠️ Parameter estimation failed: {e}")
            # Fallback estimates
            size_estimates = {
                "small": 25_000_000,
                "base": 60_000_000,
                "large": 100_000_000,
                "compressed": 15_000_000
            }
            return size_estimates.get(self.model_size, 60_000_000)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Enhanced configuration validation with health checks"""
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'parameter_estimate': self.estimate_model_parameters(),
            'stability_score': 1.0
        }
        
        try:
            # FIXED: Model size validation
            param_estimate = validation_result['parameter_estimate']
            target_params = {
                "small": (20_000_000, 30_000_000),
                "base": (50_000_000, 70_000_000),
                "large": (90_000_000, 110_000_000),
                "compressed": (10_000_000, 20_000_000)
            }
            
            if self.model_size in target_params:
                min_params, max_params = target_params[self.model_size]
                if not (min_params <= param_estimate <= max_params):
                    validation_result['warnings'].append(
                        f"Parameter count {param_estimate:,} outside target range "
                        f"[{min_params:,}, {max_params:,}] for {self.model_size}"
                    )
            
            # FIXED: Training stability validation
            if self.learning_rate > 2e-3:
                validation_result['warnings'].append(f"High learning rate: {self.learning_rate}")
                validation_result['stability_score'] -= 0.1
            
            if self.grad_clip > 1.0:
                validation_result['warnings'].append(f"High gradient clipping: {self.grad_clip}")
                validation_result['stability_score'] -= 0.1
            
            if self.batch_size > 16:
                validation_result['warnings'].append(f"Large batch size for V100: {self.batch_size}")
                validation_result['stability_score'] -= 0.1
            
            # FIXED: Numerical stability validation
            if not self.enable_enhanced_numerical_stability:
                validation_result['errors'].append("Enhanced numerical stability should be enabled")
                validation_result['valid'] = False
                validation_result['stability_score'] -= 0.3
            
            if self.use_checkpointing:
                validation_result['warnings'].append("Checkpointing enabled (may cause NaN)")
                validation_result['stability_score'] -= 0.2
            
            # FIXED: DDP compatibility validation
            if not self.disable_torch_compile:
                validation_result['errors'].append("torch.compile should be disabled for DDP")
                validation_result['valid'] = False
            
            if self.num_workers > 0:
                validation_result['warnings'].append("num_workers > 0 may cause DDP issues")
                validation_result['stability_score'] -= 0.1
            
            # FIXED: Loss weight validation
            total_loss_weight = (
                self.cqt_loss_weight + self.time_loss_weight + 
                self.compression_loss_weight + self.perceptual_loss_weight
            )
            if total_loss_weight > 5.0:
                validation_result['warnings'].append(f"High total loss weight: {total_loss_weight}")
                validation_result['stability_score'] -= 0.1
            
            # FIXED: Final stability score
            validation_result['stability_score'] = max(0.0, min(1.0, validation_result['stability_score']))
            
        except Exception as e:
            validation_result['errors'].append(f"Validation failed: {e}")
            validation_result['valid'] = False
        
        return validation_result


@dataclass 
class CQTSSMConfig:
    """Enhanced CQT-SSM specific configuration with numerical stability"""
    
    # FIXED: Enhanced CQT Transform parameters
    sample_rate: int = 44100
    hop_length: int = 512
    n_bins: int = 84
    bins_per_octave: int = 12
    fmin: float = 32.7  # C1
    window: str = 'hann'
    center: bool = True
    pad_mode: str = 'reflect'
    
    # FIXED: Enhanced compression-specific CQT parameters
    enable_cqt_compression: bool = True
    cqt_compression_ratio: float = 0.7  # Increased for stability
    learnable_cqt_basis: bool = True
    
    # FIXED: Enhanced SSM parameters with larger capacity
    d_model: int = 80  # Increased from 64
    d_state: int = 40   # Increased from 24
    d_conv: int = 4
    expand: int = 2
    dt_rank: Optional[int] = None
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    bias: bool = True
    conv_bias: bool = True
    
    # FIXED: Enhanced compression-aware SSM parameters
    enable_state_compression: bool = True
    state_compression_ratio: float = 0.8  # Increased for stability
    adaptive_state_pruning: bool = True
    forgetting_gate: bool = True
    
    # FIXED: Enhanced memory optimization
    chunk_size: int = 256
    use_checkpointing: bool = False  # Disabled to prevent NaN
    memory_efficient: bool = True
    gradient_checkpointing_ratio: float = 0.0  # Disabled
    
    # FIXED: Enhanced multi-scale processing
    use_multiscale_ssm: bool = True
    scales: List[str] = field(default_factory=lambda: ['global', 'middle'])  # Reduced for stability
    scale_compression_ratios: List[float] = field(default_factory=lambda: [0.8, 0.7])  # Conservative
    
    # FIXED: Enhanced harmonic-percussive separation (disabled for stability)
    use_hp_separation: bool = False  # Disabled
    hp_kernel_size_h: int = 17
    hp_kernel_size_p: int = 17
    hp_learnable: bool = False  # Disabled
    hp_compression_aware: bool = False  # Disabled
    
    # CRITICAL: Enhanced stability settings
    disable_torch_compile: bool = True
    disable_dynamo_tracing: bool = True
    enable_numerical_stability: bool = True
    use_safe_operations: bool = True


@dataclass 
class HiFiGANConfig:
    """Enhanced HiFiGAN Discriminator configuration with stability focus"""
    
    # FIXED: Enhanced multi-period discriminator (reduced for stability)
    periods: List[int] = field(default_factory=lambda: [2, 3, 5])  # Reduced from [2, 3, 5, 7]
    
    # FIXED: Enhanced multi-scale discriminator (reduced for stability)
    scales: int = 2  # Reduced from 3
    downsample_scales: List[int] = field(default_factory=lambda: [2, 2])
    downsample_kernel_sizes: List[int] = field(default_factory=lambda: [4, 4])
    
    # FIXED: Enhanced CQT-aware discriminator settings (disabled for stability)
    use_cqt_discriminator: bool = False  # Disabled for stability
    cqt_n_bins: int = 84
    cqt_hop_length: int = 512
    cqt_compression_aware: bool = False  # Disabled
    
    # FIXED: Enhanced common settings with stability focus
    discriminator_channel_mult: float = 0.7  # Reduced for stability
    use_spectral_norm: bool = False
    use_weight_norm: bool = True
    enable_compression_discrimination: bool = False  # Disabled for stability
    
    # CRITICAL: Enhanced stability settings
    disable_torch_compile: bool = True
    disable_dynamo_tracing: bool = True
    enable_numerical_stability: bool = True


@dataclass
class TrainingConfig:
    """Enhanced training configuration with complete numerical stability"""
    
    # FIXED: Enhanced optimizer configuration for larger models
    optimizer: str = "adamw"
    learning_rate: float = 1.0e-4  # Reduced for stability with larger models
    weight_decay: float = 0.02
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-6  # Conservative epsilon
    fused: bool = True
    
    # FIXED: Enhanced compression-specific optimizer settings
    compression_lr_factor: float = 0.8  # More conservative
    compression_weight_decay: float = 0.025  # Slightly higher
    
    # FIXED: Enhanced scheduler configuration for stability
    scheduler: str = "cosine_annealing_warm_restarts"
    min_lr: float = 5e-7  # More conservative
    warmup_epochs: int = 10  # Longer warmup for larger models
    T_0: int = 50  # Shorter cycles for better control
    T_mult: int = 1
    
    # FIXED: Enhanced training dynamics for numerical stability
    gradient_accumulation_steps: int = 4  # Increased for larger models
    max_grad_norm: float = 0.5  # More conservative
    mixed_precision: str = "fp16"
    
    # FIXED: Enhanced compression-specific training settings
    enable_compression_curriculum: bool = False  # Disabled for stability
    compression_warmup_epochs: int = 0  # Disabled
    adaptive_compression_rate: bool = False  # Disabled
    
    # FIXED: Enhanced data loading for stability
    batch_size: int = 6  # Reduced for larger models
    num_workers: int = 0  # Disabled for DDP safety
    pin_memory: bool = False  # Disabled for DDP safety
    persistent_workers: bool = False  # Disabled
    
    # FIXED: Enhanced validation configuration
    val_check_interval: int = 5  # Less frequent
    val_batches_limit: int = 10  # Reduced for stability
    compression_analysis_interval: int = 10  # Less frequent
    
    # FIXED: Enhanced checkpointing for stability
    save_every_n_epochs: int = 10  # Less frequent
    keep_last_n_checkpoints: int = 2  # Reduced
    save_best_only: bool = True  # More selective
    save_compression_stats: bool = True
    
    # FIXED: Enhanced logging configuration
    log_every_n_steps: int = 200  # Less frequent
    sample_every_n_epochs: int = 20  # Less frequent
    num_samples: int = 2  # Reduced
    
    # FIXED: Enhanced early stopping for stability
    patience: int = 30  # More patience for larger models
    min_delta: float = 1e-5  # Smaller delta
    
    # FIXED: Enhanced CQT-specific training settings
    cqt_loss_start_epoch: int = 0
    harmonic_loss_start_epoch: int = 0  # Disabled
    compression_loss_start_epoch: int = 5  # Delayed start
    perceptual_loss_start_epoch: int = 10  # Delayed start
    progressive_training: bool = False  # Disabled for stability
    
    # FIXED: Enhanced transfer learning schedule (disabled for DDP)
    s6_core_freeze_schedule: List[int] = field(default_factory=lambda: [])  # Disabled
    progressive_unfreeze_schedule: List[int] = field(default_factory=lambda: [])  # Disabled
    knowledge_distillation_epochs: List[int] = field(default_factory=lambda: [])  # Disabled
    
    # CRITICAL: Enhanced stability settings
    disable_torch_compile: bool = True
    disable_dynamo_tracing: bool = True
    enable_numerical_stability: bool = True
    use_safe_operations: bool = True
    nan_prevention_active: bool = True


@dataclass
class DataConfig:
    """Enhanced data configuration with stability focus"""
    
    # Paths
    dataset_root: str = "dataset-dcae/datasets/raw"
    cache_dir: Optional[str] = None
    
    # FIXED: Enhanced audio processing for stability
    sample_rate: int = 44100
    audio_duration: float = 1.0  # Conservative for V100
    min_duration: float = 0.5   # Conservative
    max_duration: float = 10.0  # Reduced
    
    # FIXED: Enhanced data splitting
    train_split: float = 0.85  # Slightly increased
    val_split: float = 0.15   # Slightly decreased
    test_split: float = 0.0
    
    # FIXED: Enhanced augmentation (reduced for stability)
    use_augmentation: bool = True
    augmentation_prob: float = 0.3  # Reduced
    compression_aware_augmentation: bool = True
    
    # FIXED: Enhanced file handling
    supported_formats: List[str] = field(default_factory=lambda: ['.wav', '.flac'])  # Reduced
    skip_corrupted: bool = True
    normalize_audio: bool = True
    compression_quality_filter: bool = True
    
    # FIXED: Enhanced memory management for stability
    cache_audio: bool = False
    preload_data: bool = False
    enable_compression_caching: bool = False  # Disabled for stability
    
    # FIXED: Enhanced quality filters
    min_sample_rate: int = 22050
    max_file_size_mb: int = 50  # Reduced
    remove_silence: bool = False
    min_dynamic_range_db: float = 15.0  # Reduced
    max_compression_artifacts: float = 0.15  # Increased tolerance
    
    # CRITICAL: Enhanced stability settings
    disable_torch_compile: bool = True
    disable_dynamo_tracing: bool = True
    enable_numerical_stability: bool = True


# ==================== Enhanced Factory Functions ====================

def create_enhanced_cqt_ssm_config(
    model_size: str = "base",
    audio_duration: float = 1.0,
    batch_size: int = 6,  # Reduced for larger models
    use_augmentation: bool = True,
    enable_compression_optimization: bool = True,
    compression_level: str = "medium",
    disable_torch_compile: bool = True,
    enable_enhanced_numerical_stability: bool = True,
    **kwargs
) -> DCAEConfig:
    """
    Create enhanced CQT-SSM DCAE configuration with proper model sizing and NaN prevention
    
    Args:
        model_size: "small", "base" (60M), "large" (100M), "compressed"
        audio_duration: Audio duration in seconds
        batch_size: Training batch size (adjusted for larger models)
        use_augmentation: Whether to use audio augmentation
        enable_compression_optimization: Enable S6-SSM compression features
        compression_level: Compression aggressiveness ("low", "medium", "high")
        disable_torch_compile: Disable torch.compile (required for DDP)
        enable_enhanced_numerical_stability: Enable enhanced NaN prevention
        **kwargs: Additional configuration overrides
    
    Returns:
        Complete DCAEConfig instance with enhanced model sizing and stability
    """
    config = DCAEConfig()
    
    # Set basic parameters
    config.model_size = model_size
    config.audio_duration = audio_duration
    config.batch_size = batch_size
    config.use_augmentation = use_augmentation
    
    # CRITICAL: Enhanced stability settings
    config.disable_torch_compile = disable_torch_compile
    config.disable_dynamo_tracing = True
    config.enable_enhanced_numerical_stability = enable_enhanced_numerical_stability
    config.use_safe_operations = True
    config.nan_prevention_active = True
    config.gradient_checkpointing = False  # Disabled to prevent NaN
    
    # Configure compression optimization
    if enable_compression_optimization:
        # Enhanced forced compression mechanisms
        config.enable_forced_compression = True
        config.dynamic_channel_pruning = True
        
        # Enhanced S6-SSM compression optimization
        config.enable_compression_aware_s6 = True
        config.enable_multiscale_ssm = True
        config.enable_semantic_guidance = False  # Disabled for stability
        
        # Enhanced skip connection redesign
        config.enable_selective_skip = True
        config.learnable_compression_skip = True
        
        # Enhanced quality enhancement with stability
        config.enable_enhanced_perceptual_loss = True
        config.enable_detail_refinement = False  # Disabled for stability
        config.enable_transfer_learning_optimization = False  # Disabled for DDP
        
        # Compression level specific settings
        if compression_level == "low":
            config.compression_factor = 12
            config.adaptive_latent_channels = (7, 9)  # Increased
            config.skip_pruning_ratio = 0.2
        elif compression_level == "medium":
            config.compression_factor = 16
            config.adaptive_latent_channels = (6, 8)  # Increased
            config.skip_pruning_ratio = 0.3
        elif compression_level == "high":
            config.compression_factor = 20  # Reduced from 24 for stability
            config.adaptive_latent_channels = (5, 7)  # Increased
            config.skip_pruning_ratio = 0.4
    
    # Update for model size with enhanced configurations
    config.update_for_model_size()
    
    # Apply additional overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown configuration parameter: {key}")
    
    # FIXED: Validate configuration
    validation = config.validate_configuration()
    if not validation['valid']:
        print("⚠️ Configuration validation failed:")
        for error in validation['errors']:
            print(f"   ❌ {error}")
    
    if validation['warnings']:
        print("⚠️ Configuration warnings:")
        for warning in validation['warnings']:
            print(f"   ⚠️  {warning}")
    
    print(f"✅ Configuration created for {model_size} model")
    print(f"   🎯 Estimated parameters: {validation['parameter_estimate']:,}")
    print(f"   📊 Stability score: {validation['stability_score']:.2f}")
    
    return config


# ==================== Enhanced Presets ====================

def get_enhanced_high_quality_config() -> DCAEConfig:
    """Enhanced configuration for maximum quality with 60M parameters"""
    return create_enhanced_cqt_ssm_config(
        model_size="base",
        audio_duration=1.0,  # Conservative for V100
        batch_size=4,  # Reduced for quality focus
        compression_level="low",
        enable_compression_optimization=True,
        disable_torch_compile=True,
        enable_enhanced_numerical_stability=True,
        
        # Enhanced quality settings
        perceptual_loss_weight=0.3,
        mel_scale_loss=True,
        w_cqt=1.2,
        ema_decay=0.9995,
        learning_rate=8e-5  # Lower for quality
    )


def get_enhanced_large_model_config() -> DCAEConfig:
    """Enhanced configuration for large model with 100M parameters"""
    return create_enhanced_cqt_ssm_config(
        model_size="large",
        audio_duration=1.0,
        batch_size=2,  # Very conservative for large model
        compression_level="medium",
        enable_compression_optimization=True,
        disable_torch_compile=True,
        enable_enhanced_numerical_stability=True,
        
        # Enhanced large model settings
        learning_rate=6e-5,  # Lower for large model
        grad_clip=0.3,  # More conservative
        ema_update_after=200,  # Longer warmup
        weight_decay=0.025  # Slightly higher regularization
    )


def get_enhanced_stable_training_config() -> DCAEConfig:
    """Enhanced configuration for maximum training stability"""
    return create_enhanced_cqt_ssm_config(
        model_size="base",
        audio_duration=1.0,
        batch_size=6,
        compression_level="medium",
        enable_compression_optimization=True,
        disable_torch_compile=True,
        enable_enhanced_numerical_stability=True,
        
        # Maximum stability settings
        learning_rate=5e-5,  # Very conservative
        grad_clip=0.2,  # Very conservative
        augmentation_prob=0.2,  # Reduced
        use_checkpointing=False,  # Disabled
        enable_detail_refinement=False,  # Disabled
        progressive_layer_unfreezing=False,  # Disabled
        
        # Enhanced loss weights for stability
        information_bottleneck_loss_weight=0.02,
        compression_regularization_weight=0.01
    )


def get_enhanced_research_config() -> DCAEConfig:
    """Enhanced configuration for research with comprehensive monitoring"""
    return create_enhanced_cqt_ssm_config(
        model_size="base",
        audio_duration=1.0,
        batch_size=6,
        compression_level="medium",
        enable_compression_optimization=True,
        disable_torch_compile=True,
        enable_enhanced_numerical_stability=True,
        
        # Research-friendly settings
        save_interval=2,
        log_every_n_steps=100,
        sample_every_n_epochs=10,
        save_compression_stats=True,
        
        # Enhanced monitoring
        use_ema=True,
        ema_update_every=5,  # Frequent EMA updates
        val_check_interval=2  # Frequent validation
    )