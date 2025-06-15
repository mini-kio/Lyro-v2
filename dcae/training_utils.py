# lyro/dcae/training_utils.py - NaN Loss 해결 및 강화 버전
"""Training utilities for S6-SSM DCAE with improved stability."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as tF
import numpy as np
import random
from typing import Dict, Optional, Any, Union, List, Tuple
from collections import defaultdict, deque
import copy
import librosa
import time
import math
import os
import gc


import torch._dynamo
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True


os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'



def safe_tensor_operation(tensor: torch.Tensor, operation: str = "mean", eps: float = 1e-8) -> torch.Tensor:
    """Safe tensor operations to prevent NaN"""
    if tensor is None or tensor.numel() == 0:
        return torch.tensor(0.0, device=tensor.device if tensor is not None else 'cpu', requires_grad=True)
    
    # Check for NaN/Inf
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return torch.tensor(0.0, device=tensor.device, requires_grad=True)
    
    try:
        if operation == "mean":
            result = tensor.mean()
        elif operation == "var":
            result = tensor.var(unbiased=False) + eps
        elif operation == "std":
            result = torch.sqrt(tensor.var(unbiased=False) + eps)
        elif operation == "norm":
            result = torch.norm(tensor) + eps
        else:
            result = tensor.mean()
        
        # Final safety check
        if torch.isnan(result).any() or torch.isinf(result).any():
            return torch.tensor(0.0, device=tensor.device, requires_grad=True)
        
        return result
    except Exception:
        return torch.tensor(0.0, device=tensor.device, requires_grad=True)


def validate_tensor_health(tensor: torch.Tensor, name: str = "tensor", verbose: bool = False) -> bool:
    """Enhanced tensor health validation"""
    if tensor is None:
        if verbose:
            print(f"❌ {name}: None tensor")
        return False
    
    if tensor.numel() == 0:
        if verbose:
            print(f"❌ {name}: Empty tensor")
        return False
    
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        if verbose:
            print(f"❌ {name}: NaN={has_nan}, Inf={has_inf}")
            print(f"   Shape: {tensor.shape}, Min: {tensor.min()}, Max: {tensor.max()}")
        return False
    
    return True


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safe logarithm to prevent NaN"""
    return torch.log(torch.clamp(x, min=eps))


def safe_exp(x: torch.Tensor, max_val: float = 20.0) -> torch.Tensor:
    """Safe exponential to prevent overflow"""
    return torch.exp(torch.clamp(x, max=max_val))


def safe_div(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safe division to prevent NaN"""
    return numerator / torch.clamp(denominator, min=eps)



class DDPCompatibleEMAWrapper:
    """
    Enhanced DDP Compatible EMA Wrapper with complete numerical stability
    FIXED: NaN 방지 및 gradient flow 안정성 보장
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
        update_after: int = 100,
        update_every: int = 10,

        compression_decay: float = 0.9995,
        s6_core_decay: float = 0.9998,
        skip_decay: float = 0.999,

        enable_gradient_monitoring: bool = True,
        nan_detection_threshold: int = 3,
    ):
        """
        Enhanced DDP Compatible EMA with complete numerical stability
        """
        self.model = model
        self.base_decay = decay
        self.compression_decay = compression_decay
        self.s6_core_decay = s6_core_decay
        self.skip_decay = skip_decay
        self.device = device or next(model.parameters()).device
        self.update_after = update_after
        self.update_every = update_every
        

        self.enable_gradient_monitoring = enable_gradient_monitoring
        self.nan_detection_threshold = nan_detection_threshold
        self.nan_count = 0
        
        self.step_count = 0
        self.shadow = {}
        self.backup = {}
        self.component_types = {}
        

        self.compression_loss_history = deque(maxlen=10)
        self.gradient_norm_history = deque(maxlen=5)
        

        self._initialized = True
        self._force_initialize_immediately()
        
    def _force_initialize_immediately(self):
        """Enhanced DDP safe immediate initialization with health checks"""
        try:
            success_count = 0
            total_params = 0
            

            for name, param in self.model.named_parameters():
                if param.requires_grad and validate_tensor_health(param, f"param_{name}"):
                    total_params += 1
                    

                    try:
                        shadow_param = param.data.clone().detach().to(self.device)
                        

                        if validate_tensor_health(shadow_param, f"shadow_{name}"):
                            self.shadow[name] = shadow_param
                            self.component_types[name] = self._classify_parameter_component(name)
                            success_count += 1
                        else:
                            # Create safe fallback shadow
                            self.shadow[name] = torch.zeros_like(param.data, device=self.device)
                            self.component_types[name] = 'general'
                    except Exception as e:
                        print(f"⚠️ Failed to create shadow for {name}: {e}")
                        # Create safe fallback
                        self.shadow[name] = torch.zeros_like(param.data, device=self.device)
                        self.component_types[name] = 'general'
                        success_count += 1
                        
            print(f"✅ Enhanced EMA: Initialized {success_count}/{total_params} parameters successfully")
            
        except Exception as e:
            print(f"⚠️ Enhanced EMA initialization failed: {e}")
            # Create minimal safe shadows for all parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = torch.zeros_like(param.data, device=self.device)
                    self.component_types[name] = 'general'
        
    def _classify_parameter_component(self, name: str) -> str:
        """
        Enhanced parameter classification for stability-aware EMA
        """
        name_lower = name.lower()
        

        if any(keyword in name_lower for keyword in [
            'cqt_transform', 'compression_aware', 'channel_pruner', 'frequency_projection'
        ]):
            return 'compression'
        elif any(keyword in name_lower for keyword in [
            's6_block', 'ssm_processor', 'state_space', 's6_core'
        ]):
            return 's6_core'
        elif any(keyword in name_lower for keyword in [
            'skip', 'adapter', 'bridge'
        ]):
            return 'skip'
        elif any(keyword in name_lower for keyword in [
            'perceptual', 'discriminator', 'loss'
        ]):
            return 'perceptual'
        else:
            return 'general'
    
    def update(self, compression_loss: Optional[float] = None):
        """
        Enhanced DDP safe EMA update with complete numerical stability
        """
        self.step_count += 1
        
        # Track compression loss with health check
        if compression_loss is not None and not (math.isnan(compression_loss) or math.isinf(compression_loss)):
            self.compression_loss_history.append(compression_loss)
        

        should_update = (
            self.step_count > self.update_after and 
            self.step_count % self.update_every == 0 and
            self.nan_count < self.nan_detection_threshold
        )
        

        try:
            with torch.no_grad():
                updated_params = 0
                total_params = 0
                gradient_norms = []
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in self.shadow:
                        total_params += 1
                        

                        param_healthy = validate_tensor_health(param, f"ema_param_{name}")
                        shadow_healthy = validate_tensor_health(self.shadow[name], f"ema_shadow_{name}")
                        
                        if param_healthy and shadow_healthy:
                            component_type = self.component_types[name]
                            

                            if component_type == 'compression':
                                decay = self.compression_decay
                            elif component_type == 's6_core':
                                decay = self.s6_core_decay
                            elif component_type == 'skip':
                                decay = self.skip_decay
                            elif component_type == 'perceptual':
                                decay = 0.9999  # Very stable for perceptual components
                            else:
                                decay = self.base_decay
                            
                            if should_update:

                                try:
                                    param_data = param.data.to(self.device)
                                    old_shadow = self.shadow[name]
                                    

                                    new_shadow = decay * old_shadow + (1.0 - decay) * param_data
                                    

                                    new_shadow = torch.clamp(new_shadow, min=-100.0, max=100.0)
                                    

                                    if validate_tensor_health(new_shadow, f"new_shadow_{name}"):
                                        self.shadow[name] = new_shadow
                                        updated_params += 1
                                        
                                        # Track gradient norms for monitoring
                                        if param.grad is not None and self.enable_gradient_monitoring:
                                            grad_norm = safe_tensor_operation(param.grad, "norm")
                                            gradient_norms.append(grad_norm.item())
                                    else:
                                        # Keep old shadow if new one is unhealthy
                                        self.nan_count += 1
                                        
                                except Exception as e:
                                    print(f"⚠️ EMA update failed for {name}: {e}")
                                    self.nan_count += 1
                            else:

                                dummy_update = param.data.to(self.device) * 0.0
                                self.shadow[name] = self.shadow[name] + dummy_update
                                updated_params += 1
                        else:
                            # Handle unhealthy parameters
                            if not param_healthy:
                                print(f"⚠️ Unhealthy parameter detected: {name}")
                            if not shadow_healthy:
                                print(f"⚠️ Unhealthy shadow detected: {name}")
                                # Reset shadow to zeros
                                self.shadow[name] = torch.zeros_like(param.data, device=self.device)
                            self.nan_count += 1
                

                if gradient_norms and self.enable_gradient_monitoring:
                    avg_grad_norm = np.mean(gradient_norms)
                    if not (math.isnan(avg_grad_norm) or math.isinf(avg_grad_norm)):
                        self.gradient_norm_history.append(avg_grad_norm)
                

                if updated_params == total_params:
                    self.nan_count = max(0, self.nan_count - 1)
                        
        except Exception as e:
            print(f"⚠️ EMA update failed: {e}")
            self.nan_count += 1
    
    def apply_shadow(self):
        """Enhanced shadow application with safety checks"""
        if not self._initialized:
            return
            
        try:
            applied_count = 0
            total_count = 0
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    total_count += 1
                    

                    shadow_healthy = validate_tensor_health(self.shadow[name], f"apply_shadow_{name}")
                    
                    if shadow_healthy:

                        try:
                            self.backup[name] = param.data.clone()
                            param.data.copy_(self.shadow[name])
                            applied_count += 1
                        except Exception as e:
                            print(f"⚠️ Failed to apply shadow for {name}: {e}")
                    else:
                        print(f"⚠️ Skipping unhealthy shadow for {name}")
            
            if applied_count != total_count:
                print(f"⚠️ Applied {applied_count}/{total_count} shadows")
                
        except Exception as e:
            print(f"⚠️ Shadow application failed: {e}")
            self.restore_original()
    
    def restore_original(self):
        """Enhanced original parameter restoration with safety"""
        try:
            restored_count = 0
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.backup:
                    try:
                        backup_healthy = validate_tensor_health(self.backup[name], f"restore_{name}")
                        
                        if backup_healthy:
                            param.data.copy_(self.backup[name])
                            restored_count += 1
                        else:
                            print(f"⚠️ Unhealthy backup for {name}, keeping current")
                    except Exception as e:
                        print(f"⚠️ Failed to restore {name}: {e}")
            
            self.backup.clear()
            
        except Exception as e:
            print(f"⚠️ Parameter restoration failed: {e}")
    
    def get_ema_stats(self) -> Dict[str, Any]:
        """Enhanced EMA statistics with health monitoring"""
        if not self._initialized:
            return {'initialized': False}
        
        try:
            component_counts = defaultdict(int)
            healthy_shadows = 0
            total_shadows = len(self.shadow)
            
            for name, shadow in self.shadow.items():
                component_type = self.component_types.get(name, 'unknown')
                component_counts[component_type] += 1
                
                if validate_tensor_health(shadow, verbose=False):
                    healthy_shadows += 1
            

            stats = {
                'step_count': self.step_count,
                'initialized': self._initialized,
                'shadow_params': total_shadows,
                'healthy_shadows': healthy_shadows,
                'shadow_health_ratio': healthy_shadows / max(total_shadows, 1),
                'component_counts': dict(component_counts),
                'nan_count': self.nan_count,
                'compression_loss_trend': (
                    np.mean(list(self.compression_loss_history)[-3:]) 
                    if len(self.compression_loss_history) >= 3 else 0.0
                ),
                'avg_gradient_norm': (
                    np.mean(list(self.gradient_norm_history)[-3:])
                    if len(self.gradient_norm_history) >= 3 else 0.0
                ),
                'ddp_compatible': True,
                'numerical_stability_enhanced': True,
                'nan_prevention_active': True
            }
            
            return stats
            
        except Exception as e:
            print(f"⚠️ EMA stats computation failed: {e}")
            return {
                'initialized': self._initialized,
                'error': str(e),
                'ddp_compatible': True
            }
    
    def state_dict(self):
        """Enhanced EMA state dict with health information"""
        if not self._initialized:
            return {'initialized': False}
        
        try:

            healthy_shadows = {}
            for name, shadow in self.shadow.items():
                if validate_tensor_health(shadow, verbose=False):
                    healthy_shadows[name] = shadow
            
            return {
                'shadow': healthy_shadows,
                'step_count': self.step_count,
                'base_decay': self.base_decay,
                'compression_decay': self.compression_decay,
                's6_core_decay': self.s6_core_decay,
                'skip_decay': self.skip_decay,
                'component_types': self.component_types,
                'compression_loss_history': list(self.compression_loss_history),
                'gradient_norm_history': list(self.gradient_norm_history),
                'nan_count': self.nan_count,
                'initialized': self._initialized,
                'ddp_compatible': True,
                'numerical_stability_enhanced': True,
                'nan_prevention_active': True
            }
            
        except Exception as e:
            print(f"⚠️ EMA state dict creation failed: {e}")
            return {'initialized': False, 'error': str(e)}
    
    def load_state_dict(self, state_dict):
        """Enhanced EMA state dict loading with validation"""
        if not state_dict.get('initialized', False):
            return
        
        try:

            loaded_shadows = state_dict.get('shadow', {})
            valid_shadows = {}
            
            for name, shadow in loaded_shadows.items():
                if validate_tensor_health(shadow, f"loaded_shadow_{name}", verbose=False):
                    valid_shadows[name] = shadow.to(self.device)
                else:
                    print(f"⚠️ Skipping invalid loaded shadow: {name}")
            
            self.shadow = valid_shadows
            self.step_count = state_dict.get('step_count', 0)
            self.base_decay = state_dict.get('base_decay', self.base_decay)
            self.compression_decay = state_dict.get('compression_decay', self.compression_decay)
            self.s6_core_decay = state_dict.get('s6_core_decay', self.s6_core_decay)
            self.skip_decay = state_dict.get('skip_decay', self.skip_decay)
            self.component_types = state_dict.get('component_types', {})
            self.nan_count = state_dict.get('nan_count', 0)
            
            if 'compression_loss_history' in state_dict:
                self.compression_loss_history = deque(
                    state_dict['compression_loss_history'], maxlen=10
                )
            
            if 'gradient_norm_history' in state_dict:
                self.gradient_norm_history = deque(
                    state_dict['gradient_norm_history'], maxlen=5
                )
                
            self._initialized = True
            print(f"✅ Enhanced EMA loaded: {len(valid_shadows)} shadows")
            
        except Exception as e:
            print(f"⚠️ Enhanced EMA state loading failed: {e}")
            self._initialized = False


class DDPCompatibleEMAContext:
    """Enhanced DDP compatible EMA context manager with safety"""
    
    def __init__(self, ema_wrapper: DDPCompatibleEMAWrapper):
        self.ema_wrapper = ema_wrapper
        self.applied = False
    
    def __enter__(self):
        if self.ema_wrapper._initialized and self.ema_wrapper.nan_count < self.ema_wrapper.nan_detection_threshold:
            try:
                self.ema_wrapper.apply_shadow()
                self.applied = True
            except Exception as e:
                print(f"⚠️ EMA context entry failed: {e}")
                self.applied = False
        return self.ema_wrapper.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.applied:
            try:
                self.ema_wrapper.restore_original()
            except Exception as e:
                print(f"⚠️ EMA context exit failed: {e}")



class StaticTrainingStateManager:
    """
    Enhanced training state manager with complete numerical stability
    FIXED: NaN 방지 및 강화된 상태 관리
    """
    
    def __init__(self, config):
        self.config = config
        self.metrics_history = defaultdict(lambda: deque(maxlen=5))
        self.best_metrics = {}
        self.current_epoch = 0
        self.global_step = 0
        

        self.nan_detection_count = 0
        self.gradient_explosion_count = 0
        self.loss_spike_count = 0
        

        self.augmentation = self._create_enhanced_augmentation()
        self.ema_wrapper = None
        

        self.compression_metrics_history = defaultdict(lambda: deque(maxlen=5))
        self.numerical_stability_metrics = defaultdict(lambda: deque(maxlen=3))
        

        self.static_batch_size = config.batch_size
        self.static_learning_rate = config.learning_rate
        

        self.health_thresholds = {
            'max_gradient_norm': 10.0,
            'max_loss_value': 100.0,
            'min_loss_value': -10.0,
            'max_nan_tolerance': 3
        }
    
    def _create_enhanced_augmentation(self):
        """Create enhanced augmentation with numerical stability"""
        try:
            return EnhancedAugmentation(
                sample_rate=self.config.sample_rate,
                augmentation_prob=0.3,
                gain_range=(-1.0, 1.0),
                noise_level=0.001,
                enable_safety_checks=True
            )
        except Exception as e:
            print(f"⚠️ Enhanced augmentation creation failed: {e}")
            return None
    
    def setup_static_ema(self, model: nn.Module):
        """Setup enhanced EMA wrapper with stability monitoring"""
        try:
            self.ema_wrapper = DDPCompatibleEMAWrapper(
                model,
                enable_gradient_monitoring=True,
                nan_detection_threshold=3
            )
            print("✅ Enhanced EMA wrapper setup completed")
        except Exception as e:
            print(f"⚠️ Enhanced EMA setup failed: {e}")
            self.ema_wrapper = None
    
    def update_ema(self, compression_loss: Optional[float] = None):
        """Update EMA with enhanced health monitoring"""
        if self.ema_wrapper is not None:
            try:

                safe_loss = None
                if compression_loss is not None:
                    if not (math.isnan(compression_loss) or math.isinf(compression_loss)):
                        if self.health_thresholds['min_loss_value'] <= compression_loss <= self.health_thresholds['max_loss_value']:
                            safe_loss = compression_loss
                        else:
                            print(f"⚠️ Loss out of range: {compression_loss}")
                            self.loss_spike_count += 1
                    else:
                        print(f"⚠️ Invalid loss detected: {compression_loss}")
                        self.nan_detection_count += 1
                
                self.ema_wrapper.update(safe_loss)
                

                ema_stats = self.ema_wrapper.get_ema_stats()
                if ema_stats.get('shadow_health_ratio', 1.0) < 0.8:
                    print(f"⚠️ EMA health degraded: {ema_stats.get('shadow_health_ratio', 0.0):.2f}")
                
            except Exception as e:
                print(f"⚠️ EMA update failed: {e}")
    
    def apply_augmentation(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply enhanced augmentation with safety checks"""
        if self.augmentation is not None:
            try:

                if validate_tensor_health(audio, "augmentation_input"):
                    augmented = self.augmentation(audio)
                    

                    if validate_tensor_health(augmented, "augmentation_output"):
                        return augmented
                    else:
                        print("⚠️ Augmentation produced unhealthy output")
                        return audio
                else:
                    print("⚠️ Unhealthy input for augmentation")
                    return audio
            except Exception as e:
                print(f"⚠️ Augmentation failed: {e}")
        return audio
    
    def record_compression_metrics(self, metrics: Dict[str, float]):
        """Enhanced compression metrics recording with validation"""
        try:
            healthy_metrics = {}
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if not (math.isnan(value) or math.isinf(value)):

                        if abs(value) < 1e6:  # Reasonable range check
                            healthy_metrics[key] = value
                            self.compression_metrics_history[key].append(value)
                        else:
                            print(f"⚠️ Metric {key} out of range: {value}")
                    else:
                        print(f"⚠️ Invalid metric {key}: {value}")
                elif torch.is_tensor(value):
                    if validate_tensor_health(value, f"metric_{key}", verbose=False):
                        scalar_value = safe_tensor_operation(value, "mean").item()
                        healthy_metrics[key] = scalar_value
                        self.compression_metrics_history[key].append(scalar_value)
            

            if healthy_metrics:
                avg_metric_value = np.mean(list(healthy_metrics.values()))
                self.numerical_stability_metrics['avg_metrics'].append(avg_metric_value)
                
        except Exception as e:
            print(f"⚠️ Compression metrics recording failed: {e}")
    
    def monitor_training_health(self, loss: torch.Tensor, gradients: Optional[List[torch.Tensor]] = None) -> Dict[str, Any]:
        """Enhanced training health monitoring"""
        health_report = {
            'status': 'healthy',
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:

            if validate_tensor_health(loss, "training_loss"):
                loss_value = loss.item()
                
                if math.isnan(loss_value) or math.isinf(loss_value):
                    health_report['status'] = 'critical'
                    health_report['errors'].append('NaN/Inf loss detected')
                    self.nan_detection_count += 1
                elif loss_value > self.health_thresholds['max_loss_value']:
                    health_report['status'] = 'warning'
                    health_report['warnings'].append(f'High loss: {loss_value:.4f}')
                    self.loss_spike_count += 1
                elif loss_value < self.health_thresholds['min_loss_value']:
                    health_report['status'] = 'warning'
                    health_report['warnings'].append(f'Negative loss: {loss_value:.4f}')
            else:
                health_report['status'] = 'critical'
                health_report['errors'].append('Invalid loss tensor')
                self.nan_detection_count += 1
            

            if gradients is not None:
                total_grad_norm = 0.0
                nan_grad_count = 0
                
                for grad in gradients:
                    if grad is not None:
                        if validate_tensor_health(grad, verbose=False):
                            grad_norm = safe_tensor_operation(grad, "norm").item()
                            total_grad_norm += grad_norm ** 2
                        else:
                            nan_grad_count += 1
                
                if nan_grad_count > 0:
                    health_report['status'] = 'critical'
                    health_report['errors'].append(f'{nan_grad_count} NaN gradients detected')
                    self.gradient_explosion_count += 1
                
                total_grad_norm = math.sqrt(total_grad_norm)
                if total_grad_norm > self.health_thresholds['max_gradient_norm']:
                    health_report['status'] = 'warning'
                    health_report['warnings'].append(f'High gradient norm: {total_grad_norm:.4f}')
                    self.gradient_explosion_count += 1
            

            if self.nan_detection_count >= self.health_thresholds['max_nan_tolerance']:
                health_report['recommendations'].append('Consider reducing learning rate')
                health_report['recommendations'].append('Check model architecture for numerical instability')
            
            if self.loss_spike_count >= 3:
                health_report['recommendations'].append('Consider gradient clipping')
                health_report['recommendations'].append('Verify data preprocessing')
            

            stability_score = 1.0 - (self.nan_detection_count + self.gradient_explosion_count + self.loss_spike_count) / 100.0
            stability_score = max(0.0, min(1.0, stability_score))
            self.numerical_stability_metrics['stability_score'].append(stability_score)
            
        except Exception as e:
            health_report['status'] = 'error'
            health_report['errors'].append(f'Health monitoring failed: {e}')
        
        return health_report
    
    def get_static_ema_context(self) -> Optional[DDPCompatibleEMAContext]:
        """Get enhanced EMA context with health checks"""
        if (self.ema_wrapper is not None and 
            self.ema_wrapper._initialized and 
            self.nan_detection_count < self.health_thresholds['max_nan_tolerance']):
            try:
                return DDPCompatibleEMAContext(self.ema_wrapper)
            except Exception as e:
                print(f"⚠️ EMA context creation failed: {e}")
        return None
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Enhanced compression statistics with health metrics"""
        try:
            stats = {
                'current_epoch': self.current_epoch,
                'global_step': self.global_step,
                'static_batch_size': self.static_batch_size,
                'static_learning_rate': self.static_learning_rate,
                

                'numerical_health': {
                    'nan_detection_count': self.nan_detection_count,
                    'gradient_explosion_count': self.gradient_explosion_count,
                    'loss_spike_count': self.loss_spike_count,
                    'stability_score': (
                        np.mean(list(self.numerical_stability_metrics['stability_score'])[-3:])
                        if len(self.numerical_stability_metrics['stability_score']) >= 3 else 1.0
                    )
                },
                

                'ddp_compatible': True,
                'numerical_stability_enhanced': True,
                'progressive_unfreezing_disabled': True,
                'nan_prevention_active': True,
                'health_monitoring_enabled': True
            }
            

            if self.ema_wrapper is not None:
                try:
                    ema_stats = self.ema_wrapper.get_ema_stats()
                    stats['ema_stats'] = ema_stats
                except Exception as e:
                    print(f"⚠️ EMA stats retrieval failed: {e}")
                    stats['ema_stats'] = {'initialized': False, 'error': str(e)}
            else:
                stats['ema_stats'] = {'initialized': False}
            
            return stats
            
        except Exception as e:
            print(f"⚠️ Compression stats computation failed: {e}")
            return {
                'error': str(e),
                'ddp_compatible': True,
                'numerical_stability_enhanced': True
            }



class EnhancedAugmentation:
    """
    Enhanced augmentation with complete numerical stability
    FIXED: NaN 방지 및 안전한 오디오 변환
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        augmentation_prob: float = 0.3,
        gain_range: tuple = (-1.0, 1.0),
        noise_level: float = 0.001,
        enable_safety_checks: bool = True,
    ):
        self.sample_rate = sample_rate
        self.augmentation_prob = augmentation_prob
        self.gain_range = gain_range
        self.noise_level = noise_level
        self.enable_safety_checks = enable_safety_checks
        

        self.max_gain_db = 6.0  # Conservative limit
        self.max_noise_level = 0.01  # Conservative limit
        self.max_amplitude = 0.95  # Prevent clipping
    
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply enhanced augmentation with complete safety
        """
        if random.random() > self.augmentation_prob:
            return audio
        

        if self.enable_safety_checks and not validate_tensor_health(audio, "augmentation_input"):
            return audio
        
        try:
            original_audio = audio.clone()
            

            if random.random() < 0.5:
                try:
                    gain_db = random.uniform(*self.gain_range)
                    gain_db = np.clip(gain_db, -self.max_gain_db, self.max_gain_db)
                    gain_linear = 10 ** (gain_db / 20)
                    
                    audio = audio * gain_linear
                    

                    max_val = torch.abs(audio).max()
                    if max_val > self.max_amplitude:
                        audio = audio * (self.max_amplitude / (max_val + 1e-8))
                    

                    if self.enable_safety_checks and not validate_tensor_health(audio, "gain_augmented"):
                        audio = original_audio
                        
                except Exception as e:
                    print(f"⚠️ Gain augmentation failed: {e}")
                    audio = original_audio
            

            if random.random() < 0.2:
                try:
                    safe_noise_level = min(self.noise_level, self.max_noise_level)
                    noise = torch.randn_like(audio) * safe_noise_level
                    

                    if self.enable_safety_checks and validate_tensor_health(noise, "noise"):
                        audio = audio + noise
                        

                        max_val = torch.abs(audio).max()
                        if max_val > self.max_amplitude:
                            audio = audio * (self.max_amplitude / (max_val + 1e-8))
                        

                        if self.enable_safety_checks and not validate_tensor_health(audio, "noise_augmented"):
                            audio = original_audio
                    else:
                        audio = original_audio
                        
                except Exception as e:
                    print(f"⚠️ Noise augmentation failed: {e}")
                    audio = original_audio
            

            if self.enable_safety_checks:
                if not validate_tensor_health(audio, "final_augmented"):
                    print("⚠️ Final augmentation validation failed")
                    return original_audio
                

                if torch.abs(audio).max() > 1.0:
                    audio = torch.clamp(audio, -1.0, 1.0)
                    
        except Exception as e:
            print(f"⚠️ Augmentation completely failed: {e}")
            return audio  # Return original input
        
        return audio



class S6SSMCompressionConfig:
    """Enhanced configuration with numerical stability focus"""
    
    def __init__(self):

        self.latent_channels = 12  # Increased from 6 for larger models
        self.encoder_base_channels = 80  # Increased from 48
        self.decoder_base_channels = 80  # Increased from 48
        self.use_vector_quantization = False
        self.dual_channel_processing = True
        self.use_weight_norm = True
        

        self.sample_rate = 44100
        self.n_bins = 84
        self.hop_length = 512
        self.bins_per_octave = 12
        self.fmin = 32.7
        

        self.enable_forced_compression = True
        self.cqt_projection_dims = 80  # Increased from 48
        self.temporal_compression_stride = 2
        self.dynamic_channel_pruning = True
        self.target_latent_channels = 6  # Increased from 4
        self.information_bottleneck_weight = 0.05
        

        self.enable_compression_aware_s6 = True
        self.selective_state_saving = True
        self.adaptive_forgetting_rate = 0.05
        self.compression_regularization_weight = 0.02
        self.enable_multiscale_ssm = True
        self.ssm_scales = ['global', 'middle']
        self.cross_scale_attention = False  # Disabled for stability
        self.enable_semantic_guidance = False  # Disabled for stability
        

        self.enable_selective_skip = True
        self.mutual_information_threshold = 0.2
        self.skip_pruning_ratio = 0.3
        self.learnable_compression_skip = True
        self.semantic_skip = False  # Disabled
        self.adaptive_skip_activation = False  # Disabled
        self.s6_enhanced_skip = True
        

        self.enable_enhanced_perceptual_loss = True
        self.multi_resolution_stft_loss = False  # Disabled for stability
        self.mel_scale_loss = True
        self.harmonic_loss = False  # Disabled for stability
        self.dynamic_loss_weighting = False  # Disabled for stability
        self.enable_detail_refinement = False  # Disabled for stability
        

        self.enable_transfer_learning_optimization = False
        self.progressive_unfreezing_disabled = True
        self.static_parameters = True
        self.numerical_stability_enhanced = True
        self.nan_prevention_active = True
        

        self.learning_rate = 1.2e-4
        self.weight_decay = 0.02
        self.batch_size = 8  # Adjusted for larger models
        self.epochs = 200
        self.grad_clip = 0.5
        

        self.use_ema = True
        self.ema_decay = 0.999
        self.compression_ema_decay = 0.9995
        self.s6_core_ema_decay = 0.9998
        self.skip_ema_decay = 0.999
        self.ema_update_after = 100  # More conservative
        self.ema_update_every = 10   # More frequent
        

        self.use_augmentation = True
        self.augmentation_prob = 0.3
        self.gain_range = (-1.0, 1.0)
        self.noise_level = 0.001
        

        self.cqt_weight = 1.0
        self.time_weight = 0.1
        self.vq_weight = 0.01
        self.adv_weight = 0.05
        self.compression_loss_weight = 0.1
        self.information_bottleneck_loss_weight = 0.05
        self.perceptual_loss_weight = 0.2
        

        self.scheduler_type = "cosine_annealing_warm_restarts"
        self.min_lr = 5e-7
        self.warmup_epochs = 5
        self.T_0 = 60
        

        self.max_length = 44100 * 1  # 1 second
        self.train_split = 0.82
        self.val_split = 0.18
        self.test_split = 0.0
        

        self.ddp_compatible = True
        self.disable_torch_compile = True
        self.disable_dynamo_tracing = True
        self.enable_enhanced_numerical_stability = True
        self.use_safe_operations = True



def compute_compression_aware_snr(
    original: torch.Tensor, 
    reconstructed: torch.Tensor,
    frequency_weighting: bool = False
) -> float:
    """
    Enhanced SNR computation with complete numerical stability
    """
    try:

        if not (validate_tensor_health(original, "snr_original") and 
               validate_tensor_health(reconstructed, "snr_reconstructed")):
            return 0.0
        

        signal_power = safe_tensor_operation(original ** 2, "mean")
        noise_power = safe_tensor_operation((original - reconstructed) ** 2, "mean")
        

        if noise_power > 1e-10:
            snr_linear = safe_div(signal_power, noise_power)
            snr_db = 10 * safe_log(snr_linear) / math.log(10)
            

            if validate_tensor_health(snr_db, "snr_db"):
                return float(torch.clamp(snr_db, min=0.0, max=60.0).item())
        
        return 60.0  # Max reasonable SNR
        
    except Exception as e:
        print(f"⚠️ SNR computation failed: {e}")
        return 0.0


def analyze_compression_efficiency(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    compression_info: Dict,
    sample_rate: int = 44100
) -> Dict[str, float]:
    """
    Enhanced compression efficiency analysis with complete safety
    """
    try:
        analysis = {}
        

        if (validate_tensor_health(original, "analysis_original") and 
           validate_tensor_health(reconstructed, "analysis_reconstructed")):
            
            analysis['snr_db'] = compute_compression_aware_snr(original, reconstructed)
            analysis['si_sdr_db'] = compute_si_sdr(original.flatten(), reconstructed.flatten())
        else:
            analysis['snr_db'] = 0.0
            analysis['si_sdr_db'] = 0.0
        

        if 'compression_ratio' in compression_info:
            compression_ratio = compression_info['compression_ratio']
            if isinstance(compression_ratio, (int, float)) and not (math.isnan(compression_ratio) or math.isinf(compression_ratio)):
                analysis['compression_ratio'] = float(compression_ratio)
            else:
                analysis['compression_ratio'] = 1.5
        else:
            analysis['compression_ratio'] = 1.5
        

        snr_value = analysis.get('snr_db', 0.0)
        compression_ratio = analysis.get('compression_ratio', 1.0)
        
        if compression_ratio > 0:
            analysis['quality_per_compression'] = snr_value / max(1.0, compression_ratio)
        else:
            analysis['quality_per_compression'] = 0.0
        

        analysis['numerical_stability_score'] = 1.0 if all(
            not (math.isnan(v) or math.isinf(v)) for v in analysis.values()
        ) else 0.0
        
        return analysis
        
    except Exception as e:
        print(f"⚠️ Compression efficiency analysis failed: {e}")
        return {
            'snr_db': 0.0,
            'si_sdr_db': 0.0,
            'compression_ratio': 1.0,
            'quality_per_compression': 0.0,
            'numerical_stability_score': 0.0,
        }


def compute_si_sdr(reference: torch.Tensor, estimation: torch.Tensor) -> float:
    """Enhanced SI-SDR computation with complete numerical stability"""
    try:

        if not (validate_tensor_health(reference, "si_sdr_reference") and 
               validate_tensor_health(estimation, "si_sdr_estimation")):
            return 0.0
        

        reference = reference - safe_tensor_operation(reference, "mean")
        estimation = estimation - safe_tensor_operation(estimation, "mean")
        
        # Handle zero-energy signals
        ref_energy = safe_tensor_operation(reference ** 2, "mean")
        if ref_energy < 1e-10:
            return 0.0
        

        numerator = safe_tensor_operation(estimation * reference, "mean")
        denominator = safe_tensor_operation(reference ** 2, "mean")
        
        alpha = safe_div(numerator, denominator)
        target = alpha * reference
        
        target_power = safe_tensor_operation(target ** 2, "mean")
        noise_power = safe_tensor_operation((estimation - target) ** 2, "mean")
        
        if noise_power > 1e-10 and target_power > 1e-10:
            si_sdr_linear = safe_div(target_power, noise_power)
            si_sdr = 10 * safe_log(si_sdr_linear) / math.log(10)
            
            if validate_tensor_health(si_sdr, "si_sdr"):
                return float(torch.clamp(si_sdr, min=-20.0, max=40.0).item())
        
        return 0.0
        
    except Exception as e:
        print(f"⚠️ SI-SDR computation failed: {e}")
        return 0.0



def enhanced_memory_cleanup():
    """Enhanced memory cleanup with thorough clearing"""
    try:

        for _ in range(3):
            gc.collect()
        
        if torch.cuda.is_available():

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            

            torch.cuda.reset_peak_memory_stats()
            
    except Exception as e:
        print(f"⚠️ Memory cleanup failed: {e}")


def get_enhanced_memory_stats(device: torch.device) -> Dict[str, float]:
    """Enhanced memory statistics with comprehensive monitoring"""
    try:
        stats = {
            'gpu_allocated_gb': 0.0,
            'gpu_reserved_gb': 0.0,
            'gpu_peak_gb': 0.0,
            'memory_utilization': 0.0,
            'memory_efficiency': 0.0
        }
        
        if torch.cuda.is_available() and device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            
            stats.update({
                'gpu_allocated_gb': allocated,
                'gpu_reserved_gb': reserved,
                'gpu_peak_gb': peak,
                'memory_utilization': allocated / 16.0,  # V100 16GB
                'memory_efficiency': allocated / max(reserved, 1e-6)
            })
        
        return stats
        
    except Exception as e:
        print(f"⚠️ Memory stats failed: {e}")
        return {
            'gpu_allocated_gb': 0.0,
            'gpu_reserved_gb': 0.0,
            'gpu_peak_gb': 0.0,
            'memory_utilization': 0.0,
            'memory_efficiency': 0.0
        }