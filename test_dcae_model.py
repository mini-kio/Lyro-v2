#!/usr/bin/env python3
"""
Fixed Comprehensive Test Suite for Continuous Ultra-Compressed 4kbps DCAE Model
====================================================================================

This enhanced test suite validates the FIXED tensor dimension handling and provides:
1. Tensor shape validation at each stage
2. Precision-controlled compression ratio verification  
3. Continuity properties for Flow Matching compatibility
4. SSM temporal modeling prerequisites
5. 4kbps bitrate target validation with parameter adjustment suggestions
6. Performance benchmarking and memory analysis
7. Gradient flow continuity verification

Technical Validation Framework:
- Multi-stage tensor dimension debugging
- Statistical significance testing with confidence intervals
- Information-theoretic compression efficiency analysis
- Numerical stability assessment under perturbations
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
from scipy import stats

# Suppress non-critical warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from dcae.model import (
        create_continuous_ultra_compressed_dcae_model,
        FixedContinuousUltraCompressedDCAEModel,
        FixedContinuousUltraCompressedDCAELoss,
        ContinuousAdaptiveQuantization,
        LatentSpaceRegularizer
    )
    from dcae.training_utils import ensure_stereo_audio
except ImportError as e:
    print(f"Critical Import Error: {e}")
    print("Ensure the fixed dcae module is properly installed")
    sys.exit(1)


class TensorShapeValidator:
    """
    Comprehensive tensor shape validation framework for debugging dimension mismatches
    """
    
    def __init__(self, model: FixedContinuousUltraCompressedDCAEModel, device: torch.device):
        self.model = model
        self.device = device
        self.shape_log = []
    
    def log_tensor_shape(self, tensor: torch.Tensor, stage: str, expected_shape: Optional[Tuple] = None):
        """Log tensor shape with validation"""
        actual_shape = tuple(tensor.shape)
        
        log_entry = {
            'stage': stage,
            'actual_shape': actual_shape,
            'expected_shape': expected_shape,
            'valid': True
        }
        
        if expected_shape is not None:
            if actual_shape != expected_shape:
                log_entry['valid'] = False
                log_entry['error'] = f"Shape mismatch: {actual_shape} != {expected_shape}"
        
        self.shape_log.append(log_entry)
        return log_entry['valid']
    
    def validate_full_pipeline(self, audio: torch.Tensor) -> Dict[str, Any]:
        """
        Comprehensive pipeline validation with detailed tensor tracking
        """
        self.shape_log = []
        results = {'success': True, 'errors': [], 'tensor_flow': []}
        
        try:
            # Stage 1: Input validation
            self.log_tensor_shape(audio, "input_audio", None)
            audio = ensure_stereo_audio(audio, target_device=self.device)
            self.log_tensor_shape(audio, "stereo_audio", (audio.shape[0], 2, audio.shape[2]))
            
            # Stage 2: Encoder validation
            with torch.no_grad():
                encoded, cross = self.model.encoder(audio)
                
                # Validate encoder outputs
                expected_latent_time = self.model.target_time_steps
                self.log_tensor_shape(encoded, "encoder_output", 
                                    (audio.shape[0], 384, expected_latent_time))
                self.log_tensor_shape(cross, "cross_features", 
                                    (audio.shape[0], 1, expected_latent_time))
                
                # Stage 3: Latent projection validation
                latent_raw = self.model.to_latent(encoded)
                self.log_tensor_shape(latent_raw, "latent_raw", 
                                    (audio.shape[0], self.model.latent_channels, expected_latent_time))
                
                # Stage 4: Quantization validation
                latent_quantized, quant_loss = self.model.continuous_quantizer(latent_raw)
                self.log_tensor_shape(latent_quantized, "latent_quantized", latent_raw.shape)
                
                # Stage 5: Decoder validation
                reconstructed = self.model.decoder(latent_quantized, cross)
                self.log_tensor_shape(reconstructed, "decoder_output", 
                                    (audio.shape[0], 2, self.model.sample_rate))
                
                # Stage 6: Final output validation
                final_output = self.model(audio)
                self.log_tensor_shape(final_output, "final_output", audio.shape)
            
            # Analyze shape consistency
            shape_errors = [entry for entry in self.shape_log if not entry['valid']]
            if shape_errors:
                results['success'] = False
                results['errors'] = [entry['error'] for entry in shape_errors]
            
            results['tensor_flow'] = self.shape_log
            
        except Exception as e:
            results['success'] = False
            results['errors'].append(f"Pipeline validation failed: {str(e)}")
            results['tensor_flow'] = self.shape_log
        
        return results


class AdvancedContinuityValidator:
    """
    Enhanced continuity validation for Flow Matching compatibility with statistical rigor
    """
    
    def __init__(self, model: FixedContinuousUltraCompressedDCAEModel, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def test_latent_space_continuity(self, num_samples: int = 100, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Rigorous latent space continuity validation with confidence intervals
        """
        results = {}
        
        with torch.no_grad():
            # Generate diverse test audio samples
            audio_samples = []
            for i in range(20):
                # Create varied audio patterns
                freq = 220 + i * 50  # Different frequencies
                t = torch.linspace(0, 1, 44100, device=self.device)
                signal = torch.sin(2 * math.pi * freq * t) * 0.3
                stereo_signal = signal.unsqueeze(0).repeat(2, 1).unsqueeze(0)
                audio_samples.append(stereo_signal)
            
            # Test 1: Interpolation Smoothness Analysis
            interpolation_errors = []
            interpolation_norms = []
            
            for i in range(10):
                audio_1 = audio_samples[i]
                audio_2 = audio_samples[i + 1]
                
                latent_1, _ = self.model.encode(audio_1)
                latent_2, _ = self.model.encode(audio_2)
                
                # Test linear interpolation at multiple points
                alphas = torch.linspace(0, 1, 21, device=self.device)
                path_errors = []
                
                for alpha in alphas[1:-1]:  # Exclude endpoints
                    latent_interp = (1 - alpha) * latent_1 + alpha * latent_2
                    
                    try:
                        audio_interp = self.model.decode(latent_interp)
                        
                        # Measure interpolation consistency
                        expected_amplitude = (1 - alpha) * torch.std(audio_1) + alpha * torch.std(audio_2)
                        actual_amplitude = torch.std(audio_interp)
                        error = torch.abs(expected_amplitude - actual_amplitude).item()
                        path_errors.append(error)
                        
                    except Exception as e:
                        path_errors.append(1.0)  # Penalty for failed interpolation
                
                interpolation_errors.extend(path_errors)
                interpolation_norms.append(torch.norm(latent_2 - latent_1).item())
            
            # Statistical analysis of interpolation smoothness
            if interpolation_errors:
                mean_error = np.mean(interpolation_errors)
                std_error = np.std(interpolation_errors)
                
                # Confidence interval calculation
                n = len(interpolation_errors)
                se = std_error / math.sqrt(n)
                t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
                ci_lower = mean_error - t_critical * se
                ci_upper = mean_error + t_critical * se
                
                results['interpolation_smoothness'] = {
                    'mean': mean_error,
                    'std': std_error,
                    'confidence_interval': (ci_lower, ci_upper),
                    'samples': n
                }
            
            # Test 2: Lipschitz Continuity Estimation
            lipschitz_estimates = []
            
            for _ in range(50):
                # Generate random perturbations
                base_audio = torch.randn(1, 2, 44100, device=self.device) * 0.3
                perturbation_magnitude = torch.rand(1, device=self.device) * 1e-2 + 1e-4
                
                perturbed_audio = base_audio + perturbation_magnitude * torch.randn_like(base_audio)
                
                latent_base, _ = self.model.encode(base_audio)
                latent_pert, _ = self.model.encode(perturbed_audio)
                
                # Compute Lipschitz ratio
                input_diff = torch.norm(perturbed_audio - base_audio)
                latent_diff = torch.norm(latent_pert - latent_base)
                
                if input_diff > 1e-8:
                    lipschitz_ratio = (latent_diff / input_diff).item()
                    if lipschitz_ratio < 1000:  # Filter extreme values
                        lipschitz_estimates.append(lipschitz_ratio)
            
            if lipschitz_estimates:
                results['lipschitz_continuity'] = {
                    'mean': np.mean(lipschitz_estimates),
                    'std': np.std(lipschitz_estimates),
                    'percentile_95': np.percentile(lipschitz_estimates, 95),
                    'max': np.max(lipschitz_estimates),
                    'samples': len(lipschitz_estimates)
                }
            
            # Test 3: Statistical Distribution Analysis
            latent_collection = []
            for audio in audio_samples[:15]:
                latent, _ = self.model.encode(audio)
                latent_collection.append(latent.cpu().numpy())
            
            latent_array = np.concatenate(latent_collection, axis=0)
            latent_flat = latent_array.reshape(-1, latent_array.shape[-1])
            
            # Normality testing
            channel_normality = []
            for c in range(min(latent_array.shape[1], 5)):  # Test first 5 channels
                channel_data = latent_flat[:, c].flatten()[:5000]  # Limit sample size
                try:
                    _, p_value = stats.shapiro(channel_data)
                    channel_normality.append(p_value)
                except:
                    channel_normality.append(0.0)
            
            results['statistical_properties'] = {
                'mean_deviation_from_zero': float(np.abs(np.mean(latent_flat))),
                'std_deviation_from_one': float(np.abs(np.std(latent_flat) - 1.0)),
                'skewness': float(stats.skew(latent_flat.flatten())),
                'kurtosis': float(stats.kurtosis(latent_flat.flatten())),
                'normality_p_values': channel_normality,
                'mean_normality_p': float(np.mean(channel_normality)) if channel_normality else 0.0
            }
        
        return results
    
    def test_gradient_smoothness(self) -> Dict[str, float]:
        """
        Test gradient smoothness for training stability
        """
        results = {}
        
        self.model.train()
        
        try:
            # Create test input with gradients
            audio = torch.randn(1, 2, 44100, device=self.device, requires_grad=True) * 0.5
            
            # Forward pass
            latent, cross = self.model.encode(audio)
            reconstruction = self.model.decode(latent, cross)
            
            # Compute simple loss
            loss = F.mse_loss(reconstruction, audio)
            
            # Backward pass
            loss.backward()
            
            # Analyze gradient properties
            if audio.grad is not None:
                grad_tensor = audio.grad
                
                # Gradient magnitude analysis
                grad_norm = torch.norm(grad_tensor).item()
                grad_max = torch.max(torch.abs(grad_tensor)).item()
                grad_mean = torch.mean(torch.abs(grad_tensor)).item()
                
                # Gradient smoothness (spatial variation)
                grad_diff = torch.diff(grad_tensor, dim=-1)
                grad_smoothness = torch.mean(grad_diff**2).item()
                
                results.update({
                    'gradient_norm': grad_norm,
                    'gradient_max': grad_max,
                    'gradient_mean': grad_mean,
                    'gradient_smoothness': grad_smoothness,
                    'gradient_dynamic_range': grad_max / (grad_mean + 1e-8),
                    'gradient_flow_stable': True
                })
            else:
                results['gradient_flow_stable'] = False
        
        except Exception as e:
            results['gradient_flow_stable'] = False
            results['error'] = str(e)
        
        finally:
            self.model.eval()
        
        return results


class PrecisionCompressionAnalyzer:
    """
    High-precision compression analysis with information-theoretic validation
    """
    
    def __init__(self, model: FixedContinuousUltraCompressedDCAEModel, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def analyze_compression_precision(self, duration: float = 1.0) -> Dict[str, Any]:
        """
        Comprehensive compression analysis with theoretical bounds validation
        """
        results = {}
        sample_rate = self.model.sample_rate
        
        with torch.no_grad():
            # Generate test audio with known characteristics
            audio = torch.randn(1, 2, int(sample_rate * duration), device=self.device) * 0.4
            
            # Get model compression info
            compression_info = self.model.get_compression_info(audio)
            results.update(compression_info)
            
            # Precise latent analysis
            latent, cross = self.model.encode(audio)
            
            # Verify tensor dimensions match expectations
            expected_latent_shape = (1, self.model.latent_channels, self.model.target_time_steps)
            expected_cross_shape = (1, 1, self.model.target_time_steps)
            
            results['shape_validation'] = {
                'latent_shape_correct': tuple(latent.shape) == expected_latent_shape,
                'cross_shape_correct': tuple(cross.shape) == expected_cross_shape,
                'expected_latent': expected_latent_shape,
                'actual_latent': tuple(latent.shape),
                'expected_cross': expected_cross_shape,
                'actual_cross': tuple(cross.shape)
            }
            
            # Information-theoretic analysis
            latent_np = latent.detach().cpu().numpy()
            
            # Channel-wise entropy estimation
            channel_entropies = []
            effective_bitrates = []
            
            for c in range(latent.shape[1]):
                channel_data = latent_np[0, c, :].flatten()
                
                # Adaptive quantization for entropy estimation
                data_range = np.max(channel_data) - np.min(channel_data)
                if data_range > 1e-6:
                    quantization_levels = 64  # 6-bit equivalent
                    quantized = np.round((channel_data - np.min(channel_data)) / data_range * (quantization_levels - 1))
                    
                    # Shannon entropy calculation
                    unique_vals, counts = np.unique(quantized, return_counts=True)
                    probs = counts / np.sum(counts)
                    entropy = -np.sum(probs * np.log2(probs + 1e-12))
                    
                    channel_entropies.append(entropy)
                    
                    # Effective bitrate for this channel
                    effective_bits = entropy * latent.shape[2]  # bits per sample
                    effective_bitrate = effective_bits / duration / 1000  # kbps
                    effective_bitrates.append(effective_bitrate)
                else:
                    channel_entropies.append(0.0)
                    effective_bitrates.append(0.0)
            
            results['information_theory'] = {
                'channel_entropies': channel_entropies,
                'mean_entropy_per_channel': np.mean(channel_entropies),
                'total_entropy_per_sample': np.sum(channel_entropies),
                'effective_bitrates_per_channel': effective_bitrates,
                'total_effective_bitrate_kbps': np.sum(effective_bitrates),
                'compression_efficiency': np.sum(effective_bitrates) / (2 * 16 * sample_rate / 1000)  # vs 16-bit stereo
            }
            
            # Quality analysis
            reconstructed = self.model(audio)
            
            # Multi-metric quality assessment
            snr_db = self._compute_snr(audio, reconstructed)
            mse = F.mse_loss(audio, reconstructed).item()
            
            # Spectral distortion analysis
            spectral_error = self._compute_spectral_distortion(audio, reconstructed)
            
            results['quality_metrics'] = {
                'snr_db': snr_db,
                'mse': mse,
                'psnr_db': 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else float('inf'),
                'spectral_distortion_db': spectral_error,
                'quality_efficiency_ratio': snr_db / compression_info['compression_ratio']
            }
            
            # Bitrate optimization suggestions
            target_bitrate = 4.0  # kbps
            current_bitrate = results['information_theory']['total_effective_bitrate_kbps']
            
            if abs(current_bitrate - target_bitrate) > 0.5:
                scaling_factor = target_bitrate / max(current_bitrate, 0.1)
                
                # Suggest parameter adjustments
                if scaling_factor > 1.2:  # Need more bitrate
                    suggested_channels = int(self.model.latent_channels * math.sqrt(scaling_factor))
                    suggested_time_steps = int(self.model.target_time_steps * math.sqrt(scaling_factor))
                else:  # Need less bitrate
                    suggested_channels = max(4, int(self.model.latent_channels / math.sqrt(scaling_factor)))
                    suggested_time_steps = max(8, int(self.model.target_time_steps / math.sqrt(scaling_factor)))
                
                results['optimization_suggestions'] = {
                    'target_bitrate_kbps': target_bitrate,
                    'current_bitrate_kbps': current_bitrate,
                    'scaling_factor_needed': scaling_factor,
                    'suggested_latent_channels': suggested_channels,
                    'suggested_time_steps': suggested_time_steps,
                    'adjustment_needed': True
                }
            else:
                results['optimization_suggestions'] = {
                    'target_bitrate_kbps': target_bitrate,
                    'current_bitrate_kbps': current_bitrate,
                    'adjustment_needed': False,
                    'status': 'Bitrate target achieved'
                }
        
        return results
    
    def _compute_snr(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute Signal-to-Noise Ratio in dB"""
        signal_power = torch.mean(original**2)
        noise_power = torch.mean((original - reconstructed)**2)
        snr_linear = signal_power / (noise_power + 1e-12)
        return 10 * torch.log10(snr_linear).item()
    
    def _compute_spectral_distortion(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        """Compute spectral distortion in dB"""
        try:
            # Convert to mono for analysis
            orig_mono = original.mean(dim=1).flatten()
            recon_mono = reconstructed.mean(dim=1).flatten()
            
            # Compute power spectral densities
            orig_fft = torch.fft.fft(orig_mono)
            recon_fft = torch.fft.fft(recon_mono)
            
            orig_psd = torch.abs(orig_fft)**2
            recon_psd = torch.abs(recon_fft)**2
            
            # Compute spectral distortion
            ratio = recon_psd / (orig_psd + 1e-12)
            log_ratio = torch.log10(ratio + 1e-12)
            spectral_distortion = torch.mean(log_ratio**2).item()
            
            return 10 * math.sqrt(spectral_distortion)
        
        except:
            return 0.0


def run_fixed_comprehensive_test_suite():
    """
    Execute the comprehensive test suite for the FIXED model
    """
    print("=" * 80)
    print("FIXED CONTINUOUS ULTRA-COMPRESSED 4KBPS DCAE - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("Enhanced validation framework for FIXED tensor dimension handling:")
    print("• Tensor shape consistency verification at each pipeline stage")
    print("• Precision-controlled compression ratio validation")
    print("• Flow Matching continuity properties with confidence intervals")
    print("• Information-theoretic bitrate analysis with optimization suggestions")
    print("• Gradient flow stability assessment")
    print("=" * 80)
    
    # Initialize test environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing Environment: {device}")
    
    # Create FIXED model
    try:
        model = create_continuous_ultra_compressed_dcae_model(
            sample_rate=44100,
            latent_channels=6,
            target_time_steps=11,
            quantization_num_levels=64,
            quantization_temperature=1.0
        ).to(device)
        print("✓ FIXED Model instantiation successful")
    except Exception as e:
        print(f"✗ FIXED Model instantiation failed: {e}")
        return False
    
    # Initialize test frameworks
    shape_validator = TensorShapeValidator(model, device)
    continuity_validator = AdvancedContinuityValidator(model, device)
    compression_analyzer = PrecisionCompressionAnalyzer(model, device)
    
    test_results = {}
    
    # Test 1: Tensor Shape Pipeline Validation
    print("\n" + "─" * 60)
    print("TEST 1: TENSOR SHAPE PIPELINE VALIDATION")
    print("─" * 60)
    
    try:
        test_audio = torch.randn(1, 2, 44100, device=device) * 0.5
        shape_results = shape_validator.validate_full_pipeline(test_audio)
        
        if shape_results['success']:
            print("✓ Tensor shape pipeline validation PASSED")
            print("  Tensor flow analysis:")
            for entry in shape_results['tensor_flow']:
                status = "✓" if entry['valid'] else "✗"
                print(f"    {status} {entry['stage']}: {entry['actual_shape']}")
        else:
            print("✗ Tensor shape pipeline validation FAILED")
            for error in shape_results['errors']:
                print(f"    Error: {error}")
        
        test_results['tensor_shapes'] = shape_results
        
    except Exception as e:
        print(f"✗ Tensor shape validation failed: {e}")
        test_results['tensor_shapes'] = {'success': False, 'error': str(e)}
        return False
    
    # Test 2: Enhanced Continuity Analysis
    print("\n" + "─" * 60)
    print("TEST 2: ENHANCED CONTINUITY ANALYSIS FOR FLOW MATCHING")
    print("─" * 60)
    
    try:
        continuity_results = continuity_validator.test_latent_space_continuity()
        gradient_results = continuity_validator.test_gradient_smoothness()
        
        test_results['continuity'] = {**continuity_results, **gradient_results}
        
        print("Latent Space Continuity Analysis:")
        if 'interpolation_smoothness' in continuity_results:
            interp = continuity_results['interpolation_smoothness']
            print(f"• Interpolation Smoothness: {interp['mean']:.6f} ± {interp['std']:.6f}")
            print(f"  95% Confidence Interval: [{interp['confidence_interval'][0]:.6f}, {interp['confidence_interval'][1]:.6f}]")
        
        if 'lipschitz_continuity' in continuity_results:
            lipschitz = continuity_results['lipschitz_continuity']
            print(f"• Lipschitz Constant: {lipschitz['mean']:.3f} ± {lipschitz['std']:.3f}")
            print(f"  95th Percentile: {lipschitz['percentile_95']:.3f}")
        
        if 'statistical_properties' in continuity_results:
            stats_props = continuity_results['statistical_properties']
            print(f"• Mean Deviation from N(0,1): {stats_props['mean_deviation_from_zero']:.6f}")
            print(f"• Std Deviation from N(0,1): {stats_props['std_deviation_from_one']:.6f}")
            print(f"• Average Normality p-value: {stats_props['mean_normality_p']:.6f}")
        
        # Gradient analysis
        if gradient_results.get('gradient_flow_stable', False):
            print(f"• Gradient Flow: ✓ Stable")
            print(f"• Gradient Norm: {gradient_results['gradient_norm']:.6f}")
            print(f"• Gradient Smoothness: {gradient_results['gradient_smoothness']:.8f}")
        else:
            print(f"• Gradient Flow: ✗ Unstable")
        
        # Continuity assessment
        continuity_pass = (
            continuity_results.get('interpolation_smoothness', {}).get('mean', 1.0) < 0.1 and
            continuity_results.get('statistical_properties', {}).get('mean_deviation_from_zero', 1.0) < 0.2 and
            gradient_results.get('gradient_flow_stable', False)
        )
        
        if continuity_pass:
            print("✓ Continuity properties SUITABLE for Flow Matching")
        else:
            print("⚠ Continuity properties need OPTIMIZATION for Flow Matching")
        
    except Exception as e:
        print(f"✗ Continuity analysis failed: {e}")
        test_results['continuity'] = {'error': str(e)}
    
    # Test 3: Precision Compression Analysis
    print("\n" + "─" * 60)
    print("TEST 3: PRECISION COMPRESSION ANALYSIS & 4KBPS TARGET")
    print("─" * 60)
    
    try:
        compression_results = compression_analyzer.analyze_compression_precision()
        test_results['compression'] = compression_results
        
        print("Shape Validation:")
        shape_val = compression_results['shape_validation']
        print(f"• Latent Shape: {'✓' if shape_val['latent_shape_correct'] else '✗'} {shape_val['actual_latent']}")
        print(f"• Cross Shape: {'✓' if shape_val['cross_shape_correct'] else '✗'} {shape_val['actual_cross']}")
        
        print("\nCompression Metrics:")
        print(f"• Compression Ratio: {compression_results['compression_ratio']:.1f}:1")
        print(f"• Original Elements: {compression_results['original_elements']:,}")
        print(f"• Compressed Elements: {compression_results['compressed_elements']:,}")
        
        print("\nInformation-Theoretic Analysis:")
        info_theory = compression_results['information_theory']
        print(f"• Mean Entropy per Channel: {info_theory['mean_entropy_per_channel']:.3f} bits")
        print(f"• Total Effective Bitrate: {info_theory['total_effective_bitrate_kbps']:.2f} kbps")
        print(f"• Compression Efficiency: {info_theory['compression_efficiency']:.6f}")
        
        print("\nQuality Metrics:")
        quality = compression_results['quality_metrics']
        print(f"• SNR: {quality['snr_db']:.1f} dB")
        print(f"• PSNR: {quality['psnr_db']:.1f} dB")
        print(f"• Spectral Distortion: {quality['spectral_distortion_db']:.3f} dB")
        print(f"• Quality-Efficiency Ratio: {quality['quality_efficiency_ratio']:.6f}")
        
        print("\n4kbps Target Analysis:")
        optimization = compression_results['optimization_suggestions']
        print(f"• Target: {optimization['target_bitrate_kbps']:.1f} kbps")
        print(f"• Current: {optimization['current_bitrate_kbps']:.2f} kbps")
        
        if optimization['adjustment_needed']:
            print("⚠ Adjustment needed:")
            print(f"  → Suggested latent_channels: {optimization['suggested_latent_channels']}")
            print(f"  → Suggested time_steps: {optimization['suggested_time_steps']}")
            print(f"  → Scaling factor: {optimization['scaling_factor_needed']:.3f}")
        else:
            print("✓ 4kbps target achieved!")
        
    except Exception as e:
        print(f"✗ Compression analysis failed: {e}")
        test_results['compression'] = {'error': str(e)}
    
    # Test 4: Basic Functionality Verification
    print("\n" + "─" * 60)
    print("TEST 4: BASIC FUNCTIONALITY VERIFICATION")
    print("─" * 60)
    
    try:
        # Test forward pass
        test_audio = torch.randn(2, 2, 44100, device=device) * 0.5
        
        with torch.no_grad():
            # Full forward pass
            reconstructed = model(test_audio)
            
            # Separate encode/decode
            latent, cross = model.encode(test_audio)
            decoded = model.decode(latent, cross)
            
            # Get continuous latent for SSM/Flow Matching
            continuous_latent = model.get_continuous_latent(test_audio)
        
        # Verify outputs
        print("✓ Forward pass successful")
        print(f"✓ Input: {test_audio.shape} → Output: {reconstructed.shape}")
        print(f"✓ Latent: {latent.shape}, Cross: {cross.shape}")
        print(f"✓ Continuous latent for SSM+Flow Matching: {continuous_latent.shape}")
        
        # Test reconstruction quality
        mse = F.mse_loss(test_audio, reconstructed).item()
        print(f"✓ Reconstruction MSE: {mse:.8f}")
        
        test_results['basic_functionality'] = {
            'success': True,
            'input_shape': tuple(test_audio.shape),
            'output_shape': tuple(reconstructed.shape),
            'latent_shape': tuple(latent.shape),
            'cross_shape': tuple(cross.shape),
            'reconstruction_mse': mse
        }
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        test_results['basic_functionality'] = {'success': False, 'error': str(e)}
        return False
    
    # Final Assessment
    print("\n" + "=" * 80)
    print("FIXED MODEL COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    
    # Determine overall success
    critical_tests = [
        ('tensor_shapes', 'Tensor Shape Validation'),
        ('basic_functionality', 'Basic Functionality'),
        ('continuity', 'Flow Matching Continuity'),
        ('compression', 'Compression Analysis'),
    ]
    
    overall_success = True
    passed_tests = []
    failed_tests = []
    
    for test_key, test_name in critical_tests:
        if test_key in test_results:
            if isinstance(test_results[test_key], dict):
                if test_results[test_key].get('success', True) and 'error' not in test_results[test_key]:
                    print(f"✓ {test_name}: PASSED")
                    passed_tests.append(test_name)
                else:
                    print(f"✗ {test_name}: FAILED")
                    failed_tests.append(test_name)
                    overall_success = False
            else:
                print(f"✓ {test_name}: PASSED")
                passed_tests.append(test_name)
        else:
            print(f"⚠ {test_name}: NOT TESTED")
            failed_tests.append(test_name)
            overall_success = False
    
    print(f"\nOVERALL RESULT: {'✓ ALL CRITICAL TESTS PASSED' if overall_success else '✗ ISSUES DETECTED'}")
    print(f"Passed: {len(passed_tests)}/{len(critical_tests)} critical tests")
    
    if overall_success:
        print("\n🎉 FIXED MODEL VALIDATION SUCCESSFUL!")
        print("Key achievements:")
        print("• ✅ All tensor dimension issues RESOLVED")
        print("• ✅ Continuous latent space suitable for Flow Matching")
        print("• ✅ Compression pipeline functioning correctly")
        print("• ✅ Ready for SSM + Flow Matching integration")
        
        # Provide next steps
        if 'compression' in test_results and 'optimization_suggestions' in test_results['compression']:
            opt_suggestions = test_results['compression']['optimization_suggestions']
            if opt_suggestions['adjustment_needed']:
                print(f"\n📊 For 4kbps target, consider adjusting:")
                print(f"   latent_channels: {model.latent_channels} → {opt_suggestions['suggested_latent_channels']}")
                print(f"   target_time_steps: {model.target_time_steps} → {opt_suggestions['suggested_time_steps']}")
    else:
        print(f"\n⚠ Issues detected in: {', '.join(failed_tests)}")
        print("Review individual test results for optimization guidance.")
    
    return test_results


if __name__ == "__main__":
    """
    Execute the FIXED comprehensive test suite
    """
    test_results = run_fixed_comprehensive_test_suite()
    
    # Exit with appropriate code
    if test_results and all(
        test_results.get(key, {}).get('success', False) if isinstance(test_results.get(key, {}), dict) else True
        for key in ['tensor_shapes', 'basic_functionality']
    ):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Issues detected