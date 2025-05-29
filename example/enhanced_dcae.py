# lyro/examples/enhanced_dcae_usage.py
"""
Enhanced DCAE Usage Examples
Demonstrates how to use the improved DCAE with all enhancements
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple

# Import enhanced modules
import sys
sys.path.append('..')

from dcae.model import LyroMusicDCAE, create_enhanced_lyro_dcae
from dcae.training_utils import (
    EMAWrapper, 
    MixScaleAugmentation, 
    compute_snr, 
    compute_si_sdr,
    analyze_frequency_response
)


class EnhancedDCAEInference:
    """Enhanced DCAE inference with all improvements"""
    
    def __init__(
        self, 
        model_path: str,
        use_ema: bool = True,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_ema = use_ema
        
        # Load enhanced model
        self.model = self._load_enhanced_model(model_path)
        print(f"Enhanced DCAE loaded on {self.device}")
        
    def _load_enhanced_model(self, model_path: str) -> LyroMusicDCAE:
        """Load enhanced model with EMA if available"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Check for EMA model first
        ema_path = Path(model_path).parent / 'best_model_ema.pt'
        if self.use_ema and ema_path.exists():
            print("Loading EMA model for inference")
            ema_checkpoint = torch.load(ema_path, map_location=self.device)
            model_config = ema_checkpoint['config']
            
            # Create model
            model = create_enhanced_lyro_dcae(
                model_size="base",
                use_weight_norm=model_config.get('use_weight_norm', True),
                **model_config
            )
            
            # Load EMA weights
            model.load_state_dict(ema_checkpoint['model_state_dict'])
            
        else:
            print("Loading regular model")
            model_config = checkpoint.get('config', {})
            
            # Create model  
            model = create_enhanced_lyro_dcae(
                model_size="base",
                use_weight_norm=model_config.get('use_weight_norm', True),
                **model_config
            )
            
            # Load regular weights
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def encode_audio(self, audio_path: str) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode audio to latent with skip features
        
        Args:
            audio_path: Path to audio file
        Returns:
            latent: (1, 8, T//32) latent representation
            skip_features: List of skip connection features
        """
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            audio = resampler(audio)
        
        # Ensure stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]
        
        # Add batch dimension and move to device
        audio = audio.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            latent, skip_features = self.model.encode(audio)
        
        return latent, skip_features
    
    def decode_latent(
        self, 
        latent: torch.Tensor, 
        skip_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Decode latent to audio with skip connections
        
        Args:
            latent: (B, 8, T//32) latent representation
            skip_features: List of skip connection features
        Returns:
            audio: (B, 2, T) reconstructed audio
        """
        with torch.no_grad():
            audio = self.model.decode(latent, skip_features)
        
        return audio
    
    def reconstruct_audio(self, audio_path: str, output_path: str = None) -> Dict:
        """
        Complete reconstruction pipeline with quality metrics
        
        Args:
            audio_path: Input audio path
            output_path: Output audio path (optional)
        Returns:
            metrics: Dictionary of quality metrics
        """
        print(f"Reconstructing: {audio_path}")
        
        # Load original audio
        original_audio, sr = torchaudio.load(audio_path)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            original_audio = resampler(original_audio)
        
        if original_audio.shape[0] == 1:
            original_audio = original_audio.repeat(2, 1)
        elif original_audio.shape[0] > 2:
            original_audio = original_audio[:2]
        
        # Timing
        start_time = time.time()
        
        # Encode
        encode_start = time.time()
        latent, skip_features = self.encode_audio(audio_path)
        encode_time = time.time() - encode_start
        
        # Decode
        decode_start = time.time()
        reconstructed = self.decode_latent(latent, skip_features)
        decode_time = time.time() - decode_start
        
        total_time = time.time() - start_time
        
        # Move to CPU for metrics
        original_cpu = original_audio
        reconstructed_cpu = reconstructed[0].cpu()
        
        # Ensure same length
        min_len = min(original_cpu.shape[1], reconstructed_cpu.shape[1])
        original_cpu = original_cpu[:, :min_len]
        reconstructed_cpu = reconstructed_cpu[:, :min_len]
        
        # Compute quality metrics
        snr = compute_snr(original_cpu, reconstructed_cpu)
        si_sdr = compute_si_sdr(original_cpu.flatten(), reconstructed_cpu.flatten())
        freq_analysis = analyze_frequency_response(original_cpu, reconstructed_cpu)
        
        # Calculate RTF (Real-Time Factor)
        audio_duration = original_cpu.shape[1] / 44100
        rtf = total_time / audio_duration
        
        metrics = {
            'snr_db': snr,
            'si_sdr_db': si_sdr,
            'encode_time': encode_time,
            'decode_time': decode_time,
            'total_time': total_time,
            'rtf': rtf,
            'compression_ratio': self.model.get_compression_ratio(),
            'latent_shape': latent.shape,
            **freq_analysis
        }
        
        # Save output if requested
        if output_path:
            torchaudio.save(output_path, reconstructed_cpu, 44100)
            print(f"Saved to: {output_path}")
        
        # Print metrics
        print(f"Quality Metrics:")
        print(f"  SNR: {snr:.2f} dB")
        print(f"  SI-SDR: {si_sdr:.2f} dB")
        print(f"  RTF: {rtf:.3f} (< 1.0 is real-time)")
        print(f"  Encode: {encode_time*1000:.1f} ms")
        print(f"  Decode: {decode_time*1000:.1f} ms")
        print(f"  Total: {total_time*1000:.1f} ms")
        
        return metrics
    
    def batch_reconstruct(
        self, 
        audio_paths: List[str], 
        output_dir: str = None
    ) -> Dict:
        """Batch reconstruction with statistics"""
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        all_metrics = []
        
        for i, audio_path in enumerate(audio_paths):
            output_path = None
            if output_dir:
                output_path = Path(output_dir) / f"reconstructed_{i:03d}.wav"
            
            try:
                metrics = self.reconstruct_audio(audio_path, str(output_path))
                metrics['file'] = audio_path
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
        
        # Aggregate statistics
        if all_metrics:
            avg_metrics = {}
            for key in ['snr_db', 'si_sdr_db', 'rtf', 'encode_time', 'decode_time']:
                values = [m[key] for m in all_metrics if key in m]
                if values:
                    avg_metrics[f'avg_{key}'] = np.mean(values)
                    avg_metrics[f'std_{key}'] = np.std(values)
                    avg_metrics[f'min_{key}'] = np.min(values)
                    avg_metrics[f'max_{key}'] = np.max(values)
            
            print(f"\nBatch Statistics ({len(all_metrics)} files):")
            print(f"  Avg SNR: {avg_metrics.get('avg_snr_db', 0):.2f} ± {avg_metrics.get('std_snr_db', 0):.2f} dB")
            print(f"  Avg SI-SDR: {avg_metrics.get('avg_si_sdr_db', 0):.2f} ± {avg_metrics.get('std_si_sdr_db', 0):.2f} dB")
            print(f"  Avg RTF: {avg_metrics.get('avg_rtf', 0):.3f} ± {avg_metrics.get('std_rtf', 0):.3f}")
        
        return {
            'individual_metrics': all_metrics,
            'aggregate_metrics': avg_metrics if all_metrics else {}
        }
    
    def compare_with_baseline(
        self, 
        audio_path: str, 
        baseline_model_path: str
    ) -> Dict:
        """Compare enhanced model with baseline"""
        print(f"Comparing models on: {audio_path}")
        
        # Enhanced model metrics
        enhanced_metrics = self.reconstruct_audio(audio_path)
        
        # Load baseline model (without enhancements)
        baseline_model = create_enhanced_lyro_dcae(
            model_size="base", 
            use_weight_norm=False  # Disable enhancements for baseline
        )
        
        baseline_checkpoint = torch.load(baseline_model_path, map_location=self.device)
        baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
        baseline_model.to(self.device)
        baseline_model.eval()
        
        # Baseline reconstruction
        with torch.no_grad():
            audio, _ = torchaudio.load(audio_path)
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            audio = audio.unsqueeze(0).to(self.device)
            
            baseline_latent, baseline_skip = baseline_model.encode(audio)
            baseline_recon = baseline_model.decode(baseline_latent, baseline_skip)
        
        # Baseline metrics
        original_audio, _ = torchaudio.load(audio_path)
        if original_audio.shape[0] == 1:
            original_audio = original_audio.repeat(2, 1)
        
        baseline_snr = compute_snr(original_audio, baseline_recon[0].cpu())
        baseline_si_sdr = compute_si_sdr(
            original_audio.flatten(), 
            baseline_recon[0].cpu().flatten()
        )
        
        # Comparison
        comparison = {
            'enhanced': {
                'snr_db': enhanced_metrics['snr_db'],
                'si_sdr_db': enhanced_metrics['si_sdr_db'],
                'rtf': enhanced_metrics['rtf']
            },
            'baseline': {
                'snr_db': baseline_snr,
                'si_sdr_db': baseline_si_sdr,
                'rtf': enhanced_metrics['rtf']  # Assume similar for comparison
            },
            'improvement': {
                'snr_db': enhanced_metrics['snr_db'] - baseline_snr,
                'si_sdr_db': enhanced_metrics['si_sdr_db'] - baseline_si_sdr
            }
        }
        
        print(f"Comparison Results:")
        print(f"  Enhanced SNR: {comparison['enhanced']['snr_db']:.2f} dB")
        print(f"  Baseline SNR: {comparison['baseline']['snr_db']:.2f} dB")
        print(f"  Improvement: {comparison['improvement']['snr_db']:+.2f} dB")
        print(f"  Enhanced SI-SDR: {comparison['enhanced']['si_sdr_db']:.2f} dB")
        print(f"  Baseline SI-SDR: {comparison['baseline']['si_sdr_db']:.2f} dB") 
        print(f"  Improvement: {comparison['improvement']['si_sdr_db']:+.2f} dB")
        
        return comparison


def demo_augmentation():
    """Demonstrate augmentation effects"""
    print("Demo: Enhanced Augmentation Effects")
    
    # Create augmentation pipeline
    augmentation = MixScaleAugmentation(
        sample_rate=44100,
        augmentation_prob=1.0,  # Always apply for demo
        gain_range=(-3.0, 3.0),
        pitch_range=(-0.5, 0.5),
        reverb_prob=0.5,
        eq_prob=0.5
    )
    
    # Generate test signal
    duration = 3.0  # seconds
    t = torch.linspace(0, duration, int(44100 * duration))
    
    # Create a simple test signal (sine waves)
    freq1, freq2 = 440, 880  # A4 and A5
    signal = 0.3 * (torch.sin(2 * np.pi * freq1 * t) + 0.5 * torch.sin(2 * np.pi * freq2 * t))
    stereo_signal = torch.stack([signal, signal * 0.8])  # Slight stereo difference
    
    print(f"Original signal shape: {stereo_signal.shape}")
    
    # Apply augmentation multiple times
    augmented_signals = []
    for i in range(5):
        aug_signal = augmentation(stereo_signal.clone())
        augmented_signals.append(aug_signal)
        
        # Basic analysis
        rms_original = torch.sqrt(torch.mean(stereo_signal**2))
        rms_augmented = torch.sqrt(torch.mean(aug_signal**2))
        
        print(f"Augmentation {i+1}:")
        print(f"  RMS change: {20*torch.log10(rms_augmented/rms_original):.2f} dB")
        print(f"  Shape: {aug_signal.shape}")
    
    return augmented_signals


def benchmark_enhancements():
    """Benchmark different enhancement combinations"""
    print("Benchmarking Enhancement Combinations")
    
    configs = [
        {"name": "Baseline", "use_weight_norm": False, "extended_stft": False},
        {"name": "Weight Norm Only", "use_weight_norm": True, "extended_stft": False},
        {"name": "Extended STFT Only", "use_weight_norm": False, "extended_stft": True},
        {"name": "All Enhancements", "use_weight_norm": True, "extended_stft": True},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        
        # Create model with specific configuration
        model = create_enhanced_lyro_dcae(
            model_size="base",
            use_weight_norm=config["use_weight_norm"]
        )
        
        # If we had trained models, we would load them here
        # For demo purposes, we'll just measure model complexity
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count weight normalized layers
        weight_norm_layers = 0
        for module in model.modules():
            if hasattr(module, 'weight_g'):  # Weight norm adds weight_g attribute
                weight_norm_layers += 1
        
        results[config['name']] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'weight_norm_layers': weight_norm_layers
        }
        
        print(f"  Parameters: {total_params:,}")
        print(f"  Weight Norm Layers: {weight_norm_layers}")
    
    return results


def main():
    """Main demo function"""
    print("Enhanced DCAE Demo & Usage Examples")
    print("="*50)
    
    # 1. Augmentation Demo
    print("\n1. Augmentation Effects Demo")
    demo_augmentation()
    
    # 2. Enhancement Benchmarking
    print("\n2. Enhancement Combinations Benchmark")
    benchmark_results = benchmark_enhancements()
    
    # 3. Inference Example (if model exists)
    model_path = "dcae/checkpoints/best_model.pt"
    if Path(model_path).exists():
        print(f"\n3. Enhanced Inference Demo")
        
        # Create inference object
        inference = EnhancedDCAEInference(model_path, use_ema=True)
        
        # Example audio file (replace with actual path)
        example_audio = "dataset/audio/full/example.wav"
        if Path(example_audio).exists():
            print(f"\nProcessing: {example_audio}")
            metrics = inference.reconstruct_audio(
                example_audio, 
                "outputs/enhanced_reconstruction.wav"
            )
            
            print(f"\nDetailed Results:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"Example audio not found: {example_audio}")
            print("To test inference, provide a valid audio file path")
    
    else:
        print(f"\n3. Model not found: {model_path}")
        print("Train the enhanced model first using:")
        print("python dcae/train_dcae.py --train_metadata dataset/metadata/train_metadata.jsonl --val_metadata dataset/metadata/val_metadata.jsonl")
    
    print(f"\n{'='*50}")
    print("Enhanced DCAE Demo Complete!")
    
    # Print usage instructions
    print(f"\nUsage Instructions:")
    print(f"1. Train enhanced model: python dcae/train_dcae.py [args]")
    print(f"2. Use EMA model for inference: EnhancedDCAEInference(model_path, use_ema=True)")
    print(f"3. All enhancements are automatically applied when using enhanced models")
    print(f"4. Check training logs to see improvement metrics")


if __name__ == "__main__":
    main()