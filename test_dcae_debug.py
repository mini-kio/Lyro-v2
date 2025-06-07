#!/usr/bin/env python3
"""
DCAE Model Debugging Test Script
"""

import torch
import sys
import os
import traceback

# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dcae.model import create_cqt_ssm_dcae  # Updated import
from dcae.training_utils import compute_snr, compute_si_sdr

def test_dcae_models():
    """Test DCAE models systematically for issues"""
    print("=== DCAE Model Debugging Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test model sizes
    model_sizes = ['small', 'base', 'large']
    
    for model_size in model_sizes:
        print(f"\n{'='*60}")
        print(f"Model size: {model_size}")
        print(f"{'='*60}")
        
        try:            
            # Model creation (bypass torch.compile issues)
            print(f"\n1. Creating {model_size} model...")
            
            # Bypass torch.compile issues on Windows
            import platform
            use_compile = False  # Completely disabled for debugging
            
            model = create_cqt_ssm_dcae(
                model_size=model_size,
                sample_rate=44100,
                latent_channels=8,
                use_torch_compile=use_compile,
                use_mixed_precision=True,
                compile_mode="reduce-overhead" if use_compile else "default"
            ).to(device)
            
            print(f"[SUCCESS] {model_size} model created successfully")
            print(f"   - Latent channels: {model.latent_channels}")
            
            # Model parameter count output
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   - Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"[ERROR] {model_size} model creation failed: {e}")
            traceback.print_exc()
            continue
        
        # Test with various audio lengths
        audio_lengths = [22050, 44100, 88200]  # 0.5s, 1s, 2s
        
        for audio_length in audio_lengths:
            duration = audio_length / 44100
            print(f"\n2. Testing {duration}s audio ({audio_length} samples)")
            
            try:
                # Generate test input
                batch_size = 1  # Batch size 1 for memory savings
                audio = torch.randn(batch_size, 2, audio_length).to(device)
                print(f"   Input shape: {audio.shape}")
                
                model.eval()
                with torch.no_grad():
                    # Forward pass test
                    print("   Encoding/decoding test...")
                    reconstructed, loss_dict = model(audio, return_loss=True)
                    
                    print(f"   [SUCCESS] Forward pass completed")
                    print(f"   - Input shape: {audio.shape}")
                    print(f"   - Output shape: {reconstructed.shape}")
                    print(f"   - Loss: {loss_dict['total_loss'].item():.4f}")
                    
                    # Encoding/decoding structure test
                    print("   Encoding/decoding structure test...")
                    latent, skip_features = model.encode(audio)
                    print(f"   - Latent shape: {latent.shape}")
                    print(f"   - Skip features count: {len(skip_features)}")
                    
                    # Skip features info output
                    for i, skip in enumerate(skip_features):
                        print(f"   - Skip {i} shape: {skip.shape}")
                    
                    # Individual decoding test
                    reconstructed_from_latent = model.decode(latent, skip_features)
                    print(f"   - Decoded shape: {reconstructed_from_latent.shape}")
                    
                    # Memory efficiency verification
                    latent_memory = latent.numel() * 4 / 1024 / 1024  # MB
                    audio_memory = audio.numel() * 4 / 1024 / 1024   # MB
                    compression_ratio = audio_memory / latent_memory
                    print(f"   - Compression ratio: {compression_ratio:.1f}x")
                    print(f"   - Memory savings: {(1 - latent_memory/audio_memory)*100:.1f}%")
                    
                    # SNR/SI-SDR test
                    print("   Quality metrics test...")
                    try:
                        # Align to same length
                        min_length = min(audio.shape[-1], reconstructed.shape[-1])
                        audio_trimmed = audio[0, :, :min_length]
                        reconstructed_trimmed = reconstructed[0, :, :min_length]
                        
                        snr = compute_snr(audio_trimmed, reconstructed_trimmed)
                        si_sdr = compute_si_sdr(audio_trimmed.flatten(), reconstructed_trimmed.flatten())
                        
                        print(f"   - SNR: {snr:.2f} dB")
                        print(f"   - SI-SDR: {si_sdr:.2f} dB")
                        
                    except Exception as snr_error:
                        print(f"   [ERROR] SNR/SI-SDR calculation failed: {snr_error}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   [ERROR] GPU memory insufficient: {e}")
                    torch.cuda.empty_cache()
                else:
                    print(f"   [ERROR] Runtime error: {e}")
                    traceback.print_exc()
            except Exception as e:
                print(f"   [ERROR] Test failed: {e}")
                traceback.print_exc()
        
        # Memory cleanup
        del model
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("Debugging test completed")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_dcae_models()
