#!/usr/bin/env python3
"""
빠른 단위 테스트 - CompressionAwareMixScaleAugmentation 형태 안전성
"""

import torch
import random
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dcae.training_utils import CompressionAwareMixScaleAugmentation

def test_augmentation_shapes():
    """Test augmentation with different input shapes"""
    print("🧪 Testing CompressionAwareMixScaleAugmentation shape safety...")
    
    # Create augmentation instance
    aug = CompressionAwareMixScaleAugmentation(
        sample_rate=44100,
        augmentation_prob=1.0  # Always apply for testing
    )
    
    test_shapes = [
        (44100,),           # (T)
        (2, 44100),         # (C, T) 
        (4, 2, 44100)       # (B, C, T)
    ]
    
    for i, shape in enumerate(test_shapes):
        print(f"  Test {i+1}: Input shape {shape}")
        
        try:
            x = torch.randn(*shape)
            y = aug(x.clone())
            
            assert y.shape == x.shape, f"Shape mismatch: input {x.shape}, output {y.shape}"
            print(f"    ✅ Output shape: {y.shape} (preserved)")
            
        except Exception as e:
            print(f"    ❌ Failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("✅ All augmentation shape tests passed!")
    return True

def test_specific_filters():
    """Test specific filter functions"""
    print("\n🔧 Testing specific filter functions...")
    
    aug = CompressionAwareMixScaleAugmentation(sample_rate=44100)
    
    # Test 2D input (N, T) - this is what filters receive after flattening
    test_audio = torch.randn(6, 44100)  # (N=6, T=44100) - simulates (B=2, C=3) flattened
    
    try:
        # Test high-pass filter
        if aug.compression_safe_filters and 'highpass' in aug.compression_safe_filters:
            hp_filter = aug.compression_safe_filters['highpass'][0]
            filtered_hp = hp_filter(test_audio.clone())
            assert filtered_hp.shape == test_audio.shape
            print("    ✅ High-pass filter: shape preserved")
        
        # Test low-pass filter  
        if aug.compression_safe_filters and 'lowpass' in aug.compression_safe_filters:
            lp_filter = aug.compression_safe_filters['lowpass'][0]
            filtered_lp = lp_filter(test_audio.clone())
            assert filtered_lp.shape == test_audio.shape
            print("    ✅ Low-pass filter: shape preserved")
            
        # Test pitch shift
        shifted = aug.apply_harmonic_preserving_pitch_shift(test_audio.clone())
        assert shifted.shape == test_audio.shape
        print("    ✅ Pitch shift: shape preserved")
        
    except Exception as e:
        print(f"    ❌ Filter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✅ All filter tests passed!")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Augmentation Shape Safety")
    print("=" * 50)
    
    success = True
    success &= test_augmentation_shapes()
    success &= test_specific_filters()
    
    if success:
        print("\n🎉 All tests passed! Augmentation is batch-safe.")
    else:
        print("\n💥 Some tests failed. Please check the error messages above.")
        sys.exit(1)
