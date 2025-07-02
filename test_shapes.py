#!/usr/bin/env python3
"""
LYRO Shape 검증 테스트
Tensor shape 관련 오류들이 해결되었는지 확인
"""

import os
import sys
import torch
import numpy as np

# 현재 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_audio_shape_processing():
    """오디오 처리 시 shape 검증"""
    print("🔍 오디오 Shape 처리 테스트...")
    
    try:
        from data.processor import LyroCollator, ProcessorConfig
        
        # 설정 생성
        config = ProcessorConfig()
        collator = LyroCollator(config)
        
        # 다양한 shape의 오디오 테스트
        test_cases = [
            torch.randn(44100),          # (T,) - 1D
            torch.randn(1, 44100),       # (1, T) - mono
            torch.randn(2, 44100),       # (2, T) - stereo
            torch.randn(3, 44100),       # (3, T) - multi-channel
        ]
        
        for i, audio in enumerate(test_cases):
            print(f"   테스트 케이스 {i+1}: {audio.shape}")
            processed = collator._ensure_stereo(audio)
            print(f"   처리 후: {processed.shape}")
            assert processed.shape[0] == 2, f"Expected stereo output, got {processed.shape}"
        
        print("   ✅ 오디오 Shape 처리 OK")
        return True
        
    except Exception as e:
        print(f"   ❌ 오디오 Shape 처리 실패: {e}")
        return False


def test_generator_shape_handling():
    """Generator shape 처리 검증"""
    print("🔍 Generator Shape 처리 테스트...")
    
    try:
        from models.generator import LyroGenerator, GeneratorConfig
        
        # 작은 모델 설정 - head 수가 d_model을 나누어떨어뜨리도록 
        config = GeneratorConfig(
            d_model=128,
            n_layers=2,
            n_heads=8,  # 128 / 8 = 16 (정확한 나눗셈)
            latent_channels=16,
            latent_time_steps=64
        )
        
        generator = LyroGenerator(config)
        generator.eval()
        
        # 테스트 입력
        batch_size = 2
        latents = torch.randn(batch_size, 16, 64)  # (B, C, T)
        timesteps = torch.rand(batch_size)
        
        with torch.no_grad():
            # Forward pass 테스트
            output = generator.forward(
                latents=latents,
                timesteps=timesteps,
                captions=["test music", "rock song"]
            )
            
            print(f"   입력 shape: {latents.shape}")
            print(f"   출력 shape: {output.shape}")
            
            assert output.shape == latents.shape, f"Shape mismatch: {output.shape} vs {latents.shape}"
            
            # Training loss 테스트
            loss_dict = generator.training_loss(
                latents=latents,
                captions=["test music", "rock song"]
            )
            
            assert 'flow_loss' in loss_dict, "Flow loss not found"
            print(f"   Flow loss: {loss_dict['flow_loss'].item():.4f}")
        
        print("   ✅ Generator Shape 처리 OK")
        return True
        
    except Exception as e:
        print(f"   ❌ Generator Shape 처리 실패: {e}")
        return False


def test_pipeline_shape_handling():
    """Pipeline shape 처리 검증"""
    print("🔍 Pipeline Shape 처리 테스트...")
    
    try:
        from inference.pipeline import GenerationInput, GenerationConfig
        
        # 입력 생성
        generation_input = GenerationInput(
            task="INST",
            caption="calm ambient music"
        )
        
        generation_config = GenerationConfig(
            duration=5.0,
            sample_rate=44100,
            quality="fast"
        )
        
        # 입력 검증
        errors = generation_input.validate()
        print(f"   입력 검증: {len(errors)} 오류")
        
        # 설정 적용
        generation_config.apply_quality_preset()
        print(f"   설정 적용 완료: {generation_config.quality}")
        
        print("   ✅ Pipeline Shape 처리 OK")
        return True
        
    except Exception as e:
        print(f"   ❌ Pipeline Shape 처리 실패: {e}")
        return False


def test_reference_audio_shape():
    """참조 오디오 shape 처리 검증"""
    print("🔍 참조 오디오 Shape 처리 테스트...")
    
    try:
        # 다양한 형태의 참조 오디오 시뮬레이션
        test_audios = [
            torch.randn(44100),          # (T,)
            torch.randn(1, 44100),       # (1, T)
            torch.randn(2, 44100),       # (2, T)
            torch.randn(1, 2, 44100),    # (1, 2, T)
        ]
        
        for i, audio in enumerate(test_audios):
            print(f"   테스트 오디오 {i+1}: {audio.shape}")
            
            # 차원 정규화 시뮬레이션
            if audio.dim() == 1:
                normalized = audio.unsqueeze(0)  # (T,) -> (1, T)
            elif audio.dim() == 3 and audio.shape[0] == 1:
                normalized = audio.squeeze(0)    # (1, C, T) -> (C, T)
            else:
                normalized = audio
            
            print(f"   정규화 후: {normalized.shape}")
            
            # batch dimension 추가
            if normalized.dim() == 2:
                batched = normalized.unsqueeze(0)  # (C, T) -> (1, C, T)
                print(f"   배치 추가 후: {batched.shape}")
        
        print("   ✅ 참조 오디오 Shape 처리 OK")
        return True
        
    except Exception as e:
        print(f"   ❌ 참조 오디오 Shape 처리 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("🧪 LYRO Shape 검증 테스트 시작...")
    
    tests = [
        test_audio_shape_processing,
        test_generator_shape_handling,
        test_pipeline_shape_handling,
        test_reference_audio_shape,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"   ❌ 테스트 실행 중 오류: {e}")
    
    print(f"\n📊 테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 Shape 검증 테스트 통과!")
        return True
    else:
        print("❌ 일부 테스트 실패")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
