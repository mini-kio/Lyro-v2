#!/usr/bin/env python3
"""
DCAE 모델 디버깅 테스트 스크립트
"""

import torch
import sys
import os
import traceback

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dcae.model import create_cqt_ssm_dcae  # Updated import
from dcae.training_utils import compute_snr, compute_si_sdr

def test_dcae_models():
    """DCAE 모델들의 문제점을 체계적으로 테스트"""
    print("=== DCAE 모델 디버깅 테스트 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 테스트할 모델 크기들
    model_sizes = ['small', 'base', 'large']
    
    for model_size in model_sizes:
        print(f"\n{'='*60}")
        print(f"모델 크기: {model_size}")
        print(f"{'='*60}")
        
        try:            # 모델 생성 (최적화 적용, Triton 이슈 우회)
            print(f"\n1. {model_size} 모델 생성 중...")
            
            # Windows에서 torch.compile 이슈 우회
            import platform
            use_compile = platform.system() != "Windows"
            
            model = create_cqt_ssm_dcae(
                model_size=model_size,
                sample_rate=44100,
                latent_channels=8,
                use_torch_compile=use_compile,
                use_mixed_precision=True,
                compile_mode="reduce-overhead" if use_compile else "default"
            ).to(device)
            
            print(f"✅ {model_size} 모델 생성 성공")
            print(f"   - Latent channels: {model.latent_channels}")
            
            # 모델 파라미터 수 출력
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   - 총 파라미터: {total_params:,}")
            
        except Exception as e:
            print(f"❌ {model_size} 모델 생성 실패: {e}")
            traceback.print_exc()
            continue
        
        # 다양한 오디오 길이로 테스트
        audio_lengths = [22050, 44100, 88200]  # 0.5초, 1초, 2초
        
        for audio_length in audio_lengths:
            duration = audio_length / 44100
            print(f"\n2. {duration}초 오디오 테스트 ({audio_length} samples)")
            
            try:
                # 테스트 입력 생성
                batch_size = 1  # 메모리 절약을 위해 배치 크기 1
                audio = torch.randn(batch_size, 2, audio_length).to(device)
                print(f"   입력 shape: {audio.shape}")
                
                model.eval()
                with torch.no_grad():
                    # Forward pass 테스트
                    print("   인코딩/디코딩 테스트...")
                    reconstructed, loss_dict = model(audio, return_loss=True)
                    
                    print(f"   ✅ Forward pass 성공")
                    print(f"   - 입력 shape: {audio.shape}")
                    print(f"   - 출력 shape: {reconstructed.shape}")
                    print(f"   - 손실: {loss_dict['total_loss'].item():.4f}")
                      # 인코딩/디코딩 구조 테스트
                    print("   인코딩/디코딩 구조 테스트...")
                    latent, skip_features = model.encode(audio)
                    print(f"   - Latent shape: {latent.shape}")
                    print(f"   - Skip features count: {len(skip_features)}")
                    
                    # Skip features 정보 출력
                    for i, skip in enumerate(skip_features):
                        print(f"   - Skip {i} shape: {skip.shape}")
                    
                    # 개별 디코딩 테스트
                    reconstructed_from_latent = model.decode(latent, skip_features)
                    print(f"   - Decoded shape: {reconstructed_from_latent.shape}")
                    
                    # 메모리 효율성 검증
                    latent_memory = latent.numel() * 4 / 1024 / 1024  # MB
                    audio_memory = audio.numel() * 4 / 1024 / 1024   # MB
                    compression_ratio = audio_memory / latent_memory
                    print(f"   - 압축 비율: {compression_ratio:.1f}x")
                    print(f"   - 메모리 절약: {(1 - latent_memory/audio_memory)*100:.1f}%")
                    
                    # SNR/SI-SDR 테스트
                    print("   품질 메트릭 테스트...")
                    try:
                        # 동일한 길이로 맞춤
                        min_length = min(audio.shape[-1], reconstructed.shape[-1])
                        audio_trimmed = audio[0, :, :min_length]
                        reconstructed_trimmed = reconstructed[0, :, :min_length]
                        
                        snr = compute_snr(audio_trimmed, reconstructed_trimmed)
                        si_sdr = compute_si_sdr(audio_trimmed.flatten(), reconstructed_trimmed.flatten())
                        
                        print(f"   - SNR: {snr:.2f} dB")
                        print(f"   - SI-SDR: {si_sdr:.2f} dB")
                        
                    except Exception as snr_error:
                        print(f"   ❌ SNR/SI-SDR 계산 실패: {snr_error}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   ❌ GPU 메모리 부족: {e}")
                    torch.cuda.empty_cache()
                else:
                    print(f"   ❌ Runtime 오류: {e}")
                    traceback.print_exc()
            except Exception as e:
                print(f"   ❌ 테스트 실패: {e}")
                traceback.print_exc()
        
        # 메모리 정리
        del model
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("디버깅 테스트 완료")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_dcae_models()