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

from dcae.model import create_memory_optimized_lyro_dcae
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
        
        try:
            # 모델 생성
            print(f"\n1. {model_size} 모델 생성 중...")
            model = create_memory_optimized_lyro_dcae(
                model_size=model_size,
                sample_rate=44100,
                use_vq=False,
                memory_efficient=True,
                chunk_size=512,  # 작은 청크 크기로 메모리 절약
                checkpointing_segments=2  # 적은 세그먼트
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
                    
                    # 채널 분리 테스트
                    if model.dual_channel_processing:
                        print("   채널 분리 테스트...")
                        latent, skip_features = model.encode(audio)
                        print(f"   - Latent shape: {latent.shape}")
                        
                        vocal_latent, inst_latent = model.separate_channels(latent)
                        print(f"   - Vocal latent shape: {vocal_latent.shape}")
                        print(f"   - Instrumental latent shape: {inst_latent.shape}")
                        
                        # 채널 분리가 올바른지 검증
                        expected_vocal_channels = min(4, model.latent_channels // 2)
                        expected_inst_channels = min(4, model.latent_channels // 2)
                        
                        if vocal_latent.shape[1] == expected_vocal_channels and inst_latent.shape[1] == expected_inst_channels:
                            print(f"   ✅ 채널 분리 성공 (vocal: {expected_vocal_channels}, inst: {expected_inst_channels})")
                        else:
                            print(f"   ❌ 채널 분리 실패")
                    else:
                        print("   채널 분리 비활성화됨")
                    
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