# examples/dcae_demo.py
"""
DCAE 오디오 인코더 데모 - 오디오 ↔ latent 변환 테스트
"""

import torch
import torchaudio
import sys
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import MultimodalLyroSystem, create_audio_encoder


def demo_audio_encoder():
    """오디오 인코더 단독 테스트"""
    print("DCAE Audio Encoder Demo")
    print("=" * 50)
    
    # 1. 오디오 인코더 생성
    print("1. Creating audio encoder...")
    audio_encoder = create_audio_encoder()
    print(f"   Encoder info: {audio_encoder.get_info()}")
    
    # 2. 더미 오디오 생성 (2초, 스테레오, 44.1kHz)
    print("\n2. Creating dummy audio...")
    sample_rate = 44100
    duration = 2.0
    dummy_audio = torch.randn(1, 2, int(sample_rate * duration))
    print(f"   Audio shape: {dummy_audio.shape}")
    print(f"   Sample rate: {sample_rate}Hz")
    print(f"   Duration: {duration}s")
    
    # 3. 오디오 → Latent 인코딩
    print("\n3. Encoding audio to latents...")
    latents, latent_lengths = audio_encoder.encode_audio(dummy_audio, sample_rate)
    print(f"   Latent shape: {latents.shape}")
    print(f"   Latent lengths: {latent_lengths}")
    print(f"   Latent range: [{latents.min():.3f}, {latents.max():.3f}]")
    
    # 4. Latent → 오디오 디코딩
    print("\n4. Decoding latents to audio...")
    if audio_encoder.dcae_available:
        sr_out, audio_list = audio_encoder.decode_latents(latents, sample_rate)
        print(f"   Output sample rate: {sr_out}Hz")
        print(f"   Number of audio files: {len(audio_list)}")
        if audio_list:
            reconstructed_audio = audio_list[0]
            print(f"   Reconstructed audio shape: {reconstructed_audio.shape}")
            print(f"   Audio range: [{reconstructed_audio.min():.3f}, {reconstructed_audio.max():.3f}]")
    else:
        print("   WARNING: DCAE not available, skipping audio reconstruction")
    
    print("\nAudio Encoder demo completed!")


def demo_integrated_system():
    """통합 시스템에서 오디오 인코더 테스트"""
    print("\n" + "=" * 50)
    print("🎼 Integrated System Demo")
    print("=" * 50)
    
    # 1. 통합 시스템 생성
    print("1️⃣ Creating integrated system...")
    try:
        system = MultimodalLyroSystem(use_audio_encoder=True)
        print("   ✅ System created with audio encoder")
    except Exception as e:
        print(f"   ❌ System creation failed: {e}")
        return
    
    # 2. 오디오 인코딩 테스트
    print("\n2️⃣ Testing audio encoding...")
    dummy_audio = torch.randn(2, 2, 44100)  # 2 batch, 2 channels, 1초
    print(f"   Input audio shape: {dummy_audio.shape}")
    
    if system.use_audio_encoder:
        try:
            latents = system.encode_audio(dummy_audio)
            print(f"   Encoded latents shape: {latents.shape}")
        except Exception as e:
            print(f"   ❌ Audio encoding failed: {e}")
            return
    else:
        print("   ⚠️  Audio encoder not available")
        return
    
    # 3. 텍스트 기반 음악 생성 (latent만)
    print("\n3️⃣ Generating music from text...")
    try:
        lyrics = ["Beautiful melody flows through the air"]
        style = ["piano", "emotional", "slow"]
        
        generated_latents = system.generate_music(
            lyrics=lyrics,
            style=style,
            num_steps=10,  # 빠른 테스트를 위해 적은 스텝
            return_audio=False
        )
        print(f"   Generated latents shape: {generated_latents.shape}")
    except Exception as e:
        print(f"   ❌ Music generation failed: {e}")
        return
    
    # 4. 텍스트 기반 음악 생성 (오디오 변환 포함)
    print("\n4️⃣ Generating music with audio conversion...")
    if system.use_audio_encoder and system.audio_encoder.dcae_available:
        try:
            latents, sr, audio_list = system.generate_music(
                lyrics=lyrics,
                style=style,
                num_steps=10,
                return_audio=True,
                sample_rate=44100
            )
            print(f"   Generated latents shape: {latents.shape}")
            print(f"   Output sample rate: {sr}Hz")
            print(f"   Number of audio files: {len(audio_list)}")
            if audio_list:
                print(f"   First audio shape: {audio_list[0].shape}")
        except Exception as e:
            print(f"   ❌ Audio generation failed: {e}")
    else:
        print("   ⚠️  DCAE not available, skipping audio generation")
    
    print("\n✅ Integrated system demo completed!")


def demo_real_audio_file():
    """실제 오디오 파일 테스트 (있는 경우)"""
    print("\n" + "=" * 50)
    print("🎧 Real Audio File Demo")
    print("=" * 50)
    
    # 1. 오디오 파일 찾기
    audio_file = project_root / "1.mp3"
    if not audio_file.exists():
        print("   ⚠️  No audio file found (1.mp3), skipping real audio test")
        return
    
    print(f"1️⃣ Loading audio file: {audio_file}")
    
    try:
        # 2. 오디오 로딩
        audio, sr = torchaudio.load(audio_file)
        print(f"   Original audio shape: {audio.shape}")
        print(f"   Original sample rate: {sr}Hz")
        
        # 길이 제한 (10초)
        max_length = sr * 10
        if audio.shape[-1] > max_length:
            audio = audio[..., :max_length]
            print(f"   Cropped to 10s: {audio.shape}")
        
        # 3. 오디오 인코더로 변환
        print(f"\n2️⃣ Encoding real audio...")
        audio_encoder = create_audio_encoder()
        
        # 배치 차원 추가
        audio_batch = audio.unsqueeze(0)  # (1, C, T)
        
        latents, lengths = audio_encoder.encode_audio(audio_batch, sr)
        print(f"   Encoded latents shape: {latents.shape}")
        print(f"   Latent statistics:")
        print(f"     - Mean: {latents.mean():.3f}")
        print(f"     - Std: {latents.std():.3f}")
        print(f"     - Min: {latents.min():.3f}")
        print(f"     - Max: {latents.max():.3f}")
        
        # 4. 다시 오디오로 변환
        if audio_encoder.dcae_available:
            print(f"\n3️⃣ Decoding back to audio...")
            sr_out, audio_list = audio_encoder.decode_latents(latents)
            if audio_list:
                reconstructed = audio_list[0]
                print(f"   Reconstructed audio shape: {reconstructed.shape}")
                
                # 품질 분석 (간단한 MSE)
                min_length = min(audio.shape[-1], reconstructed.shape[-1])
                if audio.shape[0] == reconstructed.shape[0]:
                    mse = torch.mean((audio[..., :min_length] - reconstructed[..., :min_length]) ** 2)
                    print(f"   Reconstruction MSE: {mse:.6f}")
                else:
                    print("   Channel mismatch, skipping MSE calculation")
        
        print(f"\n✅ Real audio test completed!")
        
    except Exception as e:
        print(f"   ❌ Real audio test failed: {e}")


def main():
    """메인 데모 실행"""
    print("DCAE Audio Encoder Complete Demo")
    print("=" * 60)
    
    # GPU 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    try:
        # 1. 오디오 인코더 단독 테스트
        demo_audio_encoder()
        
        # 2. 통합 시스템 테스트
        demo_integrated_system()
        
        # 3. 실제 오디오 파일 테스트
        demo_real_audio_file()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Demo completed! Check the results above.")


if __name__ == "__main__":
    main()