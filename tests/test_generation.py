# lyro/tests/test_generation_fixed.py
"""
음악 생성 테스트 스크립트 (CQT 모델 지원)
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import time
from pathlib import Path

# LYRO 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssm.model import LyroSSMUNet, TaskController
from ssm.flow_matching import LyroFlowMatching, FlowConfig
from dcae.model import create_cqt_ssm_dcae  # ✅ CQT 모델 사용
from dataset.tokenizer import LyroTokenizer


def create_test_models():
    """테스트용 모델 생성"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # SSM 모델 생성
    ssm_model = LyroSSMUNet(
        input_channels=8,
        hidden_dims=[64, 64],  # 작은 모델
        ssm_layers=[1, 1],
        max_seq_len=1024
    ).to(device).eval()
    
    # CQT DCAE 모델 생성
    dcae_model = create_cqt_ssm_dcae(
        sample_rate=44100,
        latent_channels=8,
        model_size="small",
        use_torch_compile=False,
        use_mixed_precision=False
    ).to(device).eval()
    
    # Flow Matching 설정
    flow_matching = LyroFlowMatching(
        model=ssm_model,
        scheduler_type="cosine",
        solver_type="euler",
        sigma=1e-4,
        flow_type="rectified"
    )
    flow_matching.flow_steps = 4
    
    tokenizer = LyroTokenizer()
    
    return {
        'ssm': ssm_model,
        'dcae': dcae_model,
        'flow': flow_matching,
        'tokenizer': tokenizer,
        'device': device
    }


def test_song_generation():
    """SONG 태스크 테스트"""
    print("Testing SONG generation...")
    
    models = create_test_models()
      # 조건 준비
    lyrics = "Test lyrics for generation"
    lyrics_tokens = models['tokenizer'].encode_text(lyrics)
    conditions = TaskController.create_conditions(
        task_type='SONG',
        lyrics=torch.tensor([lyrics_tokens]).to(models['device']),
        style_prompt=torch.randn(1, 512).to(models['device']),
        device=models['device']
    )
    
    # 생성
    shape = (1, 8, 256)  # 짧은 생성
    
    start_time = time.time()
    with torch.no_grad():
        generated, _ = models['flow'].generate(
            shape=shape,
            conditions=conditions,
            num_steps=4,
            cfg_scale=1.5
        )
    generation_time = time.time() - start_time
    
    # 검증
    assert generated.shape == shape
    assert not torch.isnan(generated).any()
    assert not torch.isinf(generated).any()
    
    # RTF 계산 (CQT 압축율 32x 반영)
    duration = shape[2] * 512 * 32 / 44100  # 초
    rtf = generation_time / duration
    
    print(f"✅ SONG generation successful")
    print(f"   Shape: {generated.shape}")
    print(f"   Time: {generation_time:.3f}s")
    print(f"   RTF: {rtf:.3f}")
    return True


def test_dcae_encoding():
    """DCAE 인코딩/디코딩 테스트"""
    print("Testing DCAE encoding/decoding...")
    
    models = create_test_models()
    
    # 더미 오디오 생성 (1초)
    dummy_audio = torch.randn(1, 2, 44100).to(models['device'])
    
    with torch.no_grad():
        # 인코딩
        start_time = time.time()
        latent, skip_features = models['dcae'].encode(dummy_audio)
        encode_time = time.time() - start_time
        
        # 디코딩
        start_time = time.time()
        reconstructed = models['dcae'].decode(latent, skip_features)
        decode_time = time.time() - start_time
    
    # 검증
    assert latent.shape[0] == 1  # batch size
    assert latent.shape[1] == 8  # latent channels
    assert reconstructed.shape == dummy_audio.shape
    
    # 압축비 계산
    compression_ratio = dummy_audio.numel() / latent.numel()
    
    print(f"✅ DCAE test successful")
    print(f"   Input shape: {dummy_audio.shape}")
    print(f"   Latent shape: {latent.shape}")
    print(f"   Compression ratio: {compression_ratio:.1f}x")
    print(f"   Encode time: {encode_time*1000:.1f}ms")
    print(f"   Decode time: {decode_time*1000:.1f}ms")
    return True


def test_tokenizer():
    """토크나이저 테스트"""
    print("Testing tokenizer...")
    
    tokenizer = LyroTokenizer()
      # 가사 인코딩
    lyrics = "This is a test song with lyrics"
    lyrics_tokens = tokenizer.encode_lyrics(lyrics)
    
    # 일반 텍스트 인코딩 (스타일용)
    style = "pop ballad, emotional"
    style_tokens = tokenizer.encode_text(style)
    
    # 검증
    assert len(lyrics_tokens) > 0
    assert len(style_tokens) > 0
    
    print(f"✅ Tokenizer test successful")
    print(f"   Lyrics tokens: {len(lyrics_tokens)}")
    print(f"   Style tokens: {len(style_tokens)}")
    return True


def test_memory_usage():
    """메모리 사용량 테스트"""
    print("Testing memory usage...")
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping memory test")
        return True
    
    models = create_test_models()
    torch.cuda.reset_peak_memory_stats()
    
    # 긴 시퀀스 생성
    shape = (1, 8, 1024)  # ~20초
    conditions = TaskController.create_conditions(
        task_type='SONG',
        lyrics=torch.tensor([models['tokenizer'].encode_text("Test")]).to(models['device']),
        style_prompt=torch.randn(1, 512).to(models['device']),
        device=models['device']
    )
    
    with torch.no_grad():
        generated, _ = models['flow'].generate(
            shape=shape,
            conditions=conditions,
            num_steps=4,
            cfg_scale=1.5
        )
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    print(f"✅ Memory test successful")
    print(f"   Peak memory: {peak_memory:.2f} GB")
    print(f"   Generated shape: {generated.shape}")
    return True


def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 50)
    print("LYRO Generation Tests (CQT Models)")
    print("=" * 50)
    
    tests = [
        test_tokenizer,
        test_dcae_encoding,
        test_song_generation,
        test_memory_usage,
    ]
    
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    print("=" * 50)
    
    return passed == len(tests)


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
