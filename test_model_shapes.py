#!/usr/bin/env python3
"""
테스트 스크립트: Lyro SSM 모델의 텐서 모양 검증
"""

import torch
import sys
import os

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ssm.model import LyroSSMUNet
from ssm.flow_matching import LyroFlowMatching, FlowConfig

def test_model_shapes():
    """모델의 텐서 모양이 올바른지 테스트"""
    print("=== Lyro SSM 모델 텐서 모양 테스트 ===")
    
    # 모델 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 모델 생성
    print("\n1. LyroSSMUNet 모델 생성 중...")
    try:
        model = LyroSSMUNet(
            input_channels=8,
            hidden_dims=[128, 256, 384, 512],
            ssm_layers=[2, 3, 4, 4],
            d_state=64,
            max_seq_len=8192,
            dropout=0.1,
            use_multiscale_ssm=True,
        ).to(device)
        print("✅ 모델 생성 성공")
        
        # 모델 파라미터 개수 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"전체 파라미터: {total_params:,}")
        print(f"학습 가능한 파라미터: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ 모델 생성 실패: {e}")
        return False
    
    # 테스트 입력 생성
    print("\n2. 테스트 입력 생성 중...")
    batch_size = 2
    seq_len = 1024
    input_channels = 8
    
    try:
        # 입력 텐서
        x = torch.randn(batch_size, input_channels, seq_len).to(device)
        time = torch.rand(batch_size).to(device)
        
        # 조건부 입력들
        key = torch.randint(0, 12, (batch_size,)).to(device)  # 키
        tempo = torch.randint(60, 180, (batch_size,)).to(device)  # 템포  
        genre = torch.randint(0, 10, (batch_size,)).to(device)  # 장르
        mood = torch.randint(0, 5, (batch_size,)).to(device)  # 무드
        structure = torch.randint(0, 8, (batch_size,)).to(device)  # 구조
        
        print(f"입력 모양: {x.shape}")
        print(f"시간 모양: {time.shape}")
        print(f"조건부 입력들: key={key.shape}, tempo={tempo.shape}, genre={genre.shape}, mood={mood.shape}, structure={structure.shape}")
        
    except Exception as e:
        print(f"❌ 테스트 입력 생성 실패: {e}")
        return False
    
    # Forward pass 테스트
    print("\n3. Forward pass 테스트 중...")
    try:
        model.eval()
        with torch.no_grad():
            output = model(
                x=x,
                time=time,
                key=key,
                tempo=tempo,
                genre=genre,
                mood=mood,
                structure=structure
            )
        
        print(f"✅ Forward pass 성공")
        print(f"출력 모양: {output.shape}")
        print(f"출력 범위: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 출력 모양 검증
        expected_shape = (batch_size, input_channels, seq_len)
        if output.shape == expected_shape:
            print(f"✅ 출력 모양이 예상과 일치: {expected_shape}")
        else:
            print(f"❌ 출력 모양 불일치. 예상: {expected_shape}, 실제: {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Forward pass 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Flow Matching 모델 테스트
    print("\n4. Flow Matching 모델 테스트 중...")
    try:
        config = FlowConfig(
            model_dim=512,
            num_layers=12,
            num_heads=8,
            hidden_dim=2048,
            max_seq_len=8192,
            dropout=0.1
        )
        
        flow_model = LyroFlowMatching(config).to(device)
        print("✅ Flow Matching 모델 생성 성공")
        
        # Flow Matching forward pass
        with torch.no_grad():
            # 잠재 공간 입력 (DCAE 출력 가정)
            latent_dim = 64
            latent_seq_len = seq_len // 4  # downsampling 가정
            z = torch.randn(batch_size, latent_dim, latent_seq_len).to(device)
            
            flow_output = flow_model(
                z=z,
                time=time,
                key=key,
                tempo=tempo,
                genre=genre,
                mood=mood,
                structure=structure
            )
        
        print(f"✅ Flow Matching forward pass 성공")
        print(f"Flow 출력 모양: {flow_output.shape}")
        
    except Exception as e:
        print(f"❌ Flow Matching 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 모든 테스트 통과!")
    return True

if __name__ == "__main__":
    success = test_model_shapes()
    if not success:
        sys.exit(1)
