#!/usr/bin/env python3
"""
Training Utils 테스트 스크립트
"""

import torch
import sys
import os

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_training_utils():
    """Training Utils 테스트"""
    print("=== Training Utils 테스트 ===")
    
    try:
        # Config 테스트
        from dcae.training_utils import EnhancedDCAEConfig
        config = EnhancedDCAEConfig()
        print(f"✅ Config 생성 성공: lr={config.learning_rate}, latent={config.latent_channels}")
        
        # EMA 테스트
        from dcae.training_utils import EMAWrapper
        from dcae.model import create_cqt_ssm_dcae
        
        model = create_cqt_ssm_dcae('small')
        ema = EMAWrapper(model, decay=0.999)
        print(f"✅ EMA wrapper 생성 성공: decay={ema.decay}")
        
        # Training State Manager 테스트
        from dcae.training_utils import TrainingStateManager
        state_manager = TrainingStateManager(config)
        print("✅ TrainingStateManager 생성 성공")
        
        # Setup EMA
        state_manager.setup_ema(model)
        print("✅ EMA setup 성공")
        
        # Compute functions 테스트
        from dcae.training_utils import compute_snr, compute_si_sdr
        
        # 테스트 데이터
        audio1 = torch.randn(2, 44100)
        audio2 = audio1 + torch.randn(2, 44100) * 0.1
        
        snr = compute_snr(audio1, audio2)
        si_sdr = compute_si_sdr(audio1.flatten(), audio2.flatten())
        
        print(f"✅ Metrics 계산 성공: SNR={snr:.2f}, SI-SDR={si_sdr:.2f}")
        
        print("\n🎉 Training Utils 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"❌ Training Utils 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_training_utils()
