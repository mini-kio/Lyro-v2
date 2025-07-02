#!/usr/bin/env python3
"""
LYRO Import 테스트 스크립트
모든 import가 정상적으로 작동하는지 확인
"""

import os
import sys

# 현재 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """모든 주요 import 테스트"""
    print("🧪 LYRO Import 테스트 시작...")
    
    try:
        # 1. 데이터 패키지
        print("1️⃣ 데이터 패키지 테스트...")
        from data import LyroDataset, LyroCollator, LyroTokenizer, create_lyro_datasets
        print("   ✅ 데이터 패키지 OK")
        
        # 2. 모델 패키지
        print("2️⃣ 모델 패키지 테스트...")
        from models import LyroGenerator, PretrainedDCAE, FlowMatchingLoss
        print("   ✅ 모델 패키지 OK")
        
        # 3. 추론 패키지
        print("3️⃣ 추론 패키지 테스트...")
        from inference import LyroPipeline, GenerationConfig, create_simple_pipeline
        print("   ✅ 추론 패키지 OK")
        
        # 4. 훈련 패키지
        print("4️⃣ 훈련 패키지 테스트...")
        from training import LyroConfig
        from training.trainer import create_trainer
        print("   ✅ 훈련 패키지 OK")
        
        # 5. 유틸리티 패키지
        print("5️⃣ 유틸리티 패키지 테스트...")
        from utils import AudioProcessor, MetricCalculator, safe_save_audio
        print("   ✅ 유틸리티 패키지 OK")
        
        print("\n🎉 모든 Import 테스트 통과!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import 오류: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
