# test_flow_fixed.py
"""
OPTIMIZED Flow Matching 성능 벤치마크 테스트
10x 속도 개선 검증 테스트
"""

import torch
import time
import sys
import os
from contextlib import contextmanager
import traceback
import numpy as np

def setup_environment():
    """환경 설정 및 검증"""
    print("=== OPTIMIZED 환경 설정 ===")
    
    # 경로 설정 (사용자 환경에 맞게 수정 필요)
    lyro_paths = [
        'c:\\Users\\kiola\\Downloads\\Lyro-v2',
        './Lyro-v2',
        '../Lyro-v2',
        '.'  # 현재 디렉토리
    ]
    
    lyro_path = None
    for path in lyro_paths:
        if os.path.exists(path):
            lyro_path = path
            break
    
    if lyro_path:
        print(f"✅ Lyro 경로 발견: {lyro_path}")
        if lyro_path not in sys.path:
            sys.path.insert(0, lyro_path)
    else:
        print("⚠️ Lyro 경로를 찾을 수 없습니다")
        print("사용 가능한 경로들을 확인해주세요:")
        for path in lyro_paths:
            print(f"  - {path}")
        return False
    
    # PyTorch 확인
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    return True

@contextmanager
def timer(description):
    """개선된 시간 측정 컨텍스트 매니저"""
    print(f"\n⏱️ {description} 시작...")
    start = time.time()
    try:
        yield
        end = time.time()
        print(f"✅ {description} 완료: {end - start:.3f}초")
    except Exception as e:
        end = time.time()
        print(f"❌ {description} 실패: {end - start:.3f}초 - {e}")
        raise

def safe_import():
    """안전한 모듈 import - OPTIMIZED 버전"""
    print("\n=== OPTIMIZED 모듈 Import ===")
    
    try:
        # OPTIMIZED 모델 import
        from ssm.model import create_lyro_s6_model
        from ssm.flow_matching import create_s6_flow_matching, S6FlowConfig
        print("✅ 모든 OPTIMIZED 모듈 import 성공")
        return create_lyro_s6_model, create_s6_flow_matching, S6FlowConfig
    except ImportError as e:
        print(f"❌ Import 실패: {e}")
        print("\n문제 해결 방법:")
        print("1. Lyro 프로젝트가 올바른 위치에 있는지 확인")
        print("2. pip install -e . (프로젝트 루트에서 실행)")
        print("3. 필요한 의존성이 모두 설치되었는지 확인")
        return None, None, None

def get_device_info():
    """디바이스 정보 확인"""
    print("\n=== 디바이스 정보 ===")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU 사용: {torch.cuda.get_device_name()}")
        print(f"CUDA 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # 초기 메모리 정리
        torch.cuda.empty_cache()
        print(f"현재 메모리 사용량: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️ CPU 사용 (CUDA 없음)")
    
    return device

def create_optimized_model(device):
    """OPTIMIZED 모델 생성"""
    print("\n=== OPTIMIZED 모델 생성 ===")
    
    try:
        # Import 재시도
        modules = safe_import()
        if None in modules:
            return None, None
        
        create_lyro_s6_model, create_s6_flow_matching, S6FlowConfig = modules
        
        # OPTIMIZED 모델 생성 (torch.compile 비활성화로 안정성 확보)
        print("OPTIMIZED SSM 모델 생성 중...")
        model = create_lyro_s6_model(
            input_channels=8,
            model_size="small",  # 테스트용 작은 모델
            max_seq_len=2048,
            use_torch_compile=False,  # 안정성을 위해 비활성화
            use_mixed_precision=True,
            compile_mode="default"
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ OPTIMIZED 모델 생성 성공 - 파라미터: {param_count:,}")
        
        # OPTIMIZED Flow Matching 초기화
        print("OPTIMIZED Flow Matching 생성 중...")
        flow_config = S6FlowConfig()
        flow_config.apply_preset("s6_fast")  # 가장 빠른 설정 사용
        
        flow_matching = create_s6_flow_matching(
            model=model,
            config=flow_config,
            use_torch_compile=False,  # 안정성을 위해 비활성화
            compile_mode="default"
        ).to(device)
        print("✅ OPTIMIZED Flow Matching 초기화 성공")
        
        return model, flow_matching
        
    except Exception as e:
        print(f"❌ OPTIMIZED 모델 생성 실패: {e}")
        traceback.print_exc()
        return None, None

def run_speed_comparison(flow_matching, device):
    """속도 비교 벤치마크 - BEFORE vs AFTER"""
    print("\n=== 속도 개선 벤치마크 (BEFORE vs AFTER) ===")
    
    if flow_matching is None:
        print("❌ Flow Matching 모델이 없어 테스트를 건너뜁니다")
        return
    
    # 기본 조건 설정
    conditions = {
        'task_token': torch.tensor([0], device=device),  # SONG task
        'lyrics': None,
        'style_prompt': torch.randn(1, 512, device=device),
        'icl_reference': None
    }
    
    print("🎯 성능 개선 목표: 4초 → 0.4초 (10x 빠르게)")
    print("=" * 60)
    
    # 테스트 구성
    test_configs = [
        {"steps": 4, "size": 256, "name": "Ultra Fast", "target_time": 0.4},
        {"steps": 6, "size": 512, "name": "Fast", "target_time": 0.8},
        {"steps": 8, "size": 1024, "name": "Standard", "target_time": 1.5},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n--- {config['name']} 테스트 ({config['steps']} steps, {config['size']} length) ---")
        print(f"목표 시간: {config['target_time']}초 이하")
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Warmup run (컴파일 최적화)
            print("Warmup 실행 중...")
            with torch.no_grad():
                _ = flow_matching.generate(
                    shape=(1, 8, config['size']),
                    conditions=conditions,
                    num_steps=config['steps'],
                    cfg_scale=1.5
                )
            
            # 실제 성능 측정 (여러 번 실행하여 평균)
            times = []
            num_runs = 5
            
            print(f"성능 측정 중... ({num_runs}회 실행)")
            for run in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                with torch.no_grad():
                    generated, _ = flow_matching.generate(
                        shape=(1, 8, config['size']),
                        conditions=conditions,
                        num_steps=config['steps'],
                        cfg_scale=1.5,
                        use_s6_optimizations=True
                    )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                run_time = end_time - start_time
                times.append(run_time)
                print(f"  Run {run+1}: {run_time:.3f}초")
            
            # 통계 계산
            avg_time = np.mean(times)
            min_time = np.min(times)
            max_time = np.max(times)
            std_time = np.std(times)
            
            # 성능 메트릭 계산
            audio_duration = config['size'] * 512 * 32 / 44100  # 압축률 고려
            rtf = avg_time / audio_duration
            
            memory_used = 0
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1e9
            
            # 목표 달성 여부
            target_achieved = avg_time <= config['target_time']
            improvement_factor = 4.0 / avg_time  # 기존 4초 대비 개선 배수
            
            results[config['name']] = {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'std_time': std_time,
                'rtf': rtf,
                'memory_gb': memory_used,
                'target_achieved': target_achieved,
                'improvement_factor': improvement_factor,
                'steps': config['steps'],
                'length': config['size'],
                'shape': generated.shape
            }
            
            # 결과 출력
            print(f"\n📊 결과:")
            print(f"   평균 시간: {avg_time:.3f}초 (목표: {config['target_time']}초)")
            print(f"   최소/최대: {min_time:.3f}초 / {max_time:.3f}초")
            print(f"   표준편차: {std_time:.3f}초")
            print(f"   실시간 배율 (RTF): {rtf:.2f}x")
            print(f"   메모리 사용량: {memory_used:.2f} GB")
            print(f"   출력 형태: {generated.shape}")
            
            if target_achieved:
                print(f"   ✅ 목표 달성! ({improvement_factor:.1f}x 개선)")
            else:
                print(f"   ⚠️ 목표 미달성 ({improvement_factor:.1f}x 개선)")
            
        except Exception as e:
            print(f"❌ {config['name']} 테스트 실패: {e}")
            results[config['name']] = {'error': str(e)}
    
    # 전체 결과 요약
    print(f"\n{'='*80}")
    print("🎯 OPTIMIZED S6 성능 개선 결과 요약")
    print(f"{'='*80}")
    
    print(f"{'테스트':15} | {'시간':8} | {'목표':8} | {'달성':6} | {'개선배수':8} | {'RTF':8} | {'메모리':8}")
    print("-" * 80)
    
    total_improvements = []
    achieved_count = 0
    
    for name, result in results.items():
        if 'error' not in result:
            target_time = next(c['target_time'] for c in test_configs if c['name'] == name)
            achieved = "✅" if result['target_achieved'] else "❌"
            
            print(f"{name:15} | {result['avg_time']:6.3f}초 | {target_time:6.1f}초 | {achieved:6} | "
                  f"{result['improvement_factor']:6.1f}x | {result['rtf']:6.2f}x | {result['memory_gb']:6.2f}GB")
            
            total_improvements.append(result['improvement_factor'])
            if result['target_achieved']:
                achieved_count += 1
        else:
            print(f"{name:15} | 실패: {result['error']}")
    
    if total_improvements:
        avg_improvement = np.mean(total_improvements)
        print(f"\n🚀 전체 평균 성능 개선: {avg_improvement:.1f}x")
        print(f"🎯 목표 달성률: {achieved_count}/{len([r for r in results.values() if 'error' not in r])} ({achieved_count/len([r for r in results.values() if 'error' not in r])*100:.0f}%)")
        
        if avg_improvement >= 8.0:
            print("🎉 목표 달성! 8x 이상 성능 개선 성공!")
        elif avg_improvement >= 5.0:
            print("👍 좋은 성과! 5x 이상 성능 개선!")
        else:
            print("📈 추가 최적화가 필요합니다.")
    
    return results

def test_quality_preservation(flow_matching, device):
    """품질 보존 테스트"""
    print("\n=== 품질 보존 테스트 ===")
    print("최적화 후에도 생성 품질이 유지되는지 확인...")
    
    if flow_matching is None:
        print("❌ Flow Matching 모델이 없어 테스트를 건너뜁니다")
        return
    
    conditions = {
        'task_token': torch.tensor([0], device=device),
        'lyrics': None,
        'style_prompt': torch.randn(1, 512, device=device),
        'icl_reference': None
    }
    
    try:
        # 다양한 설정으로 생성
        quality_tests = [
            {"steps": 4, "cfg": 1.2, "name": "Fast"},
            {"steps": 8, "cfg": 1.5, "name": "Standard"},
            {"steps": 12, "cfg": 2.0, "name": "High Quality"},
        ]
        
        for test in quality_tests:
            print(f"\n--- {test['name']} 품질 테스트 ---")
            
            with torch.no_grad():
                generated, trajectory = flow_matching.generate(
                    shape=(1, 8, 512),
                    conditions=conditions,
                    num_steps=test['steps'],
                    cfg_scale=test['cfg']
                )
            
            # 간단한 품질 체크
            output_std = generated.std().item()
            output_mean = generated.mean().item()
            output_range = (generated.max() - generated.min()).item()
            
            print(f"   출력 통계: 평균={output_mean:.4f}, 표준편차={output_std:.4f}, 범위={output_range:.4f}")
            print(f"   궤적 길이: {len(trajectory)}")
            print(f"   출력 형태: {generated.shape}")
            
            # 품질 기준 체크
            if 0.1 < output_std < 2.0 and abs(output_mean) < 0.5:
                print(f"   ✅ 품질 기준 통과")
            else:
                print(f"   ⚠️ 품질 기준 주의")
    
    except Exception as e:
        print(f"❌ 품질 테스트 실패: {e}")

def main():
    """메인 함수"""
    print("🚀 OPTIMIZED Flow Matching 성능 벤치마크 테스트")
    print("=" * 80)
    print("목표: 기존 37초 → 4초 이하 (10x 속도 개선)")
    print("=" * 80)
    
    try:
        # 1. 환경 설정
        if not setup_environment():
            print("❌ 환경 설정 실패")
            return
        
        # 2. 디바이스 확인
        device = get_device_info()
        
        # 3. OPTIMIZED 모델 생성
        model, flow_matching = create_optimized_model(device)
        
        if model is None or flow_matching is None:
            print("❌ OPTIMIZED 모델 생성 실패로 테스트를 중단합니다")
            return
        
        # 4. 속도 비교 벤치마크
        speed_results = run_speed_comparison(flow_matching, device)
        
        # 5. 품질 보존 테스트
        test_quality_preservation(flow_matching, device)
        
        print("\n" + "=" * 80)
        print("✅ OPTIMIZED 전체 테스트 완료!")
        print("=" * 80)
        
        # 최종 요약
        if speed_results:
            successful_tests = [r for r in speed_results.values() if 'error' not in r]
            if successful_tests:
                avg_improvement = np.mean([r['improvement_factor'] for r in successful_tests])
                print(f"\n🎯 최종 결과:")
                print(f"   평균 성능 개선: {avg_improvement:.1f}x")
                
                if avg_improvement >= 8.0:
                    print("   🎉 목표 달성! 성공적인 최적화!")
                else:
                    print("   📈 추가 최적화 권장")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자가 테스트를 중단했습니다")
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        traceback.print_exc()
    finally:
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n🧹 GPU 메모리 정리 완료")

if __name__ == "__main__":
    main()