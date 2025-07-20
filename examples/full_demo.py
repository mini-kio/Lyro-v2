#!/usr/bin/env python3
"""
LYRO v2 Complete System Demo
멀티모달 음악 생성 시스템 전체 기능 데모
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    MultimodalLyroSystem, GeneratorConfig, 
    get_model_info, estimate_model_memory, get_system_capabilities
)
from training.trainer import MultimodalTrainer
from training.config import LyroConfig
from inference.pipeline import create_pipeline, quick_generate

def demo_system_info():
    """시스템 정보 데모"""
    print("🎵 LYRO v2 - Multimodal Music Generation System")
    print("=" * 80)
    
    # 모델 정보
    model_info = get_model_info()
    print(f"\n📊 Model Information:")
    print(f"  Version: {model_info['version']}")
    print(f"  Total Parameters: {model_info['total_parameters']}")
    print(f"  Pipeline: {model_info['pipeline_flow']}")
    
    print(f"\n🏗️ Components:")
    for name, info in model_info['components'].items():
        print(f"  • {name.title()}:")
        print(f"    - Parameters: {info['parameters']}")
        print(f"    - Architecture: {info['architecture']}")
        if 'features' in info:
            print(f"    - Features: {', '.join(info['features'])}")
    
    print(f"\n🎯 Training Stages:")
    for stage in model_info['training_stages']:
        print(f"  {stage}")
    
    print(f"\n📝 Supported Tasks: {', '.join(model_info['supported_tasks'])}")
    print(f"📤 Output: {model_info['output']}")
    
    # 메모리 추정
    memory_info = estimate_model_memory(batch_size=4)
    print(f"\n💾 Memory Requirements:")
    print(f"  Total Estimated: {memory_info['total_estimated']}")
    print(f"  Recommended GPU: {memory_info['recommended_gpu']}")
    print(f"  Batch Size: {memory_info['batch_size']}")
    
    # 시스템 기능
    capabilities = get_system_capabilities()
    print(f"\n🎼 Text Processing Capabilities:")
    print(f"  Multi-tag Format: {capabilities['text_processing']['multi_tag_format']}")
    print(f"  Example: {capabilities['text_processing']['example']}")
    print(f"  Supported Sections: {', '.join(capabilities['text_processing']['lyric_sections'])}")
    
    print(f"\n🎧 Audio Alignment:")
    print(f"  Feature Rate: {capabilities['audio_alignment']['feature_rate']}")
    print(f"  Sample Rate: {capabilities['audio_alignment']['sample_rate']}")
    print(f"  Languages: {capabilities['audio_alignment']['languages']}")
    print(f"  Segment Types: {', '.join(capabilities['audio_alignment']['segment_classification'])}")

def demo_multimodal_system():
    """멀티모달 시스템 데모"""
    print("\n" + "=" * 80)
    print("🚀 Multimodal System Demo")
    print("=" * 80)
    
    # 시스템 초기화
    print("\n1. 시스템 초기화...")
    try:
        config = GeneratorConfig(
            d_model=1536,
            n_layers=26,
            latent_channels=16,
            latent_time_steps=128
        )
        
        system = MultimodalLyroSystem(
            generator_config=config,
            training_mode="joint"
        )
        
        print(f"   ✅ 시스템 초기화 완료")
        print(f"   📊 총 파라미터: {sum(p.numel() for p in system.parameters()):,}")
        
    except Exception as e:
        print(f"   ❌ 초기화 실패: {e}")
        return

def demo_text_processing():
    """텍스트 처리 데모"""
    print("\n2. 텍스트 처리 데모...")
    
    # 다양한 가사 형태 테스트
    test_cases = [
        {
            "name": "K-pop Ballad",
            "lyrics": "[verse 1:piano] 사랑이란 무엇인가요 이 마음을 어떻게 표현할까요",
            "style": "k-pop ballad, emotional, piano driven, slow tempo"
        },
        {
            "name": "Rock Anthem", 
            "lyrics": "[chorus:guitar:energetic] 자유를 향해 달려가자 끝없는 꿈을 향해서",
            "style": "rock anthem, powerful, electric guitar, energetic drums"
        },
        {
            "name": "Jazz Instrumental",
            "lyrics": "",  # 악기 연주만
            "style": "jazz fusion, saxophone solo, complex harmony, sophisticated"
        },
        {
            "name": "Acoustic Cover",
            "lyrics": "[bridge:acoustic] 조용한 밤에 홀로 서서 별들을 바라보며",
            "style": "acoustic cover, gentle, stripped down, intimate"
        }
    ]
    
    try:
        from models.encoders import UnifiedTextEncoder
        
        encoder = UnifiedTextEncoder(
            embed_dim=1536,
            hidden_dim=1536,
            num_layers=8,
            num_heads=12,
            max_lyrics_length=256,
            max_style_length=128
        )
        
        # 어휘 구축
        all_lyrics = [case["lyrics"] for case in test_cases]
        all_styles = [case["style"] for case in test_cases]
        encoder.build_vocab_from_data(all_lyrics, all_styles)
        
        print(f"   ✅ 통합 텍스트 인코더 준비 완료")
        print(f"   📚 어휘 크기: {len(encoder.vocab)}")
        
        # 각 케이스 처리
        for i, case in enumerate(test_cases, 1):
            print(f"\n   {i}. {case['name']}:")
            print(f"      가사: {case['lyrics'] if case['lyrics'] else '(instrumental)'}")
            print(f"      스타일: {case['style'][:50]}...")
            
            with torch.no_grad():
                embedding = encoder(
                    lyrics=[case["lyrics"]],
                    style=[case["style"]]
                )[0]
            
            print(f"      임베딩: {embedding.shape}, norm={embedding.norm().item():.3f}")
            
            # 파싱 결과
            if case["lyrics"]:
                lyrics_content, sections, lyric_styles = encoder.parse_lyrics(case["lyrics"])
                print(f"      파싱된 섹션: {sections}")
                if lyric_styles:
                    print(f"      가사 스타일: {lyric_styles}")
        
    except Exception as e:
        print(f"   ❌ 텍스트 처리 실패: {e}")

def demo_alignment_system():
    """정렬 시스템 데모"""
    print("\n3. 오디오-가사 정렬 데모...")
    
    try:
        from models.alignment import AudioLyricsAligner
        
        # 정렬 시스템 초기화
        aligner = AudioLyricsAligner(
            feature_rate=75,
            sample_rate=24000,
            alignment_dim=512,
            max_audio_length=30.0
        )
        
        print(f"   ✅ 정렬 시스템 초기화 완료")
        print(f"   🎧 MERT 사용 가능: {aligner.mert_available}")
        print(f"   🗣️ mHuBERT 사용 가능: {aligner.mhubert_available}")
        
        # 더미 오디오 및 가사 타임스탬프
        audio_duration = 10.0
        sample_rate = 24000
        audio = torch.randn(1, int(audio_duration * sample_rate))
        
        lyrics_timestamps = [
            (0.0, 2.5, "첫 번째 버스 가사"),
            (3.0, 5.5, "코러스 부분 가사"),
            (6.0, 8.5, "두 번째 버스 가사"),
            (9.0, 10.0, "아웃트로")
        ]
        
        print(f"   🎵 오디오 길이: {audio_duration}초")
        print(f"   📝 가사 세그먼트: {len(lyrics_timestamps)}개")
        
        # 정렬 수행
        with torch.no_grad():
            results = aligner(
                audio=audio,
                lyrics_timestamps=lyrics_timestamps,
                return_alignment=True
            )
        
        print(f"\n   📊 정렬 결과:")
        print(f"      오디오 특징: {results['audio_features'].shape}")
        print(f"      음성 특징: {results['speech_features'].shape}")
        print(f"      정렬 점수: {results['alignment_scores'].shape}")
        print(f"      세그먼트 분류: {results['segment_probs'].shape}")
        
        # 세그먼트 분석
        segment_probs = results['segment_probs'][0].cpu().numpy()
        lyrics_confidence = segment_probs[:, 0].mean()
        instrumental_confidence = segment_probs[:, 1].mean()
        silence_confidence = segment_probs[:, 2].mean()
        
        print(f"\n   🎯 세그먼트 분류:")
        print(f"      가사 구간: {lyrics_confidence:.3f}")
        print(f"      악기 구간: {instrumental_confidence:.3f}")
        print(f"      무음 구간: {silence_confidence:.3f}")
        
        alignment_quality = results['alignment_scores'][0].mean().item()
        print(f"      전체 정렬 품질: {alignment_quality:.3f}")
        
    except Exception as e:
        print(f"   ❌ 정렬 시스템 실패: {e}")

def demo_music_generation():
    """음악 생성 데모"""
    print("\n4. 음악 생성 데모...")
    
    generation_scenarios = [
        {
            "name": "감성 발라드",
            "lyrics": "[verse 1:piano] 너와 나 함께 걸어가는 이 길에서",
            "style": "ballad, emotional, piano, slow tempo",
            "task": "SONG"
        },
        {
            "name": "재즈 인스트루멘탈",
            "lyrics": "",
            "style": "jazz fusion, saxophone solo, complex harmony",
            "task": "INST"
        },
        {
            "name": "락 앤쓸",
            "lyrics": "[chorus:guitar] 끝없는 꿈을 향해 달려가자",
            "style": "rock anthem, energetic, electric guitar, powerful",
            "task": "SONG"
        }
    ]
    
    try:
        print(f"   🎼 {len(generation_scenarios)}가지 시나리오 생성...")
        
        for i, scenario in enumerate(generation_scenarios, 1):
            print(f"\n   {i}. {scenario['name']} ({scenario['task']}):")
            print(f"      가사: {scenario['lyrics'] if scenario['lyrics'] else '(instrumental)'}")
            print(f"      스타일: {scenario['style']}")
            
            try:
                # 빠른 생성 (실제로는 멀티모달 시스템 필요)
                result = quick_generate(
                    lyrics=scenario['lyrics'] if scenario['lyrics'] else None,
                    caption=scenario['style'],
                    task=scenario['task'],
                    duration=5.0,
                    quality="fast"
                )
                
                if 'generated_latents' in result:
                    print(f"      ✅ 생성 완료: {result['generated_latents'].shape}")
                    print(f"      ⏱️ 생성 시간: {result['generation_time']:.2f}초")
                    
                    if 'alignment_info' in result:
                        confidence = result['alignment_info'].get('confidence', 0)
                        print(f"      🎯 정렬 신뢰도: {confidence:.3f}")
                else:
                    print(f"      ⚠️ 생성 결과 확인 필요")
                    
            except Exception as e:
                print(f"      ❌ 생성 실패: {e}")
    
    except Exception as e:
        print(f"   ❌ 음악 생성 데모 실패: {e}")

def demo_training_pipeline():
    """훈련 파이프라인 데모"""
    print("\n5. 훈련 파이프라인 구조...")
    
    try:
        config = LyroConfig()
        
        print(f"   📚 훈련 단계:")
        print(f"      1. Pre-training: Generator 기본 훈련 (20 epochs)")
        print(f"      2. Alignment: MERT+mHuBERT 정렬 훈련 (15 epochs)")
        print(f"      3. Joint: 전체 시스템 통합 훈련 (30 epochs)")
        print(f"      4. Fine-tuning: TTS 보조 학습 강화 (10 epochs)")
        
        print(f"\n   ⚙️ 훈련 설정:")
        print(f"      Generator 학습률: {config.generator.learning_rate}")
        print(f"      배치 크기: {config.generator.batch_size}")
        print(f"      그래디언트 클리핑: {config.generator.grad_clip}")
        print(f"      Mixed Precision: {config.generator.mixed_precision}")
        
        print(f"\n   💾 체크포인트:")
        print(f"      저장 디렉토리: {config.training.checkpoint_dir}")
        print(f"      저장 간격: {config.training.save_interval} epochs")
        print(f"      최고 모델 유지: {config.training.keep_best}개")
        
    except Exception as e:
        print(f"   ❌ 훈련 설정 로드 실패: {e}")

def demo_performance_analysis():
    """성능 분석 데모"""
    print("\n6. 성능 분석...")
    
    # 다양한 배치 크기별 메모리 분석
    batch_sizes = [1, 2, 4, 8]
    
    print(f"   📊 메모리 사용량 분석:")
    for batch_size in batch_sizes:
        memory_info = estimate_model_memory(batch_size)
        print(f"      Batch {batch_size}: {memory_info['total_estimated']}")
    
    print(f"\n   🏆 벤치마크 (추정):")
    print(f"      단일 곡 생성: ~10-30초 (RTX 4090)")
    print(f"      배치 생성 (4곡): ~30-90초")
    print(f"      정렬 처리: ~2-5초 (10초 오디오)")
    print(f"      TTS 통합: ~1-3초")
    
    print(f"\n   🎯 품질 지표:")
    print(f"      음악적 일관성: MERT 기반 평가")
    print(f"      가사-오디오 정렬: 신뢰도 점수")
    print(f"      음성 품질: mHuBERT 기반 평가")
    print(f"      크로스 모달 일관성: 코사인 유사도")

if __name__ == "__main__":
    try:
        # 시스템 정보
        demo_system_info()
        
        # 멀티모달 시스템
        demo_multimodal_system()
        
        # 텍스트 처리
        demo_text_processing()
        
        # 정렬 시스템  
        demo_alignment_system()
        
        # 음악 생성
        demo_music_generation()
        
        # 훈련 파이프라인
        demo_training_pipeline()
        
        # 성능 분석
        demo_performance_analysis()
        
        print("\n" + "=" * 80)
        print("🎉 LYRO v2 Complete Demo Finished!")
        print("=" * 80)
        print("\n✨ Ready for:")
        print("   • Audio-lyrics alignment with 147 language support")
        print("   • Multi-conditional music generation (SONG/INST/COVER)")
        print("   • TTS-enhanced training pipeline")
        print("   • Real-time cross-modal attention")
        print("   • Large-scale multimodal learning")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()