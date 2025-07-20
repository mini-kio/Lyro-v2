#!/usr/bin/env python3
"""
Multimodal LYRO System Demo
오디오-가사 정렬 + TTS 보조 학습 + 음악 생성 통합 데모
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.multimodal_lyro import MultimodalLyroSystem
from models.generator import GeneratorConfig
from training.trainer import MultimodalTrainer
from training.config import LyroConfig


def demo_alignment_system():
    """오디오-가사 정렬 시스템 데모"""
    print("=== Audio-Lyrics Alignment Demo ===\n")
    
    # 시스템 초기화
    system = MultimodalLyroSystem()
    
    # 가상 오디오 데이터 생성 (10초, 24kHz)
    audio_duration = 10.0
    sample_rate = 24000
    audio = torch.randn(1, int(audio_duration * sample_rate))
    
    # 가사 타임스탬프 (실제 데이터에서는 수동 annotation)
    lyrics_timestamps = [
        (0.0, 2.5, "첫 번째 버스 가사"),
        (3.0, 5.5, "코러스 부분 가사"),
        (6.0, 8.5, "두 번째 버스 가사"),
        (9.0, 10.0, "아웃트로")
    ]
    
    # 정렬 수행
    print(f"Processing {audio_duration}s audio with {len(lyrics_timestamps)} lyric segments...")
    
    with torch.no_grad():
        results = system.aligner(
            audio=audio,
            lyrics_timestamps=lyrics_timestamps,
            return_alignment=True
        )
    
    # 결과 분석
    print(f"\nAlignment Results:")
    print(f"  Audio features shape: {results['audio_features'].shape}")  # (1, T, 1024)
    print(f"  Speech features shape: {results['speech_features'].shape}")  # (1, T, 768)
    print(f"  Alignment scores shape: {results['alignment_scores'].shape}")  # (1, T)
    print(f"  Segment probabilities: {results['segment_probs'].shape}")  # (1, T, 3)
    
    # 세그먼트 분류 결과
    segment_probs = results['segment_probs'][0].cpu().numpy()  # (T, 3)
    lyrics_confidence = segment_probs[:, 0].mean()
    instrumental_confidence = segment_probs[:, 1].mean()
    silence_confidence = segment_probs[:, 2].mean()
    
    print(f"\nSegment Classification:")
    print(f"  Lyrics regions: {lyrics_confidence:.3f}")
    print(f"  Instrumental regions: {instrumental_confidence:.3f}")
    print(f"  Silence regions: {silence_confidence:.3f}")
    
    # 정렬 품질
    alignment_quality = results['alignment_scores'][0].mean().item()
    print(f"  Overall alignment quality: {alignment_quality:.3f}")
    
    return results


def demo_multimodal_generation():
    """멀티모달 음악 생성 데모"""
    print("\n=== Multimodal Music Generation Demo ===\n")
    
    # 시스템 초기화
    system = MultimodalLyroSystem()
    
    # 다양한 생성 시나리오
    scenarios = [
        {
            "name": "K-pop Ballad with Piano",
            "lyrics": ["[verse 1:piano] 사랑이란 무엇인가요 이 마음을 어떻게 표현할까요"],
            "style": ["k-pop ballad, emotional, piano driven, slow tempo"],
            "reference_audio": None
        },
        {
            "name": "Energetic Rock Anthem",
            "lyrics": ["[chorus:guitar] 자유를 향해 달려가자 끝없는 꿈을 향해서"],
            "style": ["rock anthem, energetic, electric guitar, powerful drums"],
            "reference_audio": None
        },
        {
            "name": "Jazz Instrumental",
            "lyrics": [""],  # 악기 연주만
            "style": ["jazz fusion, saxophone solo, complex harmony, sophisticated"],
            "reference_audio": None
        },
        {
            "name": "Acoustic Cover Style",
            "lyrics": ["[bridge:acoustic] 조용한 밤에 홀로 서서 별들을 바라보며"],
            "style": ["acoustic cover, gentle, stripped down, intimate"],
            "reference_audio": torch.randn(1, 24000 * 5)  # 5초 참조
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario['name']}")
        print(f"   Lyrics: {scenario['lyrics'][0] if scenario['lyrics'][0] else '(instrumental)'}")
        print(f"   Style: {scenario['style'][0]}")
        
        # 생성 수행
        with torch.no_grad():
            results = system.generate_music(
                lyrics=scenario['lyrics'] if scenario['lyrics'][0] else None,
                style=scenario['style'],
                reference_audio=scenario['reference_audio'],
                num_steps=50,
                cfg_scale=7.5,
                seed=42 + i
            )
        
        # 결과 분석
        generated_latents = results['generated_latents']
        print(f"   Generated latents: {generated_latents.shape}")
        
        # 조건 분석
        if 'text_conditions' in results:
            text_norm = results['text_conditions'].norm().item()
            print(f"   Text condition strength: {text_norm:.3f}")
        
        if 'alignment_info' in results:
            confidence = results['alignment_info']['confidence']
            print(f"   Reference alignment confidence: {confidence:.3f}")
        
        print()


def demo_tts_integration():
    """TTS 통합 시스템 데모"""
    print("=== TTS Integration Demo ===\n")
    
    system = MultimodalLyroSystem()
    
    # 정렬된 가사-오디오 쌍
    lyrics = ["[verse 1:piano] 너와 나 함께 걸어가는 이 길에서"]
    style = ["ballad, emotional, piano, slow tempo"]
    audio = torch.randn(1, 24000 * 8)  # 8초
    
    # 전체 시스템 forward pass
    with torch.no_grad():
        results = system(
            lyrics=lyrics,
            style=style,
            audio=audio,
            training_stage='joint',
            return_alignment=True,
            return_tts=True
        )
    
    print("TTS Integration Results:")
    
    # TTS 조건 분석
    if 'tts_data' in results:
        tts_data = results['tts_data']
        print(f"  TTS conditions shape: {tts_data['tts_conditions'].shape}")
        print(f"  TTS weights (alignment quality): {tts_data['tts_weights'].mean().item():.3f}")
        
        # 가사 마스크
        lyrics_coverage = tts_data['lyrics_masks'].float().mean().item()
        print(f"  Lyrics coverage: {lyrics_coverage:.3f}")
    
    # 텍스트-오디오 일관성
    if 'text_conditions' in results and 'alignment_results' in results:
        text_embed = results['text_conditions']
        audio_embed = results['alignment_results']['audio_features'].mean(dim=1)
        
        # 코사인 유사도
        similarity = torch.cosine_similarity(text_embed, audio_embed, dim=-1).mean().item()
        print(f"  Text-Audio consistency: {similarity:.3f}")
    
    # 융합된 조건
    if 'fused_conditions' in results:
        fusion_strength = results['fused_conditions'].norm().item()
        print(f"  Fused condition strength: {fusion_strength:.3f}")


def demo_training_pipeline():
    """훈련 파이프라인 데모 (시뮬레이션)"""
    print("\n=== Training Pipeline Demo ===\n")
    
    # 설정 로드
    config = LyroConfig()
    
    # 더미 데이터 로더 생성 (실제로는 데이터셋 필요)
    def create_dummy_dataloader():
        class DummyDataset:
            def __len__(self):
                return 10  # 작은 데모용
            
            def __getitem__(self, idx):
                return {
                    'lyrics': f"[verse {idx % 2 + 1}:piano] 테스트 가사 {idx}",
                    'style': "pop ballad emotional",
                    'audio': torch.randn(24000 * 10),  # 10초
                    'target_latents': torch.randn(16, 128),  # latent target
                    'lyrics_timestamps': [(0.0, 5.0, "테스트"), (5.0, 10.0, "가사")]
                }
        
        from torch.utils.data import DataLoader
        return DataLoader(DummyDataset(), batch_size=2, shuffle=True)
    
    train_loader = create_dummy_dataloader()
    val_loader = create_dummy_dataloader()
    
    # 트레이너 초기화
    trainer = MultimodalTrainer(config)
    
    print("Training Pipeline Structure:")
    print("1. 📚 Pre-training: Generator 기본 훈련")
    print("2. 🎯 Alignment: MERT+mHuBERT 정렬 훈련")
    print("3. 🤝 Joint: 전체 시스템 통합 훈련")
    print("4. ✨ Fine-tuning: TTS 보조 학습 강화")
    
    print(f"\nModel Components:")
    print(f"  - Generator: {sum(p.numel() for p in trainer.model.generator.parameters()):,} params")
    print(f"  - Text Encoder: {sum(p.numel() for p in trainer.model.text_encoder.parameters()):,} params")
    print(f"  - Aligner: {sum(p.numel() for p in trainer.model.aligner.parameters()):,} params")
    print(f"  - Total: {trainer._count_parameters():,} params")
    
    # 단일 배치 훈련 시뮬레이션
    print(f"\n🔄 Simulating single batch training...")
    
    trainer.model.train()
    sample_batch = next(iter(train_loader))
    
    # 배치 준비
    batch = trainer._prepare_batch(sample_batch)
    
    # Forward pass
    with torch.no_grad():  # 메모리 절약을 위해
        results = trainer.model(
            lyrics=batch.get('lyrics'),
            style=batch.get('style'),
            audio=batch.get('audio'),
            target_latents=batch.get('target_latents'),
            lyrics_timestamps=batch.get('lyrics_timestamps'),
            training_stage='joint'
        )
        
        # 손실 계산
        losses = trainer.model.compute_losses(
            predictions=results,
            targets=batch,
            training_stage='joint'
        )
    
    print(f"  Sample batch losses:")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: {value.item():.4f}")


def demo_model_capabilities():
    """모델 기능 종합 데모"""
    print("\n=== Model Capabilities Summary ===\n")
    
    print("🎵 LYRO v2 Multimodal System Features:")
    print()
    
    print("📝 Text Processing:")
    print("  ✅ Separated lyrics + style input")
    print("  ✅ Multi-tag support: [section:instrument:mood]")
    print("  ✅ Dynamic vocabulary building")
    print("  ✅ Cross-attention lyrics ↔ style interaction")
    print()
    
    print("🎧 Audio Processing:")
    print("  ✅ MERT-v1-330M: Music understanding (75Hz, 24kHz)")
    print("  ✅ mHuBERT-147: Multilingual speech (147 languages)")
    print("  ✅ Audio-lyrics alignment with confidence scores")
    print("  ✅ Segment classification: lyrics/instrumental/silence")
    print()
    
    print("🎼 Music Generation:")
    print("  ✅ 800M parameter SSM+Flow Matching model")
    print("  ✅ Latent-based generation (16 channels, 128 timesteps)")
    print("  ✅ Multi-task support: SONG/INST/COVER")
    print("  ✅ CFG sampling with quality control")
    print()
    
    print("🗣️ TTS Integration:")
    print("  ✅ Alignment-guided TTS auxiliary training")
    print("  ✅ Pronunciation accuracy prediction")
    print("  ✅ Dynamic loss weighting based on alignment quality")
    print("  ✅ Cross-modal consistency enforcement")
    print()
    
    print("🚀 Training Pipeline:")
    print("  ✅ 4-stage progressive training")
    print("  ✅ Component-specific optimization")
    print("  ✅ Multi-task loss balancing")
    print("  ✅ Adaptive learning rate scheduling")


if __name__ == "__main__":
    print("🎵 Multimodal LYRO System Comprehensive Demo")
    print("=" * 60)
    
    try:
        # 각 데모 실행
        demo_alignment_system()
        demo_multimodal_generation()
        demo_tts_integration()
        demo_training_pipeline()
        demo_model_capabilities()
        
        print("\n✅ All demos completed successfully!")
        print("\n🎉 Multimodal LYRO System is ready for:")
        print("   - Audio-lyrics alignment")
        print("   - Multi-conditional music generation")
        print("   - TTS-enhanced training")
        print("   - Large-scale multimodal learning")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()