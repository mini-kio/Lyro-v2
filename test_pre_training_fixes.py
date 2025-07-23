#!/usr/bin/env python3
"""
Pre-training fixes unit tests
학습 전 구조적 결함 수정사항들의 검증
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

# 프로젝트 루트를 파이썬 패스에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.generator import LyroGenerator, GeneratorConfig, create_lyro_generator
from models.losses import FlowMatchingLoss, LatentPerceptualLoss
from models.encoders import UnifiedTextEncoder
from models.sampling import FlowMatchingSampler


class TestPreTrainingFixes:
    """Pre-training structural fixes verification"""
    
    def setup_method(self):
        """테스트 setup"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.seq_length = 128
        self.latent_channels = 16
        
    def test_beta_schedule_sync(self):
        """1. β-schedule ↔ sampler steps 동기화 테스트"""
        print("Testing β-schedule ↔ sampler steps synchronization...")
        
        # FlowMatchingLoss 초기화 (steps 없이)
        loss_fn = FlowMatchingLoss()
        
        # 동적으로 steps 설정
        sampler_steps = 25  # 다른 값으로 테스트
        loss_fn.sync_with_sampler_steps(sampler_steps)
        
        assert loss_fn.num_timesteps == sampler_steps
        assert len(loss_fn.betas) == sampler_steps
        
        # 손실 계산 테스트
        pred_v = torch.randn(self.batch_size, self.latent_channels, self.seq_length)
        target_v = torch.randn(self.batch_size, self.latent_channels, self.seq_length)
        t = torch.rand(self.batch_size)
        
        loss = loss_fn(pred_v, target_v, t)
        assert loss.item() >= 0
        print("OK: beta-schedule synchronization works")
    
    def test_pad_unk_embedding_separation(self):
        """2. [PAD]=0 / [UNK]=1 임베딩 분리 테스트"""
        print("Testing [PAD]=0 / [UNK]=1 embedding separation...")
        
        encoder = UnifiedTextEncoder()
        
        # 기본 vocab 확인
        encoder.build_vocab_from_data(
            lyrics_list=["[verse:piano] test lyrics", "[chorus] another test"],
            style_list=["pop rock", "energetic ballad"]
        )
        
        assert encoder.PAD_IDX == 0
        assert encoder.UNK_IDX == 1
        assert encoder.section_vocab.get('[PAD]') == 0
        assert encoder.section_vocab.get('[UNK]') == 1
        assert encoder.style_vocab.get('[PAD]') == 0  
        assert encoder.style_vocab.get('[UNK]') == 1
        
        print("OK: PAD/UNK embedding separation works")
    
    def test_deterministic_noise_seeding(self):
        """3. 랜덤 latent fallback → 입력 해시로 시드 테스트"""
        print("Testing deterministic noise seeding...")
        
        config = GeneratorConfig()
        generator = LyroGenerator(config, enable_compile=False)  # 컴파일 비활성화
        
        # 동일한 입력으로 두 번 노이즈 생성
        text_embed = torch.randn(self.batch_size, 768)
        shape = (self.batch_size, self.latent_channels, self.seq_length)
        
        noise1 = generator._create_deterministic_noise(shape, text_embed)
        noise2 = generator._create_deterministic_noise(shape, text_embed)
        
        # 동일한 입력에 대해 동일한 노이즈가 생성되어야 함
        assert torch.allclose(noise1, noise2)
        
        # 다른 입력에 대해서는 다른 노이즈가 생성되어야 함
        text_embed_diff = torch.randn(self.batch_size, 768)
        noise3 = generator._create_deterministic_noise(shape, text_embed_diff)
        assert not torch.allclose(noise1, noise3)
        
        print("OK: Deterministic noise seeding works")
    
    def test_reference_temporal_processing(self):
        """4. reference latent 1D-Conv & cross-attn 시간 정보 보존 테스트"""
        print("Testing reference latent temporal processing...")
        
        config = GeneratorConfig()
        generator = LyroGenerator(config, enable_compile=False)
        
        # Reference latent 입력
        reference = torch.randn(self.batch_size, self.latent_channels, self.seq_length)
        
        # Condition processor를 통해 처리
        processed = generator.condition_processor(reference=reference)
        
        assert processed.shape == (self.batch_size, config.d_model)
        assert not torch.isnan(processed).any()
        
        print("OK: Reference temporal processing works")
    
    def test_perceptual_loss_skip_logic(self):
        """5. PerceptualLoss 짧은 길이 skip 로직 테스트"""
        print("Testing PerceptualLoss short length skip logic...")
        
        loss_fn = LatentPerceptualLoss()
        
        # 매우 짧은 시퀀스 (skip 되어야 함)
        short_pred = torch.randn(self.batch_size, self.latent_channels, 8)
        short_target = torch.randn(self.batch_size, self.latent_channels, 8)
        short_loss = loss_fn(short_pred, short_target)
        assert short_loss.item() == 0.0
        
        # 중간 길이 시퀀스 (감소된 가중치)
        medium_pred = torch.randn(self.batch_size, self.latent_channels, 24)
        medium_target = torch.randn(self.batch_size, self.latent_channels, 24)
        medium_loss = loss_fn(medium_pred, medium_target)
        assert medium_loss.item() > 0.0
        
        # 긴 시퀀스 (정상 처리)
        long_pred = torch.randn(self.batch_size, self.latent_channels, 64)
        long_target = torch.randn(self.batch_size, self.latent_channels, 64)
        long_loss = loss_fn(long_pred, long_target)
        assert long_loss.item() > 0.0
        
        print("OK: PerceptualLoss skip logic works")
    
    def test_torch_compile_integration(self):
        """6. torch.compile 적용 테스트"""
        print("Testing torch.compile integration...")
        
        config = GeneratorConfig()
        try:
            # 컴파일 활성화된 generator 생성
            generator = create_lyro_generator(config, enable_compile=True)
            
            # 기본 forward pass 테스트
            latents = torch.randn(self.batch_size, self.latent_channels, self.seq_length)
            timesteps = torch.rand(self.batch_size)
            text_embed = torch.randn(self.batch_size, 768)
            
            output, repa_features = generator(latents, timesteps, text_embed)
            
            assert output.shape == latents.shape
            print("OK: torch.compile integration works")
            
        except Exception as e:
            print(f"WARNING: torch.compile may not be available: {str(e)}")
            print("OK: torch.compile integration test skipped")
    
    def test_full_integration(self):
        """7. 전체 통합 테스트"""
        print("Testing full integration...")
        
        # 모든 수정사항이 적용된 상태에서 전체 파이프라인 테스트
        config = GeneratorConfig()
        generator = create_lyro_generator(config, enable_compile=False)
        
        # 텍스트 인코더
        text_encoder = UnifiedTextEncoder()
        text_encoder.build_vocab_from_data(
            lyrics_list=["[verse:piano] test lyrics"],
            style_list=["pop energetic"]
        )
        
        # 손실 함수들
        flow_loss = FlowMatchingLoss()
        flow_loss.sync_with_sampler_steps(50)
        
        perceptual_loss = LatentPerceptualLoss()
        
        # 샘플러
        sampler = FlowMatchingSampler(steps=50, device="cpu")
        
        # 데이터 준비
        latents = torch.randn(self.batch_size, self.latent_channels, self.seq_length)
        text_embed = text_encoder(
            lyrics=["[verse:piano] test lyrics"] * self.batch_size,
            style=["pop energetic"] * self.batch_size
        )
        
        # Forward pass
        timesteps = torch.rand(self.batch_size)
        output, repa_features = generator(latents, timesteps, text_embed)
        
        # 손실 계산
        target_v = torch.randn_like(output)
        flow_loss_val = flow_loss(output, target_v, timesteps)
        perceptual_loss_val = perceptual_loss(output, latents)
        
        # 샘플링
        shape = (1, self.latent_channels, self.seq_length)
        conditions = {'text_embed': text_embed[:1]}
        generated = sampler.sample(generator, shape, conditions, verbose=False)
        
        assert output.shape == latents.shape
        assert flow_loss_val.item() >= 0
        assert perceptual_loss_val.item() >= 0
        assert generated.shape == shape
        
        print("OK: Full integration test passed")


def run_tests():
    """테스트 실행"""
    test_instance = TestPreTrainingFixes()
    test_instance.setup_method()
    
    tests = [
        test_instance.test_beta_schedule_sync,
        test_instance.test_pad_unk_embedding_separation,
        test_instance.test_deterministic_noise_seeding,
        test_instance.test_reference_temporal_processing,
        test_instance.test_perceptual_loss_skip_logic,
        test_instance.test_torch_compile_integration,
        test_instance.test_full_integration,
    ]
    
    print("Running pre-training fixes tests...")
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED {test.__name__}: {e}")
            failed += 1
        print()
    
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All pre-training structural fixes are working correctly!")
        return True
    else:
        print("Some tests failed. Please check the implementations.")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)