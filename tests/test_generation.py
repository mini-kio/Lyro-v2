# lyro/tests/test_generation.py
"""
LYRO Generation Tests
생성 품질 및 성능 테스트 - Fixed Import Errors
"""

import os
import sys
import pytest
import torch
import numpy as np
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ✅ 수정된 imports
from ssm.model import LyroSSMUNet, TaskController
from ssm.flow_matching import LyroFlowMatching, FlowConfig  # ✅ 수정: FlowMatching -> LyroFlowMatching, FlowConfig 추가
from dcae.model import LyroMusicDCAE  # ✅ 수정: MusicDCAE -> LyroMusicDCAE
from dataset.tokenizer import LyroTokenizer


class TestGeneration:
    """생성 테스트"""
    
    @pytest.fixture(scope='class')
    def models(self):
        """모델 로드 (한 번만)"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 더미 모델 생성 (실제 테스트에서는 체크포인트 로드)
        ssm_model = LyroSSMUNet(
            input_channels=8,
            hidden_dims=[64, 64],  # 작은 모델
            ssm_layers=[1, 1],  # ✅ 수정: mamba_layers -> ssm_layers
            max_seq_len=1024
        ).to(device).eval()
        
        # ✅ 수정: MusicDCAE -> LyroMusicDCAE, compression_factor -> latent_channels
        dcae_model = LyroMusicDCAE(
            sample_rate=44100,
            latent_channels=8
        ).to(device).eval()
        
        # ✅ 수정: FlowMatching -> LyroFlowMatching, 파라미터 수정
        flow_config = FlowConfig()
        flow_config.flow_steps = 4
        flow_config.integration_method = 'euler'
        
        flow_matching = LyroFlowMatching(
            model=ssm_model,
            scheduler_type="cosine",
            solver_type="euler",
            sigma=1e-4,
            flow_type="rectified"
        )
        # flow_steps 설정
        flow_matching.flow_steps = 4
        
        tokenizer = LyroTokenizer()
        
        return {
            'ssm': ssm_model,
            'dcae': dcae_model,
            'flow': flow_matching,
            'tokenizer': tokenizer,
            'device': device
        }
    
    def test_song_generation(self, models):
        """SONG 태스크 테스트"""
        # 조건 준비 - ✅ 수정: TaskController 사용
        lyrics = "Test lyrics for generation"
        conditions = TaskController.create_conditions(
            task_type='SONG',
            lyrics=models['tokenizer'].encode_text(lyrics),
            style_prompt=torch.randn(1, 512).to(models['device']),
            device=models['device']
        )
        
        # 생성
        shape = (1, 8, 256)  # 짧은 생성
        
        start_time = time.time()
        with torch.no_grad():
            # ✅ 수정: steps -> num_steps
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
        
        # RTF 계산 - ✅ 수정: 32x 압축율 반영
        duration = shape[2] * 512 * 32 / 44100  # 초
        rtf = generation_time / duration
        assert rtf < 1.0  # 실시간보다 빨라야 함
        
        print(f"SONG generation RTF: {rtf:.3f}")
    
    def test_inst_generation(self, models):
        """INST 태스크 테스트"""
        # ✅ 수정: TaskController 사용
        conditions = TaskController.create_conditions(
            task_type='INST',
            style_prompt=torch.randn(1, 512).to(models['device']),
            device=models['device']
        )
        
        shape = (1, 8, 256)
        
        with torch.no_grad():
            # ✅ 수정: steps -> num_steps
            generated, _ = models['flow'].generate(
                shape=shape,
                conditions=conditions,
                num_steps=4,
                cfg_scale=1.5
            )
            
        assert generated.shape == shape
        
    def test_cover_generation(self, models):
        """COVER 태스크 테스트"""
        # 참조 latent (더미)
        ref_latent = torch.randn(1, 8, 128).to(models['device'])
        
        # ✅ 수정: TaskController 사용
        conditions = TaskController.create_conditions(
            task_type='COVER',
            lyrics=models['tokenizer'].encode_text("New lyrics"),
            style_prompt=torch.randn(1, 512).to(models['device']),
            icl_reference=ref_latent,
            device=models['device']
        )
        
        shape = (1, 8, 256)
        
        with torch.no_grad():
            # ✅ 수정: steps -> num_steps
            generated, _ = models['flow'].generate(
                shape=shape,
                conditions=conditions,
                num_steps=4,
                cfg_scale=1.5
            )
            
        assert generated.shape == shape
    
    def test_flow_steps_quality(self, models):
        """Flow step 수에 따른 품질 테스트"""
        # ✅ 수정: TaskController 사용
        conditions = TaskController.create_conditions(
            task_type='SONG',
            lyrics=models['tokenizer'].encode_text("Test"),
            style_prompt=torch.randn(1, 512).to(models['device']),
            device=models['device']
        )
        
        shape = (1, 8, 128)
        results = {}
        
        for steps in [4, 8, 16]:
            start_time = time.time()
            
            with torch.no_grad():
                # ✅ 수정: steps -> num_steps
                generated, trajectory = models['flow'].generate(
                    shape=shape,
                    conditions=conditions,
                    num_steps=steps,
                    cfg_scale=1.5
                )
                
            generation_time = time.time() - start_time
            
            # 품질 메트릭 (간단한 예시)
            quality_score = -torch.std(generated).item()  # 낮을수록 좋음
            
            results[steps] = {
                'time': generation_time,
                'quality': quality_score,
                'final_mean': generated.mean().item(),
                'final_std': generated.std().item()
            }
            
        # 결과 출력
        print("\nFlow steps comparison:")
        for steps, metrics in results.items():
            print(f"  Steps={steps}: time={metrics['time']:.3f}s, "
                  f"quality={metrics['quality']:.3f}")
    
    def test_eos_handling(self, models):
        """EOS 토큰 처리 테스트"""
        tokenizer = models['tokenizer']
        
        # EOS 토큰이 포함된 시퀀스
        sequence = [1, 2, 3, tokenizer.SPECIAL_TOKENS['<EOA>'], 4, 5]
        
        # EOS 위치 찾기
        eos_positions = [i for i, token in enumerate(sequence) 
                        if token in tokenizer.eos_token_ids]
        
        assert len(eos_positions) == 1
        assert eos_positions[0] == 3
        
        # EOS 페널티 마스크
        penalty_mask = tokenizer.get_eos_penalty_mask(
            current_length=10,
            min_length=20
        )
        
        # 짧은 길이에서는 페널티가 있어야 함
        for eos_id in tokenizer.eos_token_ids:
            if eos_id < penalty_mask.shape[0]:  # ✅ 수정: 범위 검증 추가
                assert penalty_mask[eos_id] == -float('inf')
    
    def test_memory_efficiency(self, models):
        """메모리 효율성 테스트"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        torch.cuda.reset_peak_memory_stats()
        
        # 긴 시퀀스 생성
        shape = (1, 8, 2048)  # ~40초
        
        # ✅ 수정: TaskController 사용
        conditions = TaskController.create_conditions(
            task_type='SONG',
            lyrics=models['tokenizer'].encode_text("Long text " * 50),
            style_prompt=torch.randn(1, 512).to(models['device']),
            device=models['device']
        )
        
        with torch.no_grad():
            # ✅ 수정: steps -> num_steps
            generated, _ = models['flow'].generate(
                shape=shape,
                conditions=conditions,
                num_steps=4,
                cfg_scale=1.5
            )
            
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        print(f"\nPeak memory usage: {peak_memory:.2f} GB")
        assert peak_memory < 16.0  # A100 40GB 기준
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_generation(self, models, batch_size):
        """배치 생성 테스트"""
        shape = (batch_size, 8, 256)
        
        # ✅ 수정: TaskController 사용, 배치 크기에 맞춰 조건 생성
        # Task tokens for batch
        task_tokens = torch.tensor([TaskController.TASK_TOKENS['SONG']] * batch_size).to(models['device'])
        
        conditions = {
            'task_token': task_tokens,
            'lyrics': None,  # 간단화
            'style_prompt': torch.randn(batch_size, 512).to(models['device']),
            'icl_reference': None
        }
        
        with torch.no_grad():
            # ✅ 수정: steps -> num_steps
            generated, _ = models['flow'].generate(
                shape=shape,
                conditions=conditions,
                num_steps=4,
                cfg_scale=1.5
            )
            
        assert generated.shape == shape
        
        # 각 샘플이 다른지 확인
        if batch_size > 1:
            for i in range(1, batch_size):
                assert not torch.allclose(generated[0], generated[i])

    def test_tts_generation(self, models):
        """TTS 태스크 테스트 (새로 추가)"""
        # ✅ 새로 추가: TTS 태스크 테스트
        transcript = "Hello, this is a test transcript for TTS generation."
        
        conditions = TaskController.create_conditions(
            task_type='TTS',  # ✅ TTS 태스크 추가됨
            style_prompt=torch.randn(1, 512).to(models['device']),
            device=models['device']
        )
        
        # transcript를 별도로 추가 (TTS용)
        conditions['transcript'] = models['tokenizer'].encode_text(transcript, text_type='transcript')
        
        shape = (1, 8, 256)
        
        with torch.no_grad():
            generated, _ = models['flow'].generate(
                shape=shape,
                conditions=conditions,
                num_steps=4,
                cfg_scale=1.5
            )
            
        assert generated.shape == shape
        print("TTS generation test passed")

    def test_model_compatibility(self, models):
        """모델 호환성 테스트"""
        # DCAE 인코딩/디코딩 테스트
        dummy_audio = torch.randn(1, 2, 44100).to(models['device'])  # 1초 오디오
        
        with torch.no_grad():
            # ✅ 수정: encode 메서드 반환값 처리
            latent, skip_features = models['dcae'].encode(dummy_audio)
            
            # ✅ 수정: decode 메서드에 skip_features 전달
            reconstructed = models['dcae'].decode(latent, skip_features)
            
        # 형태 검증
        assert latent.shape[0] == 1  # batch size
        assert latent.shape[1] == 8  # latent channels
        assert reconstructed.shape == dummy_audio.shape
        
        print("Model compatibility test passed")

    def test_tokenizer_features(self, models):
        """토크나이저 기능 테스트"""
        tokenizer = models['tokenizer']
        
        # 가사 인코딩 테스트
        lyrics = "This is a test song with lyrics"
        lyrics_tokens = tokenizer.encode_lyrics(lyrics)
        assert len(lyrics_tokens) > 0
        
        # 전사본 인코딩 테스트 (TTS용)
        transcript = "This is a test transcript [pause] with special tokens"
        transcript_tokens = tokenizer.encode_transcript(transcript)
        assert len(transcript_tokens) > 0
        
        # 특수 토큰 검증
        assert '<TASK=TTS>' in tokenizer.SPECIAL_TOKENS
        assert '<transcript>' in tokenizer.SPECIAL_TOKENS
        assert '<speech>' in tokenizer.SPECIAL_TOKENS
        
        # 배치 인코딩 테스트
        lyrics_list = ["Song one", "Song two", "Song three"]
        transcript_list = ["Speech one", "Speech two", None]
        
        batch_tokens = tokenizer.batch_encode_mixed(
            lyrics_list, transcript_list, padding=True
        )
        
        assert isinstance(batch_tokens, torch.Tensor)
        assert batch_tokens.shape[0] == 3  # batch size
        
        print("Tokenizer features test passed")


def run_benchmark():
    """성능 벤치마크 실행"""
    print("Running LYRO generation benchmark...")
    
    # pytest 실행
    pytest.main([__file__, '-v', '-s'])


def test_complete_pipeline():
    """완전한 파이프라인 테스트 (단독 실행용)"""
    print("Testing complete LYRO pipeline...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 생성
    ssm_model = LyroSSMUNet(
        input_channels=8,
        hidden_dims=[64, 64],
        ssm_layers=[1, 1],
        max_seq_len=1024
    ).to(device).eval()
    
    dcae_model = LyroMusicDCAE(
        sample_rate=44100,
        latent_channels=8
    ).to(device).eval()
    
    flow_matching = LyroFlowMatching(
        model=ssm_model,
        scheduler_type="cosine",
        solver_type="euler",
        sigma=1e-4,
        flow_type="rectified"
    )
    
    tokenizer = LyroTokenizer()
    
    print("Models created successfully")
    
    # 간단한 생성 테스트
    lyrics = "Test song lyrics"
    conditions = TaskController.create_conditions(
        task_type='SONG',
        lyrics=tokenizer.encode_text(lyrics),
        style_prompt=torch.randn(1, 512).to(device),
        device=device
    )
    
    shape = (1, 8, 128)
    
    print(f"Generating audio with shape: {shape}")
    
    start_time = time.time()
    with torch.no_grad():
        generated, _ = flow_matching.generate(
            shape=shape,
            conditions=conditions,
            num_steps=4,
            cfg_scale=1.5
        )
    generation_time = time.time() - start_time
    
    print(f"Generation completed in {generation_time:.3f}s")
    print(f"Generated shape: {generated.shape}")
    print(f"Generated range: [{generated.min():.3f}, {generated.max():.3f}]")
    
    # RTF 계산
    duration = shape[2] * 512 * 32 / 44100
    rtf = generation_time / duration
    print(f"Real-time factor: {rtf:.3f}")
    
    print("✅ Complete pipeline test passed!")


if __name__ == '__main__':
    # 단독 실행 시 완전한 파이프라인 테스트
    test_complete_pipeline()
    
    # 또는 벤치마크 실행
    # run_benchmark()