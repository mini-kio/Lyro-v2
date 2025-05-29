# lyro/tests/test_generation.py
"""
LYRO Generation Tests
생성 품질 및 성능 테스트
"""

import os
import sys
import pytest
import torch
import numpy as np
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ssm.model import LyroSSMUNet
from ssm.flow_matching import FlowMatching
from dcae.model import MusicDCAE
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
            mamba_layers=[1, 1],
            max_seq_len=1024
        ).to(device).eval()
        
        dcae_model = MusicDCAE().to(device).eval()
        
        flow_matching = FlowMatching(
            model=ssm_model,
            flow_steps=4,
            integration_method='euler'
        )
        
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
        # 조건 준비
        lyrics = "Test lyrics for generation"
        conditions = {
            'task_token': torch.tensor([0]).to(models['device']),  # SONG
            'lyrics': models['tokenizer'].encode_text(lyrics),
            'style_prompt': torch.randn(1, 512).to(models['device'])
        }
        
        # 생성
        shape = (1, 8, 256)  # 짧은 생성
        
        start_time = time.time()
        with torch.no_grad():
            generated, _ = models['flow'].generate(
                shape=shape,
                conditions=conditions,
                steps=4
            )
        generation_time = time.time() - start_time
        
        # 검증
        assert generated.shape == shape
        assert not torch.isnan(generated).any()
        assert not torch.isinf(generated).any()
        
        # RTF 계산
        duration = shape[2] * 512 / 44100  # 초
        rtf = generation_time / duration
        assert rtf < 1.0  # 실시간보다 빨라야 함
        
        print(f"SONG generation RTF: {rtf:.3f}")
    
    def test_inst_generation(self, models):
        """INST 태스크 테스트"""
        conditions = {
            'task_token': torch.tensor([1]).to(models['device']),  # INST
            'lyrics': None,
            'style_prompt': torch.randn(1, 512).to(models['device'])
        }
        
        shape = (1, 8, 256)
        
        with torch.no_grad():
            generated, _ = models['flow'].generate(
                shape=shape,
                conditions=conditions,
                steps=4
            )
            
        assert generated.shape == shape
        
    def test_cover_generation(self, models):
        """COVER 태스크 테스트"""
        # 참조 latent (더미)
        ref_latent = torch.randn(1, 8, 128).to(models['device'])
        
        conditions = {
            'task_token': torch.tensor([2]).to(models['device']),  # COVER
            'lyrics': models['tokenizer'].encode_text("New lyrics"),
            'style_prompt': torch.randn(1, 512).to(models['device']),
            'icl_reference': ref_latent
        }
        
        shape = (1, 8, 256)
        
        with torch.no_grad():
            generated, _ = models['flow'].generate(
                shape=shape,
                conditions=conditions,
                steps=4
            )
            
        assert generated.shape == shape
    
    def test_flow_steps_quality(self, models):
        """Flow step 수에 따른 품질 테스트"""
        conditions = {
            'task_token': torch.tensor([0]).to(models['device']),
            'lyrics': models['tokenizer'].encode_text("Test"),
            'style_prompt': torch.randn(1, 512).to(models['device'])
        }
        
        shape = (1, 8, 128)
        results = {}
        
        for steps in [4, 8, 16]:
            start_time = time.time()
            
            # Flow steps 변경
            models['flow'].flow_steps = steps
            
            with torch.no_grad():
                generated, trajectory = models['flow'].generate(
                    shape=shape,
                    conditions=conditions,
                    steps=steps
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
            assert penalty_mask[eos_id] == -float('inf')
    
    def test_memory_efficiency(self, models):
        """메모리 효율성 테스트"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        torch.cuda.reset_peak_memory_stats()
        
        # 긴 시퀀스 생성
        shape = (1, 8, 2048)  # ~40초
        conditions = {
            'task_token': torch.tensor([0]).to(models['device']),
            'lyrics': models['tokenizer'].encode_text("Long text " * 50),
            'style_prompt': torch.randn(1, 512).to(models['device'])
        }
        
        with torch.no_grad():
            generated, _ = models['flow'].generate(
                shape=shape,
                conditions=conditions,
                steps=4
            )
            
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        print(f"\nPeak memory usage: {peak_memory:.2f} GB")
        assert peak_memory < 16.0  # A100 40GB 기준
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batch_generation(self, models, batch_size):
        """배치 생성 테스트"""
        shape = (batch_size, 8, 256)
        
        conditions = {
            'task_token': torch.tensor([0] * batch_size).to(models['device']),
            'lyrics': None,  # 간단화
            'style_prompt': torch.randn(batch_size, 512).to(models['device'])
        }
        
        with torch.no_grad():
            generated, _ = models['flow'].generate(
                shape=shape,
                conditions=conditions,
                steps=4
            )
            
        assert generated.shape == shape
        
        # 각 샘플이 다른지 확인
        if batch_size > 1:
            for i in range(1, batch_size):
                assert not torch.allclose(generated[0], generated[i])


def run_benchmark():
    """성능 벤치마크 실행"""
    print("Running LYRO generation benchmark...")
    
    # pytest 실행
    pytest.main([__file__, '-v', '-s'])


if __name__ == '__main__':
    run_benchmark()