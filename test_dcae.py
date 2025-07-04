#!/usr/bin/env python3
"""
LYRO DCAE + Vocoder 파이프라인 테스트 스크립트
올바른 파이프라인: 오디오 -> 멜 -> DCAE latent -> 멜 -> Vocoder -> 오디오
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
import time
import argparse
import json

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dcae import create_dcae_model
from models.generator import create_lyro_generator, GeneratorConfig
from utils.audio import AudioProcessor, AudioConfig
from utils.metrics import MetricCalculator


class DCPipelineTester:
    """DCAE + Vocoder 파이프라인 테스터"""
    
    def __init__(
        self,
        dcae_model_name: str = "ACE-Step/ACE-Step-v1-3.5B",
        cache_dir: str = "checkpoints",
        device: str = "auto"
    ):
        # 디바이스 설정
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        
        # 모델 로드 (DCAE + Vocoder 통합)
        print("📥 Loading DCAE model with integrated Vocoder...")
        self.dcae_model = create_dcae_model(
            model_name=dcae_model_name,
            cache_dir=cache_dir
        ).to(self.device).eval()
        
        # 유틸리티 초기화
        self.audio_processor = AudioProcessor(AudioConfig(sample_rate=44100))
        self.metric_calculator = MetricCalculator(44100)
        
        print("✅ Models loaded successfully!")
        print(f"   DCAE: {type(self.dcae_model).__name__}")
        print(f"   Vocoder: {'Integrated' if hasattr(self.dcae_model, 'vocoder') and self.dcae_model.vocoder is not None else 'Fallback'}")
    
    def process_audio_with_chunking(self, audio: torch.Tensor, chunk_duration: float = 10.0, overlap: float = 1.0) -> torch.Tensor:
        """
        긴 오디오를 청킹하여 처리 (오버랩 포함)
        
        Args:
            audio: (B, C, T) 입력 오디오
            chunk_duration: 청크 길이 (초)
            overlap: 오버랩 길이 (초)
            
        Returns:
            reconstructed: (B, C, T) 재구성된 오디오
        """
        sample_rate = 44100
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        
        B, C, T = audio.shape
        
        # 짧은 오디오는 청킹하지 않음
        if T <= chunk_samples:
            print(f"   📦 Audio is short ({T/sample_rate:.1f}s), processing without chunking")
            latents, _ = self.dcae_model.encode(audio)
            return self.dcae_model.decode(latents)
        
        print(f"   📦 Processing {T/sample_rate:.1f}s audio with chunking (chunk: {chunk_duration}s, overlap: {overlap}s)")
        
        reconstructed_chunks = []
        step_size = chunk_samples - overlap_samples
        
        for start in range(0, T - overlap_samples, step_size):
            end = min(start + chunk_samples, T)
            
            # 청크 추출
            chunk = audio[:, :, start:end]
            print(f"   🔄 Processing chunk {start/sample_rate:.1f}s-{end/sample_rate:.1f}s ({chunk.shape[-1]/sample_rate:.1f}s)")
            
            # 청크 처리
            try:
                latents, _ = self.dcae_model.encode(chunk)
                reconstructed_chunk = self.dcae_model.decode(latents, target_length=chunk.shape[-1])
                
                print(f"      Input: {chunk.shape} -> Output: {reconstructed_chunk.shape}")
                
                # 길이 검증
                if reconstructed_chunk.shape[-1] != chunk.shape[-1]:
                    print(f"      ⚠️ Length mismatch: expected {chunk.shape[-1]}, got {reconstructed_chunk.shape[-1]}")
                    # 길이 조정
                    if reconstructed_chunk.shape[-1] > chunk.shape[-1]:
                        reconstructed_chunk = reconstructed_chunk[:, :, :chunk.shape[-1]]
                    else:
                        pad_length = chunk.shape[-1] - reconstructed_chunk.shape[-1]
                        reconstructed_chunk = torch.nn.functional.pad(reconstructed_chunk, (0, pad_length))
                
                reconstructed_chunks.append((start, end, reconstructed_chunk))
                
            except Exception as e:
                print(f"   ⚠️ Chunk processing failed: {e}")
                # 더미 청크로 대체
                dummy_chunk = torch.zeros_like(chunk)
                reconstructed_chunks.append((start, end, dummy_chunk))
        
        # 청크들을 병합 (단순 연결 방식)
        print(f"   🔗 Merging {len(reconstructed_chunks)} chunks")
        
        # 결과 텐서 초기화
        result = torch.zeros_like(audio)
        prev_end = 0
        
        for i, (start, end, chunk) in enumerate(reconstructed_chunks):
            actual_len = min(chunk.shape[-1], end - start, result.shape[-1] - start)
            
            if actual_len > 0:
                # 오버랩 처리
                if i > 0 and start < prev_end:
                    # 오버랩 영역에서 블렌딩
                    overlap_start = start
                    overlap_end = min(prev_end, start + actual_len)
                    overlap_len = overlap_end - overlap_start
                    
                    if overlap_len > 0:
                        # 선형 블렌딩
                        alpha = torch.linspace(1.0, 0.0, overlap_len, device=audio.device)
                        alpha = alpha.view(1, 1, -1)
                        
                        # 기존 값과 새 값을 블렌딩
                        result[:, :, overlap_start:overlap_end] = (
                            alpha * result[:, :, overlap_start:overlap_end] +
                            (1 - alpha) * chunk[:, :, :overlap_len]
                        )
                        
                        # 오버랩 이후 부분 추가
                        non_overlap_start = overlap_end
                        non_overlap_end = start + actual_len
                        if non_overlap_end > non_overlap_start:
                            chunk_start = non_overlap_start - start
                            chunk_len = non_overlap_end - non_overlap_start
                            result[:, :, non_overlap_start:non_overlap_end] = chunk[:, :, chunk_start:chunk_start + chunk_len]
                    else:
                        # 오버랩이 없으면 그냥 추가
                        result[:, :, start:start + actual_len] = chunk[:, :, :actual_len]
                else:
                    # 첫 청크이거나 오버랩이 없으면 그냥 추가
                    result[:, :, start:start + actual_len] = chunk[:, :, :actual_len]
                
                prev_end = start + actual_len
        
        return result
    
    def test_pipeline_basic(self, duration: float = 3.0) -> dict:
        """기본 파이프라인 테스트"""
        print(f"\n🧪 Testing basic pipeline with {duration}s audio...")
        
        results = {}
        sample_rate = 44100
        
        # 1. 테스트 오디오 생성
        test_audio = torch.randn(1, 2, int(sample_rate * duration), device=self.device)
        print(f"   📄 Input audio: {test_audio.shape}")
        
        try:
            # 2. 오디오 -> DCAE latent (인코딩)
            start_time = time.time()
            latents, quant_loss = self.dcae_model.encode(test_audio)
            encode_time = time.time() - start_time
            print(f"   ✅ Audio -> Latent: {test_audio.shape} -> {latents.shape} ({encode_time:.3f}s)")
            results['audio_to_latent'] = {'success': True, 'time': encode_time, 'shapes': (str(test_audio.shape), str(latents.shape))}
            
            # 3. DCAE latent -> 오디오 (디코딩 + Vocoder)
            start_time = time.time()
            reconstructed_audio = self.dcae_model.decode(latents, target_length=test_audio.shape[-1])
            decode_time = time.time() - start_time
            print(f"   ✅ Latent -> Audio: {latents.shape} -> {reconstructed_audio.shape} ({decode_time:.3f}s)")
            results['latent_to_audio'] = {'success': True, 'time': decode_time, 'shapes': (str(latents.shape), str(reconstructed_audio.shape))}
            
            # 전체 파이프라인 시간
            total_time = encode_time + decode_time
            print(f"   🎯 Total pipeline time: {total_time:.3f}s")
            
            # 품질 평가
            if reconstructed_audio.shape == test_audio.shape:
                mse_loss = torch.nn.functional.mse_loss(reconstructed_audio, test_audio)
                print(f"   📊 Reconstruction MSE: {mse_loss.item():.6f}")
                results['reconstruction_mse'] = mse_loss.item()
            
            results['total_time'] = total_time
            results['pipeline_success'] = True
            
        except Exception as e:
            print(f"   ❌ Pipeline failed: {e}")
            results['pipeline_success'] = False
            results['error'] = str(e)
        
        return results
    
    def test_pipeline_with_real_audio(self, audio_path: str, use_chunking: bool = True, chunk_duration: float = 10.0) -> dict:
        """실제 오디오 파일로 파이프라인 테스트"""
        print(f"\n🎵 Testing pipeline with real audio: {audio_path}")
        
        results = {}
        
        try:
            # 오디오 로드
            if not Path(audio_path).exists():
                print(f"   ❌ Audio file not found: {audio_path}")
                return {'success': False, 'error': 'File not found'}
            
            original_audio, sr = self.audio_processor.load_audio(
                audio_path, 
                target_sr=44100, 
                normalize=True
            )
            
            if original_audio.dim() == 2:
                original_audio = original_audio.unsqueeze(0)  # (C, T) -> (1, C, T)
            
            original_audio = original_audio.to(self.device)
            duration = original_audio.shape[-1] / 44100
            print(f"   📄 Loaded audio: {original_audio.shape}, SR: {sr}, Duration: {duration:.1f}s")
            
            # 파이프라인 실행
            start_time = time.time()
            
            if use_chunking and duration > chunk_duration:
                print(f"   🔄 Using chunking for {duration:.1f}s audio")
                reconstructed_audio = self.process_audio_with_chunking(
                    original_audio, 
                    chunk_duration=chunk_duration, 
                    overlap=1.0
                )
            else:
                print(f"   🔄 Processing {duration:.1f}s audio without chunking")
                # 1. 오디오 -> latent (DCAE 인코딩)
                latents, _ = self.dcae_model.encode(original_audio)
                
                # 2. latent -> 오디오 (DCAE 디코딩 + Vocoder)
                reconstructed_audio = self.dcae_model.decode(latents, target_length=original_audio.shape[-1])
            
            total_time = time.time() - start_time
            print(f"   ⏱️ Processing time: {total_time:.3f}s")
            
            # 메트릭 계산
            if reconstructed_audio.shape[-1] != original_audio.shape[-1]:
                # 길이 맞춤
                min_len = min(reconstructed_audio.shape[-1], original_audio.shape[-1])
                reconstructed_audio = reconstructed_audio[..., :min_len]
                original_audio = original_audio[..., :min_len]
                print(f"   ✂️ Trimmed to {min_len/44100:.1f}s for comparison")
            
            # 기본 메트릭
            mse_loss = torch.nn.functional.mse_loss(reconstructed_audio, original_audio)
            l1_loss = torch.nn.functional.l1_loss(reconstructed_audio, original_audio)
            
            print(f"   📊 Reconstruction metrics:")
            print(f"      MSE Loss: {mse_loss.item():.6f}")
            print(f"      L1 Loss: {l1_loss.item():.6f}")
            
            # 고급 메트릭 (가능한 경우)
            try:
                metrics = self.metric_calculator.compute_all_metrics(
                    target_audio=original_audio.squeeze(0),
                    generated_audio=reconstructed_audio.squeeze(0)
                )
                
                print(f"      SNR: {metrics['snr'].value:.2f} dB")
                print(f"      SI-SDR: {metrics['si_sdr'].value:.2f} dB")
                
                results['advanced_metrics'] = {k: v.value for k, v in metrics.items()}
                
            except Exception as e:
                print(f"   ⚠️ Advanced metrics failed: {e}")
            
            # 결과 오디오 저장
            output_path = f"reconstructed_{Path(audio_path).stem}.wav"
            torchaudio.save(output_path, reconstructed_audio.squeeze(0).cpu(), 44100)
            print(f"   💾 Reconstructed audio saved: {output_path}")
            
            results.update({
                'success': True,
                'processing_time': total_time,
                'mse_loss': mse_loss.item(),
                'l1_loss': l1_loss.item(),
                'original_shape': str(original_audio.shape),
                'reconstructed_shape': str(reconstructed_audio.shape),
                'duration': duration,
                'used_chunking': use_chunking and duration > chunk_duration,
                'output_file': output_path
            })
            
        except Exception as e:
            print(f"   ❌ Real audio test failed: {e}")
            import traceback
            print(f"   🔍 Traceback:\n{traceback.format_exc()}")
            results = {'success': False, 'error': str(e)}
        
        return results
    
    def test_compression_efficiency(self) -> dict:
        """압축 효율성 테스트"""
        print(f"\n📦 Testing compression efficiency...")
        
        results = {}
        durations = [1.0, 3.0, 5.0, 10.0]  # 다양한 길이
        
        for duration in durations:
            print(f"   Testing {duration}s audio...")
            
            # 테스트 오디오 생성
            sample_rate = 44100
            test_audio = torch.randn(1, 2, int(sample_rate * duration), device=self.device)
            
            try:
                # 압축 정보 계산
                compression_info = self.dcae_model.get_compression_info(test_audio)
                
                print(f"      Original size: {compression_info['original_size']:,} elements")
                print(f"      Compressed size: {compression_info['compressed_size']:,} elements")
                print(f"      Compression ratio: {compression_info['compression_ratio']:.1f}:1")
                
                # 성공 여부 확인
                if compression_info['compression_ratio'] > 0:
                    compression_info['success'] = True
                else:
                    compression_info['success'] = False
                    
                results[f'{duration}s'] = compression_info
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                results[f'{duration}s'] = {'success': False, 'error': str(e)}
        
        # 전체 성공 여부 결정
        all_success = all(result.get('success', False) for result in results.values())
        results['success'] = all_success
        
        return results
    
    def test_generator_integration(self) -> dict:
        """Generator와의 통합 테스트"""
        print(f"\n🔗 Testing Generator integration...")
        
        results = {}
        
        try:
            # Generator 생성
            generator_config = GeneratorConfig(
                latent_channels=self.dcae_model.latent_channels,
                latent_time_steps=256,  # DCAE latent의 실제 시간 차원에 맞춤
                d_model=512,  # 작은 크기로 테스트
                n_layers=4,
                n_heads=8,  # d_model이 512이므로 8 헤드가 적절 (512 / 8 = 64)
                d_ff=2048   # 작은 크기로 테스트
            )
            
            generator = create_lyro_generator(generator_config).to(self.device).eval()
            print(f"   ✅ Generator created: {generator.count_parameters():,} parameters")
            
            # 테스트 조건 준비
            batch_size = 1
            device = self.device
            
            # 더미 조건들 (배치 크기 일치)
            lyrics = torch.randint(0, 1000, (batch_size, 50), device=device)
            lyrics_mask = torch.ones_like(lyrics, dtype=torch.bool)
            captions = ["This is a test music piece"]
            
            # 타겟 latent 형태 생성
            test_audio = torch.randn(batch_size, 2, 44100 * 3, device=device)
            
            # DCAE로 실제 latent 얻기
            target_latents, _ = self.dcae_model.encode(test_audio)
            
            print(f"   📄 Target latents shape: {target_latents.shape}")
            
            # Generator는 3D 형태 (B, C, T)를 기대하므로 필요시 변환
            if target_latents.dim() == 4:
                # (B, C, H, W) -> (B, C, H*W) 변환
                B, C, H, W = target_latents.shape
                target_latents_3d = target_latents.view(B, C, H * W)
            else:
                target_latents_3d = target_latents
            
            # Generator의 기대 시간 차원에 맞춤
            expected_time_steps = generator_config.latent_time_steps
            current_time_steps = target_latents_3d.shape[2]
            
            if current_time_steps != expected_time_steps:
                if current_time_steps < expected_time_steps:
                    # 패딩
                    pad_size = expected_time_steps - current_time_steps
                    target_latents_3d = F.pad(target_latents_3d, (0, pad_size))
                else:
                    # 잘라내기
                    target_latents_3d = target_latents_3d[:, :, :expected_time_steps]
            
            print(f"   📄 Target latents shape for generator: {target_latents_3d.shape}")
            
            # Generator 훈련 손실 테스트
            try:
                print(f"   🔍 Debug: target_latents shape: {target_latents_3d.shape}")
                print(f"   🔍 Debug: lyrics shape: {lyrics.shape}")
                print(f"   🔍 Debug: lyrics_mask shape: {lyrics_mask.shape}")
                
                # 배치 크기 확인 및 조정
                latent_batch_size = target_latents_3d.shape[0]
                condition_batch_size = lyrics.shape[0]
                
                if latent_batch_size != condition_batch_size:
                    print(f"   🔧 Adjusting batch sizes: latent={latent_batch_size}, condition={condition_batch_size}")
                    # latent 배치 크기에 맞춰 조건 조정
                    if latent_batch_size > condition_batch_size:
                        repeat_factor = latent_batch_size // condition_batch_size
                        lyrics = lyrics.repeat(repeat_factor, 1)
                        lyrics_mask = lyrics_mask.repeat(repeat_factor, 1)
                        captions = captions * repeat_factor
                    else:
                        # latent 배치 크기에 맞춰 잘라내기
                        lyrics = lyrics[:latent_batch_size]
                        lyrics_mask = lyrics_mask[:latent_batch_size]
                        captions = captions[:latent_batch_size]
                
                with torch.no_grad():
                    loss_dict = generator.training_loss(
                        latents=target_latents_3d.float(),
                        lyrics=lyrics,
                        lyrics_mask=lyrics_mask,
                        captions=captions,
                        task_type='SONG'
                    )
                
                print(f"   ✅ Training loss: {loss_dict['flow_loss'].item():.4f}")
                training_success = True
            except Exception as e:
                print(f"   ⚠️ Training loss failed: {e}")
                import traceback
                print(f"   🔍 Traceback:\n{traceback.format_exc()}")
                training_success = False
            
            # Generator로 생성 테스트
            try:
                # 배치 크기 조정된 조건 사용
                adjusted_batch_size = target_latents_3d.shape[0]
                
                with torch.no_grad():
                    generated_latents = generator.generate_fast(
                        shape=target_latents_3d.shape,
                        lyrics=lyrics[:adjusted_batch_size] if lyrics.shape[0] > adjusted_batch_size else lyrics,
                        lyrics_mask=lyrics_mask[:adjusted_batch_size] if lyrics_mask.shape[0] > adjusted_batch_size else lyrics_mask,
                        captions=captions[:adjusted_batch_size] if len(captions) > adjusted_batch_size else captions,
                        task_type='SONG',
                        num_steps=10,
                        device=device
                    )
                
                print(f"   ✅ Generated latents: {generated_latents.shape}")
                
                # 생성된 latent를 원본 형태로 변환하여 오디오 복원
                if generated_latents.shape[2] > current_time_steps:
                    generated_latents = generated_latents[:, :, :current_time_steps]
                
                # 4D 형태로 변환 (필요한 경우)
                if target_latents.dim() == 4:
                    B, C, HW = generated_latents.shape
                    H, W = target_latents.shape[2], target_latents.shape[3]
                    generated_latents_4d = generated_latents.view(B, C, H, W)
                else:
                    generated_latents_4d = generated_latents
                
                # 오디오로 디코딩
                generated_audio = self.dcae_model.decode(generated_latents_4d)
                
                generation_success = True
                
            except Exception as e:
                print(f"   ⚠️ Generation failed: {e}")
                generation_success = False
                generated_audio = torch.zeros_like(test_audio)  # 더미 출력
            
            print(f"   ✅ Final audio: {generated_audio.shape}")
            
            results = {
                'success': training_success and generation_success,
                'generator_params': generator.count_parameters(),
                'training_success': training_success,
                'generation_success': generation_success,
                'latent_shape_original': str(target_latents.shape),
                'latent_shape_3d': str(target_latents_3d.shape),
                'generated_shape': str(generated_audio.shape)
            }
            
            if training_success:
                results['flow_loss'] = loss_dict['flow_loss'].item()
            
        except Exception as e:
            print(f"   ❌ Generator integration failed: {e}")
            results = {'success': False, 'error': str(e)}
        
        return results
    
    def benchmark_performance(self, num_runs: int = 5) -> dict:
        """성능 벤치마크"""
        print(f"\n⚡ Benchmarking performance ({num_runs} runs)...")
        
        times = {
            'encode': [],
            'decode': [],
            'total': []
        }
        
        # 테스트 오디오 (5초)
        test_audio = torch.randn(1, 2, 44100 * 5, device=self.device)
        
        for run in range(num_runs):
            print(f"   Run {run + 1}/{num_runs}...")
            
            total_start = time.time()
            
            # 워밍업
            if run == 0:
                with torch.no_grad():
                    _ = self.dcae_model.encode(test_audio)
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
            
            with torch.no_grad():
                # Audio -> Latent (인코딩)
                start = time.time()
                latents, _ = self.dcae_model.encode(test_audio)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                times['encode'].append(time.time() - start)
                
                # Latent -> Audio (디코딩 + Vocoder)
                start = time.time()
                recon_audio = self.dcae_model.decode(latents)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                times['decode'].append(time.time() - start)
            
            times['total'].append(time.time() - total_start)
        
        # 통계 계산
        results = {}
        for stage, stage_times in times.items():
            mean_time = np.mean(stage_times)
            std_time = np.std(stage_times)
            min_time = np.min(stage_times)
            max_time = np.max(stage_times)
            
            print(f"   {stage}: {mean_time:.3f}±{std_time:.3f}s (min: {min_time:.3f}, max: {max_time:.3f})")
            
            results[stage] = {
                'mean': mean_time,
                'std': std_time,
                'min': min_time,
                'max': max_time
            }
        
        # 실시간 팩터 계산
        audio_duration = 5.0  # 5초 오디오
        rt_factor = audio_duration / results['total']['mean']
        print(f"   🚀 Real-time factor: {rt_factor:.2f}x")
        
        results['real_time_factor'] = rt_factor
        
        return results
    
    def save_test_results(self, results: dict, output_path: str):
        """테스트 결과 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Test results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='LYRO DCAE + Vocoder Pipeline Test')
    
    parser.add_argument('--dcae_model', type=str, default='ACE-Step/ACE-Step-v1-3.5B', help='DCAE model name')
    parser.add_argument('--cache_dir', type=str, default='checkpoints', help='Model cache directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--audio_file', type=str, default=None, help='Test with real audio file')
    parser.add_argument('--output', type=str, default='dcae_test_results.json', help='Output results file')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    print("🎵 LYRO DCAE + Vocoder Pipeline Tester")
    print("=" * 50)
    
    # 테스터 초기화
    tester = DCPipelineTester(
        dcae_model_name=args.dcae_model,
        cache_dir=args.cache_dir,
        device=args.device
    )
    
    all_results = {
        'test_info': {
            'dcae_model': args.dcae_model,
            'device': str(tester.device),
            'timestamp': time.time()
        }
    }
    
    # 기본 파이프라인 테스트
    all_results['basic_pipeline'] = tester.test_pipeline_basic()
    
    # 실제 오디오 테스트 (파일이 제공된 경우)
    if args.audio_file:
        all_results['real_audio'] = tester.test_pipeline_with_real_audio(args.audio_file)
    
    # 압축 효율성 테스트
    all_results['compression'] = tester.test_compression_efficiency()
    
    # Generator 통합 테스트
    all_results['generator_integration'] = tester.test_generator_integration()
    
    # 성능 벤치마크 (요청된 경우)
    if args.benchmark:
        all_results['benchmark'] = tester.benchmark_performance(args.num_runs)
    
    # 결과 저장
    tester.save_test_results(all_results, args.output)
    
    # 요약 출력
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    success_count = 0
    total_tests = 0
    
    for test_name, result in all_results.items():
        if test_name == 'test_info':
            continue
        
        total_tests += 1
        if isinstance(result, dict):
            if result.get('success', result.get('pipeline_success', False)):
                success_count += 1
                print(f"   ✅ {test_name}: PASSED")
            else:
                print(f"   ❌ {test_name}: FAILED")
        
    print(f"   📊 Overall: {success_count}/{total_tests} tests passed")
    
    if all_results.get('benchmark'):
        rt_factor = all_results['benchmark'].get('real_time_factor', 0)
        print(f"   🚀 Performance: {rt_factor:.2f}x real-time")
    
    print(f"   💾 Detailed results: {args.output}")
    
    return 0 if success_count == total_tests else 1


if __name__ == '__main__':
    exit(main())