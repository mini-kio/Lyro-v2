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

from models.dcae import create_dcae_model, create_vocoder_model
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
        
        # 모델 로드
        print("📥 Loading DCAE model...")
        self.dcae_model = create_dcae_model(
            model_type="pretrained",
            model_name=dcae_model_name,
            cache_dir=cache_dir
        ).to(self.device).eval()
        
        print("📥 Loading Vocoder model...")
        self.vocoder_model = create_vocoder_model(
            model_name=dcae_model_name,
            cache_dir=cache_dir
        ).to(self.device).eval()
        
        # 유틸리티 초기화
        self.audio_processor = AudioProcessor(AudioConfig(sample_rate=44100))
        self.metric_calculator = MetricCalculator(44100)
        
        print("✅ Models loaded successfully!")
        print(f"   DCAE: {type(self.dcae_model).__name__}")
        print(f"   Vocoder: {type(self.vocoder_model).__name__}")
    
    def test_pipeline_basic(self, duration: float = 3.0) -> dict:
        """기본 파이프라인 테스트"""
        print(f"\n🧪 Testing basic pipeline with {duration}s audio...")
        
        results = {}
        sample_rate = 44100
        
        # 1. 테스트 오디오 생성
        test_audio = torch.randn(1, 2, int(sample_rate * duration), device=self.device)
        print(f"   📄 Input audio: {test_audio.shape}")
        
        try:
            # 2. 오디오 -> 멜 스펙트로그램
            start_time = time.time()
            mel = self.dcae_model.audio_to_mel(test_audio)
            mel_time = time.time() - start_time
            print(f"   ✅ Audio -> Mel: {test_audio.shape} -> {mel.shape} ({mel_time:.3f}s)")
            results['audio_to_mel'] = {'success': True, 'time': mel_time, 'shapes': (str(test_audio.shape), str(mel.shape))}
            
            # 3. 멜 -> DCAE latent
            start_time = time.time()
            latents, quant_loss = self.dcae_model.encode_mel(mel)
            encode_time = time.time() - start_time
            print(f"   ✅ Mel -> Latent: {mel.shape} -> {latents.shape} ({encode_time:.3f}s)")
            results['mel_to_latent'] = {'success': True, 'time': encode_time, 'shapes': (str(mel.shape), str(latents.shape))}
            
            # 4. DCAE latent -> 멜 스펙트로그램
            start_time = time.time()
            reconstructed_mel = self.dcae_model.decode_to_mel(latents)
            decode_time = time.time() - start_time
            print(f"   ✅ Latent -> Mel: {latents.shape} -> {reconstructed_mel.shape} ({decode_time:.3f}s)")
            results['latent_to_mel'] = {'success': True, 'time': decode_time, 'shapes': (str(latents.shape), str(reconstructed_mel.shape))}
            
            # 5. 멜 -> 오디오 (Vocoder)
            start_time = time.time()
            reconstructed_audio = self.vocoder_model.mel_to_audio(reconstructed_mel)
            vocoder_time = time.time() - start_time
            print(f"   ✅ Mel -> Audio: {reconstructed_mel.shape} -> {reconstructed_audio.shape} ({vocoder_time:.3f}s)")
            results['mel_to_audio'] = {'success': True, 'time': vocoder_time, 'shapes': (str(reconstructed_mel.shape), str(reconstructed_audio.shape))}
            
            # 전체 파이프라인 시간
            total_time = mel_time + encode_time + decode_time + vocoder_time
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
    
    def test_pipeline_with_real_audio(self, audio_path: str) -> dict:
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
            print(f"   📄 Loaded audio: {original_audio.shape}, SR: {sr}")
            
            # 파이프라인 실행
            start_time = time.time()
            
            # 1. 오디오 -> 멜
            mel = self.dcae_model.audio_to_mel(original_audio)
            
            # 2. 멜 -> latent
            latents, _ = self.dcae_model.encode_mel(mel)
            
            # 3. latent -> 멜
            reconstructed_mel = self.dcae_model.decode_to_mel(latents)
            
            # 4. 멜 -> 오디오
            reconstructed_audio = self.vocoder_model.mel_to_audio(reconstructed_mel)
            
            total_time = time.time() - start_time
            print(f"   ⏱️ Processing time: {total_time:.3f}s")
            
            # 메트릭 계산
            if reconstructed_audio.shape[-1] != original_audio.shape[-1]:
                # 길이 맞춤
                min_len = min(reconstructed_audio.shape[-1], original_audio.shape[-1])
                reconstructed_audio = reconstructed_audio[..., :min_len]
                original_audio = original_audio[..., :min_len]
            
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
            
            results.update({
                'success': True,
                'processing_time': total_time,
                'mse_loss': mse_loss.item(),
                'l1_loss': l1_loss.item(),
                'original_shape': str(original_audio.shape),
                'reconstructed_shape': str(reconstructed_audio.shape),
                'compression_ratio': self.dcae_model.compression_ratio
            })
            
        except Exception as e:
            print(f"   ❌ Real audio test failed: {e}")
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
            
            # 더미 조건들
            lyrics = torch.randint(0, 1000, (batch_size, 50), device=device)
            lyrics_mask = torch.ones_like(lyrics, dtype=torch.bool)
            captions = ["This is a test music piece"]
            
            # 타겟 latent 형태 생성
            test_audio = torch.randn(batch_size, 2, 44100 * 3, device=device)
            
            # DCAE로 실제 latent 얻기
            mel = self.dcae_model.audio_to_mel(test_audio)
            target_latents_4d, _ = self.dcae_model.encode_mel(mel)
            
            # Generator용 3D 형식으로 변환
            target_latents = self.dcae_model.latents_to_generator_format(target_latents_4d)
            
            # Generator의 기대 시간 차원에 맞춤
            expected_time_steps = generator_config.latent_time_steps
            current_time_steps = target_latents.shape[2]
            
            if current_time_steps != expected_time_steps:
                if current_time_steps < expected_time_steps:
                    # 패딩
                    pad_size = expected_time_steps - current_time_steps
                    target_latents = F.pad(target_latents, (0, pad_size))
                else:
                    # 잘라내기
                    target_latents = target_latents[:, :, :expected_time_steps]
            
            print(f"   📄 Target latents shape: {target_latents.shape} (from 4D: {target_latents_4d.shape})")
            
            # Generator 훈련 손실 테스트
            try:
                print(f"   🔍 Debug: target_latents shape: {target_latents.shape}")
                print(f"   🔍 Debug: lyrics shape: {lyrics.shape}")
                print(f"   🔍 Debug: lyrics_mask shape: {lyrics_mask.shape}")
                
                with torch.no_grad():
                    loss_dict = generator.training_loss(
                        latents=target_latents.float(),
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
                with torch.no_grad():
                    generated_latents = generator.generate_fast(
                        shape=target_latents.shape,
                        lyrics=lyrics,
                        lyrics_mask=lyrics_mask,
                        captions=captions,
                        task_type='SONG',
                        num_steps=10,
                        device=device
                    )
                
                print(f"   ✅ Generated latents: {generated_latents.shape}")
                
                # 생성된 latent를 4D로 변환하여 오디오 복원 (원본 크기 정보 필요)
                # 원본 4D 크기로 다시 변환하기 위해 잘라내기
                if generated_latents.shape[2] > current_time_steps:
                    generated_latents = generated_latents[:, :, :current_time_steps]
                
                generated_latents_4d = self.dcae_model.latents_from_generator_format(
                    generated_latents, 
                    original_4d_shape=target_latents_4d.shape
                )
                generated_mel = self.dcae_model.decode_to_mel(generated_latents_4d)
                generated_audio = self.vocoder_model.mel_to_audio(generated_mel)
                
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
                'latent_shape_4d': str(target_latents_4d.shape),
                'latent_shape_3d': str(target_latents.shape),
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
            'audio_to_mel': [],
            'mel_to_latent': [],
            'latent_to_mel': [],
            'mel_to_audio': [],
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
                    _ = self.dcae_model.audio_to_mel(test_audio)
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
            
            with torch.no_grad():
                # Audio -> Mel
                start = time.time()
                mel = self.dcae_model.audio_to_mel(test_audio)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                times['audio_to_mel'].append(time.time() - start)
                
                # Mel -> Latent
                start = time.time()
                latents, _ = self.dcae_model.encode_mel(mel)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                times['mel_to_latent'].append(time.time() - start)
                
                # Latent -> Mel
                start = time.time()
                recon_mel = self.dcae_model.decode_to_mel(latents)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                times['latent_to_mel'].append(time.time() - start)
                
                # Mel -> Audio
                start = time.time()
                recon_audio = self.vocoder_model.mel_to_audio(recon_mel)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None
                times['mel_to_audio'].append(time.time() - start)
            
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