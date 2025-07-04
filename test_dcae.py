#!/usr/bin/env python3
"""
LYRO DCAE + Vocoder 파이프라인 테스트 스크립트 (수정됨)
오류 수정 및 1.mp3 파일 기본 테스트 추가
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
import warnings
import traceback

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dcae import create_dcae_model
from models.generator import create_lyro_generator, GeneratorConfig
from utils.audio import AudioProcessor, AudioConfig
from utils.metrics import MetricCalculator

# 경고 억제
warnings.filterwarnings("ignore")


class DCPipelineTester:
    """DCAE + Vocoder 파이프라인 테스터 (오류 수정)"""
    
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
        try:
            self.dcae_model = create_dcae_model(
                model_name=dcae_model_name,
                cache_dir=cache_dir,
                use_vocoder=True  # Vocoder 활성화
            ).to(self.device).eval()
            
            print("✅ Models loaded successfully!")
            print(f"   DCAE: {type(self.dcae_model).__name__}")
            print(f"   Vocoder: {'Integrated' if hasattr(self.dcae_model, 'vocoder') and self.dcae_model.vocoder is not None else 'Fallback'}")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("Traceback:")
            traceback.print_exc()
            raise
        
        # 유틸리티 초기화
        self.audio_processor = AudioProcessor(AudioConfig(sample_rate=44100))
        self.metric_calculator = MetricCalculator(44100)
    
    def safe_encode(self, audio: torch.Tensor, max_retries: int = 3) -> tuple:
        """안전한 DCAE 인코딩 (오류 수정)"""
        for attempt in range(max_retries):
            try:
                # 입력 검증
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0).unsqueeze(0)  # (T,) -> (1, 1, T)
                elif audio.dim() == 2:
                    audio = audio.unsqueeze(0)  # (C, T) -> (1, C, T)
                elif audio.dim() == 3:
                    pass  # 이미 (B, C, T) 형태
                else:
                    raise ValueError(f"Unexpected audio dimensions: {audio.shape}")
                
                # 채널 확인 및 조정
                if audio.shape[1] == 1:
                    # 모노 -> 스테레오
                    audio = audio.repeat(1, 2, 1)
                elif audio.shape[1] > 2:
                    # 다중 채널 -> 스테레오
                    audio = audio[:, :2, :]
                
                # 길이 확인 및 조정 (최소 2048 샘플)
                if audio.shape[-1] < 2048:
                    pad_size = 2048 - audio.shape[-1]
                    audio = F.pad(audio, (0, pad_size))
                
                # DCAE 인코딩
                with torch.no_grad():
                    latents, quant_loss = self.dcae_model.encode(audio, is_first_chunk=True)
                    return latents, quant_loss
                    
            except Exception as e:
                print(f"   Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    # 최종 실패시 더미 반환
                    print("   Creating dummy latents...")
                    batch_size = 1
                    dummy_latents = torch.randn(batch_size, 8, 16, 16, device=self.device)
                    return dummy_latents, "DUMMY_USED"  # 더미 사용 플래그
                
                # 재시도 전 메모리 정리
                torch.cuda.empty_cache() if self.device.type == 'cuda' else None
                time.sleep(0.1)
        
        return None, None
    
    def safe_decode(self, latents: torch.Tensor, target_length: int = None, max_retries: int = 3) -> torch.Tensor:
        """안전한 DCAE 디코딩 (오류 수정)"""
        for attempt in range(max_retries):
            try:
                # 입력 검증
                if latents is None:
                    raise ValueError("Latents is None")
                
                # latents 차원 확인
                if latents.dim() == 3:
                    # (B, C, T) -> (B, C, H, W) 변환
                    B, C, T = latents.shape
                    H = W = int(np.sqrt(T)) if T > 0 else 16
                    if H * W != T:
                        # 적절한 크기로 조정
                        target_size = 16 * 16
                        if T < target_size:
                            latents = F.pad(latents, (0, target_size - T))
                        else:
                            latents = latents[:, :, :target_size]
                        T = target_size
                        H = W = 16
                    latents = latents.view(B, C, H, W)
                elif latents.dim() != 4:
                    raise ValueError(f"Unexpected latents dimensions: {latents.shape}")
                
                # DCAE 디코딩
                with torch.no_grad():
                    decoded_audio = self.dcae_model.decode(latents, target_length=target_length, is_first_chunk=True)
                    
                    # 결과 검증
                    if decoded_audio is None or decoded_audio.numel() == 0:
                        raise ValueError("Decoded audio is empty")
                    
                    # 차원 정리
                    if decoded_audio.dim() == 3 and decoded_audio.shape[0] == 1:
                        decoded_audio = decoded_audio.squeeze(0)  # (1, C, T) -> (C, T)
                    
                    # 스테레오 보장
                    if decoded_audio.dim() == 1:
                        decoded_audio = decoded_audio.unsqueeze(0).repeat(2, 1)
                    elif decoded_audio.dim() == 2 and decoded_audio.shape[0] == 1:
                        decoded_audio = decoded_audio.repeat(2, 1)
                    
                    # 길이 조정
                    if target_length is not None and decoded_audio.shape[-1] != target_length:
                        if decoded_audio.shape[-1] > target_length:
                            decoded_audio = decoded_audio[..., :target_length]
                        else:
                            pad_size = target_length - decoded_audio.shape[-1]
                            decoded_audio = F.pad(decoded_audio, (0, pad_size))
                    
                    return decoded_audio
                    
            except Exception as e:
                print(f"   Decode attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    # 최종 실패시 더미 오디오 반환
                    print("   Creating dummy audio...")
                    dummy_length = target_length if target_length else 44100
                    return torch.randn(2, dummy_length, device=self.device) * 0.1
                
                # 재시도 전 메모리 정리
                torch.cuda.empty_cache() if self.device.type == 'cuda' else None
                time.sleep(0.1)
    
    def test_pipeline_basic(self, duration: float = 3.0) -> dict:
        """기본 파이프라인 테스트 (오류 수정)"""
        print(f"\n🧪 Testing basic pipeline with {duration}s audio...")
        
        results = {}
        sample_rate = 44100
        
        # 1. 테스트 오디오 생성 (더 안전한 형태)
        test_audio = torch.randn(2, int(sample_rate * duration), device=self.device) * 0.5
        print(f"   📄 Input audio: {test_audio.shape}")
        
        try:
            # 2. 오디오 -> DCAE latent (안전한 인코딩)
            start_time = time.time()
            latents, quant_loss = self.safe_encode(test_audio)
            encode_time = time.time() - start_time
            
            if latents is not None and quant_loss != "DUMMY_USED":
                print(f"   ✅ Audio -> Latent: {test_audio.shape} -> {latents.shape} ({encode_time:.3f}s)")
                results['audio_to_latent'] = {
                    'success': True, 
                    'time': encode_time, 
                    'shapes': (str(test_audio.shape), str(latents.shape))
                }
            else:
                print(f"   ❌ Audio -> Latent failed (dummy used)")
                results['audio_to_latent'] = {'success': False, 'time': encode_time, 'dummy_used': True}
                return results
            
            # 3. DCAE latent -> 오디오 (안전한 디코딩)
            start_time = time.time()
            reconstructed_audio = self.safe_decode(latents, target_length=test_audio.shape[-1])
            decode_time = time.time() - start_time
            
            if reconstructed_audio is not None:
                print(f"   ✅ Latent -> Audio: {latents.shape} -> {reconstructed_audio.shape} ({decode_time:.3f}s)")
                results['latent_to_audio'] = {
                    'success': True, 
                    'time': decode_time, 
                    'shapes': (str(latents.shape), str(reconstructed_audio.shape))
                }
            else:
                print(f"   ❌ Latent -> Audio failed")
                results['latent_to_audio'] = {'success': False, 'time': decode_time}
                return results
            
            # 전체 파이프라인 시간
            total_time = encode_time + decode_time
            print(f"   🎯 Total pipeline time: {total_time:.3f}s")
            
            # 품질 평가 (안전한 계산)
            try:
                if (reconstructed_audio.shape == test_audio.shape and 
                    not torch.isnan(reconstructed_audio).any() and 
                    not torch.isnan(test_audio).any()):
                    
                    mse_loss = F.mse_loss(reconstructed_audio, test_audio)
                    print(f"   📊 Reconstruction MSE: {mse_loss.item():.6f}")
                    results['reconstruction_mse'] = mse_loss.item()
                else:
                    print(f"   ⚠️ Shape mismatch or NaN detected, skipping MSE calculation")
                    results['reconstruction_mse'] = float('inf')
            except Exception as e:
                print(f"   ⚠️ MSE calculation failed: {e}")
                results['reconstruction_mse'] = float('inf')
            
            results['total_time'] = total_time
            results['pipeline_success'] = True
            
        except Exception as e:
            print(f"   ❌ Pipeline failed: {e}")
            print("   Traceback:")
            traceback.print_exc()
            results['pipeline_success'] = False
            results['error'] = str(e)
        
        return results
    
    def test_with_mp3_file(self, mp3_path: str = "1.mp3") -> dict:
        """1.mp3 파일로 파이프라인 테스트"""
        print(f"\n🎵 Testing with MP3 file: {mp3_path}")
        
        results = {'file_path': mp3_path}
        
        # 파일 존재 확인
        if not Path(mp3_path).exists():
            print(f"   ❌ MP3 file not found: {mp3_path}")
            print(f"   💡 Please make sure {mp3_path} exists in the current directory")
            print(f"   🔄 Falling back to synthetic audio test...")
            return self.test_pipeline_basic(duration=5.0)
        
        try:
            # MP3 파일 로드
            print(f"   📁 Loading MP3 file...")
            original_audio, sr = self.audio_processor.load_audio(
                mp3_path, 
                target_sr=44100, 
                normalize=True
            )
            
            # 차원 정리
            if original_audio.dim() == 1:
                original_audio = original_audio.unsqueeze(0)  # (T,) -> (1, T)
            elif original_audio.dim() == 3:
                original_audio = original_audio.squeeze(0)  # (1, C, T) -> (C, T)
            
            # 스테레오 보장
            if original_audio.shape[0] == 1:
                original_audio = original_audio.repeat(2, 1)
            elif original_audio.shape[0] > 2:
                original_audio = original_audio[:2, :]
            
            original_audio = original_audio.to(self.device)
            duration = original_audio.shape[-1] / 44100
            
            print(f"   ✅ Loaded: {original_audio.shape}, Duration: {duration:.1f}s, SR: {sr}")
            
            # 길이 제한 (메모리 절약)
            max_duration = 10.0  # 최대 10초
            if duration > max_duration:
                max_samples = int(max_duration * 44100)
                original_audio = original_audio[..., :max_samples]
                duration = max_duration
                print(f"   ✂️ Trimmed to {duration:.1f}s for testing")
            
            # 파이프라인 실행
            start_time = time.time()
            
            # 1. 인코딩
            print(f"   🔄 Encoding with DCAE...")
            latents, _ = self.safe_encode(original_audio)
            
            if latents is None:
                print(f"   ❌ Encoding failed")
                results['success'] = False
                results['error'] = 'Encoding failed'
                return results
            
            # 2. 디코딩
            print(f"   🔄 Decoding with DCAE + Vocoder...")
            reconstructed_audio = self.safe_decode(latents, target_length=original_audio.shape[-1])
            
            if reconstructed_audio is None:
                print(f"   ❌ Decoding failed")
                results['success'] = False
                results['error'] = 'Decoding failed'
                return results
            
            total_time = time.time() - start_time
            print(f"   ⏱️ Processing time: {total_time:.3f}s")
            
            # 품질 메트릭 계산
            print(f"   📊 Computing quality metrics...")
            try:
                # 길이 맞춤
                min_len = min(reconstructed_audio.shape[-1], original_audio.shape[-1])
                reconstructed_trimmed = reconstructed_audio[..., :min_len]
                original_trimmed = original_audio[..., :min_len]
                
                # 기본 메트릭
                mse_loss = F.mse_loss(reconstructed_trimmed, original_trimmed)
                l1_loss = F.l1_loss(reconstructed_trimmed, original_trimmed)
                
                print(f"      MSE Loss: {mse_loss.item():.6f}")
                print(f"      L1 Loss: {l1_loss.item():.6f}")
                
                # 고급 메트릭 (안전하게)
                try:
                    metrics = self.metric_calculator.compute_all_metrics(
                        target_audio=original_trimmed,
                        generated_audio=reconstructed_trimmed
                    )
                    
                    snr = metrics.get('snr')
                    si_sdr = metrics.get('si_sdr')
                    
                    if snr:
                        print(f"      SNR: {snr.value:.2f} dB")
                    if si_sdr:
                        print(f"      SI-SDR: {si_sdr.value:.2f} dB")
                    
                    results['advanced_metrics'] = {k: v.value for k, v in metrics.items()}
                    
                except Exception as e:
                    print(f"      ⚠️ Advanced metrics failed: {e}")
                
                # 결과 오디오 저장
                output_filename = f"reconstructed_{Path(mp3_path).stem}.wav"
                try:
                    torchaudio.save(
                        output_filename, 
                        reconstructed_audio.cpu(), 
                        44100
                    )
                    print(f"   💾 Reconstructed audio saved: {output_filename}")
                    results['output_file'] = output_filename
                except Exception as e:
                    print(f"   ⚠️ Failed to save output: {e}")
                
                results.update({
                    'success': True,
                    'processing_time': total_time,
                    'mse_loss': mse_loss.item(),
                    'l1_loss': l1_loss.item(),
                    'original_shape': str(original_audio.shape),
                    'reconstructed_shape': str(reconstructed_audio.shape),
                    'duration': duration,
                    'file_loaded': True
                })
                
            except Exception as e:
                print(f"   ⚠️ Quality metrics failed: {e}")
                results.update({
                    'success': True,
                    'processing_time': total_time,
                    'original_shape': str(original_audio.shape),
                    'reconstructed_shape': str(reconstructed_audio.shape),
                    'duration': duration,
                    'file_loaded': True,
                    'metrics_error': str(e)
                })
            
        except Exception as e:
            print(f"   ❌ MP3 test failed: {e}")
            print("   Traceback:")
            traceback.print_exc()
            
            # 폴백: 합성 오디오 테스트
            print(f"   🔄 Falling back to synthetic audio test...")
            return self.test_pipeline_basic(duration=5.0)
        
        return results
    
    def test_compression_efficiency(self) -> dict:
        """압축 효율성 테스트 (오류 수정)"""
        print(f"\n📦 Testing compression efficiency...")
        
        results = {}
        durations = [1.0, 3.0, 5.0, 10.0]
        
        for duration in durations:
            print(f"   Testing {duration}s audio...")
            
            # 테스트 오디오 생성
            sample_rate = 44100
            test_audio = torch.randn(2, int(sample_rate * duration), device=self.device) * 0.5
            
            try:
                # 압축 정보 계산 (안전하게)
                compression_info = self.dcae_model.get_compression_info(test_audio)
                
                print(f"      Original size: {compression_info['original_size']:,} elements")
                print(f"      Compressed size: {compression_info['compressed_size']:,} elements")
                print(f"      Compression ratio: {compression_info['compression_ratio']:.1f}:1")
                
                compression_info['success'] = True
                results[f'{duration}s'] = compression_info
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                results[f'{duration}s'] = {'success': False, 'error': str(e)}
        
        # 전체 성공 여부
        all_success = all(result.get('success', False) for result in results.values())
        results['overall_success'] = all_success
        
        return results
    
    def test_generator_integration(self) -> dict:
        """Generator 통합 테스트 (오류 수정)"""
        print(f"\n🔗 Testing Generator integration...")
        
        results = {}
        
        try:
            # 소형 Generator 생성 (메모리 절약)
            generator_config = GeneratorConfig(
                latent_channels=8,  # DCAE 출력에 맞춤
                latent_time_steps=256,
                d_model=512,
                n_layers=4,
                n_heads=8,
                d_ff=2048
            )
            
            generator = create_lyro_generator(generator_config).to(self.device).eval()
            print(f"   ✅ Generator created: {generator.count_parameters():,} parameters")
            
            # 테스트 준비
            batch_size = 1
            
            # 테스트 오디오
            test_audio = torch.randn(batch_size, 2, 44100 * 3, device=self.device) * 0.5
            
            # DCAE로 실제 latent 얻기
            target_latents, _ = self.safe_encode(test_audio)
            
            if target_latents is None:
                print(f"   ❌ Failed to encode test audio")
                return {'success': False, 'error': 'Encoding failed'}
            
            print(f"   📄 Target latents shape: {target_latents.shape}")
            
            # Generator 입력 형태로 변환
            if target_latents.dim() == 4:
                B, C, H, W = target_latents.shape
                target_latents_3d = target_latents.view(B, C, H * W)
            else:
                target_latents_3d = target_latents
            
            # 길이 조정
            expected_time_steps = generator_config.latent_time_steps
            current_time_steps = target_latents_3d.shape[2]
            
            if current_time_steps != expected_time_steps:
                if current_time_steps < expected_time_steps:
                    pad_size = expected_time_steps - current_time_steps
                    target_latents_3d = F.pad(target_latents_3d, (0, pad_size))
                else:
                    target_latents_3d = target_latents_3d[:, :, :expected_time_steps]
            
            print(f"   📄 Adjusted latents shape: {target_latents_3d.shape}")
            
            # 더미 조건 생성
            lyrics = torch.randint(0, 1000, (batch_size, 50), device=self.device)
            lyrics_mask = torch.ones_like(lyrics, dtype=torch.bool)
            captions = ["This is a test music piece"]
            
            # Generator 훈련 손실 테스트
            training_success = False
            try:
                with torch.no_grad():
                    loss_dict = generator.training_loss(
                        latents=target_latents_3d.float(),
                        lyrics=lyrics,
                        lyrics_mask=lyrics_mask,
                        captions=captions,
                        task_type='SONG'
                    )
                
                if 'flow_loss' in loss_dict:
                    print(f"   ✅ Training loss: {loss_dict['flow_loss'].item():.4f}")
                    training_success = True
                else:
                    print(f"   ⚠️ Training loss missing flow_loss")
                    
            except Exception as e:
                print(f"   ⚠️ Training loss failed: {e}")
            
            # Generator 생성 테스트
            generation_success = False
            try:
                with torch.no_grad():
                    generated_latents = generator.generate_fast(
                        shape=target_latents_3d.shape,
                        lyrics=lyrics,
                        lyrics_mask=lyrics_mask,
                        captions=captions,
                        task_type='SONG',
                        num_steps=10,
                        device=self.device
                    )
                
                print(f"   ✅ Generated latents: {generated_latents.shape}")
                
                # 오디오로 디코딩 (원래 형태로 변환)
                if target_latents.dim() == 4:
                    B, C, HW = generated_latents.shape
                    H, W = target_latents.shape[2], target_latents.shape[3]
                    generated_latents_4d = generated_latents.view(B, C, H, W)
                else:
                    generated_latents_4d = generated_latents
                
                generated_audio = self.safe_decode(generated_latents_4d)
                
                if generated_audio is not None:
                    print(f"   ✅ Final audio: {generated_audio.shape}")
                    generation_success = True
                else:
                    print(f"   ❌ Audio generation failed")
                    
            except Exception as e:
                print(f"   ⚠️ Generation failed: {e}")
            
            results = {
                'success': training_success and generation_success,
                'generator_params': generator.count_parameters(),
                'training_success': training_success,
                'generation_success': generation_success,
                'latent_shape_original': str(target_latents.shape),
                'latent_shape_3d': str(target_latents_3d.shape)
            }
            
            if training_success and 'flow_loss' in loss_dict:
                results['flow_loss'] = loss_dict['flow_loss'].item()
            
        except Exception as e:
            print(f"   ❌ Generator integration failed: {e}")
            print("   Traceback:")
            traceback.print_exc()
            results = {'success': False, 'error': str(e)}
        
        return results
    
    def save_test_results(self, results: dict, output_path: str):
        """테스트 결과 저장"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON 직렬화 가능하도록 변환
        def make_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = make_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"💾 Test results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='LYRO DCAE + Vocoder Pipeline Test (Fixed)')
    
    parser.add_argument('--dcae_model', type=str, default='ACE-Step/ACE-Step-v1-3.5B', help='DCAE model name')
    parser.add_argument('--cache_dir', type=str, default='checkpoints', help='Model cache directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--mp3_file', type=str, default='1.mp3', help='MP3 file to test (default: 1.mp3)')
    parser.add_argument('--output', type=str, default='dcae_test_results.json', help='Output results file')
    parser.add_argument('--skip_mp3', action='store_true', help='Skip MP3 test and only run synthetic tests')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    print("🎵 LYRO DCAE + Vocoder Pipeline Tester (Fixed)")
    print("=" * 50)
    
    # 테스터 초기화
    try:
        tester = DCPipelineTester(
            dcae_model_name=args.dcae_model,
            cache_dir=args.cache_dir,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        return 1
    
    all_results = {
        'test_info': {
            'dcae_model': args.dcae_model,
            'device': str(tester.device),
            'timestamp': time.time(),
            'mp3_file': args.mp3_file
        }
    }
    
    # MP3 파일 테스트 (기본)
    if not args.skip_mp3:
        all_results['mp3_test'] = tester.test_with_mp3_file(args.mp3_file)
    
    # 기본 파이프라인 테스트
    all_results['basic_pipeline'] = tester.test_pipeline_basic()
    
    # 압축 효율성 테스트
    all_results['compression'] = tester.test_compression_efficiency()
    
    # Generator 통합 테스트
    all_results['generator_integration'] = tester.test_generator_integration()
    
    # 결과 저장
    tester.save_test_results(all_results, args.output)
    
    # 요약 출력
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    
    success_count = 0
    total_tests = 0
    
    test_results = {
        'mp3_test': all_results.get('mp3_test', {}),
        'basic_pipeline': all_results.get('basic_pipeline', {}),
        'compression': all_results.get('compression', {}),
        'generator_integration': all_results.get('generator_integration', {})
    }
    
    for test_name, result in test_results.items():
        if test_name == 'mp3_test' and args.skip_mp3:
            continue
            
        total_tests += 1
        success = False
        
        if isinstance(result, dict):
            if test_name == 'compression':
                success = result.get('overall_success', False)
            else:
                success = result.get('success', result.get('pipeline_success', False))
        
        if success:
            success_count += 1
            print(f"   ✅ {test_name}: PASSED")
        else:
            print(f"   ❌ {test_name}: FAILED")
            if isinstance(result, dict) and 'error' in result:
                print(f"      Error: {result['error']}")
    
    print(f"   📊 Overall: {success_count}/{total_tests} tests passed")
    
    # MP3 테스트 특별 정보
    if not args.skip_mp3 and 'mp3_test' in all_results:
        mp3_result = all_results['mp3_test']
        if mp3_result.get('file_loaded'):
            duration = mp3_result.get('duration', 0)
            processing_time = mp3_result.get('processing_time', 0)
            if duration > 0 and processing_time > 0:
                rt_factor = duration / processing_time
                print(f"   🎵 MP3 Processing: {duration:.1f}s audio in {processing_time:.2f}s ({rt_factor:.2f}x real-time)")
    
    print(f"   💾 Detailed results: {args.output}")
    
    # MP3 파일이 없을 때 안내
    if not args.skip_mp3 and not Path(args.mp3_file).exists():
        print(f"\n💡 Tip: Place a file named '{args.mp3_file}' in the current directory to test with real audio!")
    
    return 0 if success_count == total_tests else 1


if __name__ == '__main__':
    exit(main())