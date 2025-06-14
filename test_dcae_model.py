#!/usr/bin/env python3
"""
ULTRA-OPTIMIZED CQT-SSM DCAE 모델 테스트 스크립트 (Latent 저장 포함)
학습된 모델을 로드하여 오디오 압축/재구성 성능 평가 및 latent representation 저장
"""

import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import librosa
import warnings
from tqdm import tqdm

# LYRO 모듈 import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcae.model import create_cqt_ssm_dcae, CQTSSMDCAE
from dcae.training_utils import compute_snr, compute_si_sdr

warnings.filterwarnings("ignore")


class DCAEModelTester:
    """DCAE 모델 테스트 클래스 (Latent 저장 포함)"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 sample_rate: int = 44100,
                 device: str = 'auto',
                 save_latents: bool = True):
        
        self.checkpoint_path = Path(checkpoint_path)
        self.sample_rate = sample_rate
        self.save_latents = save_latents
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 DCAE 모델 테스터 초기화 (Latent 저장: {save_latents})")
        print(f"📱 디바이스: {self.device}")
        print(f"🎵 샘플레이트: {sample_rate}Hz")
        
        # 모델 로드
        self.model = self._load_model()
        self.model.eval()
        
        print(f"✅ 모델 로드 완료: {self.checkpoint_path}")
    
    def _load_model(self) -> CQTSSMDCAE:
        """체크포인트에서 모델 로드"""
        
        # 메타데이터 로드
        metadata_path = self.checkpoint_path / 'ultra_optimized_metadata.json'
        if not metadata_path.exists():
            metadata_path = self.checkpoint_path / 'ultra_optimized_metadata_minimal.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"📋 메타데이터 로드: {metadata_path}")
            
            # 설정 추출
            config = metadata.get('config', {})
            model_type = metadata.get('model_type', 'ULTRA-OPTIMIZED-CQT-SSM-DCAE')
            audio_duration = metadata.get('audio_duration', 10.0)
            
            print(f"🎼 모델 타입: {model_type}")
            print(f"🔊 오디오 길이: {audio_duration}s")
            
        else:
            print("⚠️ 메타데이터 없음 - 기본 설정 사용")
            config = {}
        
        # 모델 생성
        model = create_cqt_ssm_dcae(
            model_size=config.get('model_size', 'base'),
            sample_rate=self.sample_rate,
            use_vq=config.get('use_vector_quantization', False),
            latent_channels=config.get('latent_channels', 8),
            n_bins=config.get('n_bins', 84),
        )
        
        # 체크포인트 로드
        try:
            # Accelerate 형식 체크포인트
            pytorch_model_path = self.checkpoint_path / 'pytorch_model.bin'
            if pytorch_model_path.exists():
                checkpoint = torch.load(pytorch_model_path, map_location=self.device)
                model.load_state_dict(checkpoint)
                print(f"✅ Accelerate 체크포인트 로드: {pytorch_model_path}")
            else:
                # 일반 체크포인트
                checkpoint_files = list(self.checkpoint_path.glob('*.pt'))
                if checkpoint_files:
                    checkpoint = torch.load(checkpoint_files[0], map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    print(f"✅ 일반 체크포인트 로드: {checkpoint_files[0]}")
                else:
                    raise FileNotFoundError("체크포인트 파일을 찾을 수 없습니다")
                    
        except Exception as e:
            print(f"❌ 체크포인트 로드 실패: {e}")
            print("🔄 기본 모델로 계속 진행...")
        
        return model.to(self.device)
    
    def load_audio(self, audio_path: str, max_duration: Optional[float] = None) -> torch.Tensor:
        """오디오 파일 로드"""
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"오디오 파일 없음: {audio_path}")
        
        # 오디오 로드
        audio, sr = torchaudio.load(audio_path)
        
        # 리샘플링
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # 스테레오로 변환
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]
        
        # 길이 제한
        if max_duration is not None:
            max_samples = int(max_duration * self.sample_rate)
            if audio.shape[1] > max_samples:
                audio = audio[:, :max_samples]
        
        return audio.unsqueeze(0)  # 배치 차원 추가
    
    def extract_latent_representation(self, audio: torch.Tensor) -> torch.Tensor:
        """오디오에서 latent representation 추출"""
        with torch.no_grad():
            # CQT 변환
            cqt = self.model.cqt_transform(audio)
            
            # Encoder로 latent 추출
            if hasattr(self.model, 'encoder'):
                latent = self.model.encoder(cqt)
            else:
                # 모델 구조에 따라 다른 방법으로 latent 추출
                # forward 과정의 일부만 실행
                latent = self.model._encode(cqt)  # 모델에 _encode 메서드가 있다면
            
            return latent
    
    def test_single_audio(self, audio: torch.Tensor) -> Tuple[Dict, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """단일 오디오 테스트 (latent 포함)"""
        start_time = time.time()
        
        with torch.no_grad():
            audio = audio.to(self.device)
            
            # 모델 추론
            try:
                # 기본 추론
                reconstructed, loss_dict = self.model(audio, return_loss=True)
                
                # Latent 추출
                latent = None
                if self.save_latents:
                    try:
                        latent = self.extract_latent_representation(audio)
                        if latent is not None:
                            latent = latent.cpu()
                    except Exception as e:
                        print(f"⚠️ Latent 추출 실패: {e}")
                        # 대안: forward pass 중간 결과 가로채기
                        try:
                            # Hook을 사용하여 중간 표현 추출
                            latent = self._extract_latent_with_hook(audio)
                        except:
                            print("⚠️ Hook을 통한 latent 추출도 실패")
                            latent = None
                
                inference_time = time.time() - start_time
                
                # CPU로 이동
                audio_cpu = audio.cpu()
                reconstructed_cpu = reconstructed.cpu()
                
                # 품질 측정
                metrics = self._compute_quality_metrics(audio_cpu, reconstructed_cpu)
                
                # Latent 정보 추가
                if latent is not None:
                    metrics.update({
                        'latent_shape': list(latent.shape),
                        'latent_mean': float(latent.mean()),
                        'latent_std': float(latent.std()),
                        'latent_min': float(latent.min()),
                        'latent_max': float(latent.max()),
                        'latent_sparsity': float((latent.abs() < 0.01).float().mean()),  # 희소성 측정
                    })
                
                # 결과 정리
                result = {
                    'inference_time': inference_time,
                    'audio_length': audio.shape[-1] / self.sample_rate,
                    'original_shape': list(audio.shape),
                    'reconstructed_shape': list(reconstructed.shape),
                    'compression_ratio': self.model.get_compression_ratio(),
                    **metrics,
                    **{k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
                }
                
                return result, audio_cpu, reconstructed_cpu, latent
                
            except Exception as e:
                print(f"❌ 추론 실패: {e}")
                return None, None, None, None
    
    def _extract_latent_with_hook(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        """Hook을 사용하여 latent representation 추출"""
        latent_output = {}
        
        def hook_fn(module, input, output):
            latent_output['latent'] = output.detach().cpu()
        
        # 적절한 레이어에 hook 등록 (모델 구조에 따라 조정 필요)
        hook_handles = []
        try:
            # Encoder의 마지막 레이어나 bottleneck에 hook 등록
            if hasattr(self.model, 'encoder'):
                for name, module in self.model.encoder.named_modules():
                    if 'conv' in name.lower() and len(list(module.children())) == 0:  # leaf conv layer
                        handle = module.register_forward_hook(hook_fn)
                        hook_handles.append(handle)
                        break
            
            # Forward pass 실행
            _ = self.model(audio, return_loss=False)
            
            # Hook 제거
            for handle in hook_handles:
                handle.remove()
            
            return latent_output.get('latent')
            
        except Exception as e:
            # Hook 제거
            for handle in hook_handles:
                handle.remove()
            print(f"Hook 기반 latent 추출 실패: {e}")
            return None
    
    def _compute_quality_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict:
        """오디오 품질 측정"""
        metrics = {}
        
        try:
            # 기본 메트릭
            orig_sample = original[0].mean(0)  # 모노로 변환
            recon_sample = reconstructed[0].mean(0)
            
            # SNR
            metrics['snr_db'] = compute_snr(orig_sample, recon_sample).item()
            
            # SI-SDR
            metrics['si_sdr_db'] = compute_si_sdr(orig_sample, recon_sample).item()
            
            # MSE
            metrics['mse'] = F.mse_loss(reconstructed, original).item()
            
            # L1 Loss
            metrics['l1_loss'] = F.l1_loss(reconstructed, original).item()
            
            # 스펙트럼 메트릭
            orig_np = orig_sample.numpy()
            recon_np = recon_sample.numpy()
            
            if len(orig_np) > 1024:  # 충분한 길이가 있을 때만
                # 스펙트럼 centroid
                orig_centroid = librosa.feature.spectral_centroid(y=orig_np, sr=self.sample_rate)[0]
                recon_centroid = librosa.feature.spectral_centroid(y=recon_np, sr=self.sample_rate)[0]
                
                min_frames = min(len(orig_centroid), len(recon_centroid))
                if min_frames > 0:
                    metrics['spectral_centroid_error'] = np.mean(np.abs(
                        orig_centroid[:min_frames] - recon_centroid[:min_frames]
                    ))
                
                # 주파수 응답 비교
                orig_fft = np.fft.rfft(orig_np)
                recon_fft = np.fft.rfft(recon_np)
                
                min_bins = min(len(orig_fft), len(recon_fft))
                if min_bins > 0:
                    freq_corr = np.corrcoef(
                        np.abs(orig_fft[:min_bins]), 
                        np.abs(recon_fft[:min_bins])
                    )[0, 1]
                    metrics['frequency_correlation'] = freq_corr if not np.isnan(freq_corr) else 0.0
            
        except Exception as e:
            print(f"⚠️ 메트릭 계산 실패: {e}")
            metrics.update({
                'snr_db': 0.0,
                'si_sdr_db': 0.0,
                'mse': float('inf'),
                'l1_loss': float('inf')
            })
        
        return metrics
    
    def test_directory(self, audio_dir: str, output_dir: str, max_files: int = 10) -> Dict:
        """디렉토리 내 모든 오디오 파일 테스트"""
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Latent 저장 디렉토리 생성
        if self.save_latents:
            latent_dir = output_dir / 'latents'
            latent_dir.mkdir(exist_ok=True)
        
        # 오디오 파일 찾기
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(audio_dir.glob(f'*{ext}'))
            audio_files.extend(audio_dir.glob(f'**/*{ext}'))
        
        audio_files = sorted(list(set(audio_files)))[:max_files]
        
        if not audio_files:
            print(f"❌ 오디오 파일 없음: {audio_dir}")
            return {}
        
        print(f"🎵 {len(audio_files)}개 파일 테스트 시작...")
        
        all_results = []
        total_time = 0
        latent_stats = []
        
        for i, audio_file in enumerate(tqdm(audio_files, desc="테스트 진행")):
            try:
                # 오디오 로드 (최대 30초)
                audio = self.load_audio(str(audio_file), max_duration=30.0)
                
                # 테스트
                result, original, reconstructed, latent = self.test_single_audio(audio)
                
                if result is not None:
                    result['filename'] = audio_file.name
                    all_results.append(result)
                    total_time += result['inference_time']
                    
                    # 결과 저장
                    if original is not None and reconstructed is not None:
                        self._save_comparison(
                            original, reconstructed, 
                            output_dir / f"{audio_file.stem}_comparison.wav",
                            result
                        )
                    
                    # Latent 저장
                    if latent is not None and self.save_latents:
                        latent_path = latent_dir / f"{audio_file.stem}_latent.pt"
                        torch.save(latent, latent_path)
                        
                        # Latent 시각화도 저장
                        self._save_latent_visualization(latent, latent_dir / f"{audio_file.stem}_latent.png")
                        
                        # Latent 통계 수집
                        latent_stats.append({
                            'filename': audio_file.name,
                            'shape': list(latent.shape),
                            'mean': float(latent.mean()),
                            'std': float(latent.std()),
                            'sparsity': float((latent.abs() < 0.01).float().mean())
                        })
                
            except Exception as e:
                print(f"❌ {audio_file.name} 테스트 실패: {e}")
                continue
        
        # 전체 결과 정리
        if all_results:
            summary = self._summarize_results(all_results, total_time)
            
            # Latent 통계 추가
            if latent_stats:
                summary['latent_statistics'] = self._summarize_latent_stats(latent_stats)
            
            # 결과 저장
            with open(output_dir / 'test_results.json', 'w') as f:
                json.dump({
                    'summary': summary,
                    'individual_results': all_results,
                    'latent_statistics': latent_stats if latent_stats else []
                }, f, indent=2)
            
            # 시각화
            self._create_visualizations(all_results, output_dir, latent_stats)
            
            return summary
        else:
            return {}
    
    def _save_comparison(self, original: torch.Tensor, reconstructed: torch.Tensor, 
                        output_path: Path, metrics: Dict):
        """원본과 재구성 오디오 비교 저장"""
        
        # 스테레오 결합 (원본 왼쪽, 재구성 오른쪽)
        comparison = torch.zeros_like(original)
        comparison[0, 0] = original[0].mean(0)  # 원본 모노 -> 왼쪽
        comparison[0, 1] = reconstructed[0].mean(0)  # 재구성 모노 -> 오른쪽
        
        # 저장
        torchaudio.save(output_path, comparison[0], self.sample_rate)
        
        # 메타데이터 저장
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _save_latent_visualization(self, latent: torch.Tensor, output_path: Path):
        """Latent representation 시각화 저장"""
        try:
            latent_np = latent.squeeze().numpy()
            
            plt.figure(figsize=(12, 8))
            
            if len(latent_np.shape) == 3:  # [C, H, W]
                # 첫 번째 채널만 시각화
                plt.subplot(2, 2, 1)
                plt.imshow(latent_np[0], aspect='auto', cmap='viridis')
                plt.title('Latent Representation (Channel 0)')
                plt.colorbar()
                
                # 여러 채널의 평균
                plt.subplot(2, 2, 2)
                plt.imshow(latent_np.mean(0), aspect='auto', cmap='viridis')
                plt.title('Latent Representation (Mean across channels)')
                plt.colorbar()
                
                # 채널별 통계
                plt.subplot(2, 2, 3)
                channel_means = [latent_np[i].mean() for i in range(min(8, latent_np.shape[0]))]
                plt.bar(range(len(channel_means)), channel_means)
                plt.title('Channel-wise Mean Values')
                plt.xlabel('Channel')
                plt.ylabel('Mean Value')
                
                # 분포 히스토그램
                plt.subplot(2, 2, 4)
                plt.hist(latent_np.flatten(), bins=50, alpha=0.7)
                plt.title('Latent Values Distribution')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                
            elif len(latent_np.shape) == 2:  # [H, W]
                plt.subplot(1, 2, 1)
                plt.imshow(latent_np, aspect='auto', cmap='viridis')
                plt.title('Latent Representation')
                plt.colorbar()
                
                plt.subplot(1, 2, 2)
                plt.hist(latent_np.flatten(), bins=50, alpha=0.7)
                plt.title('Latent Values Distribution')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Latent 시각화 저장 실패: {e}")
    
    def _summarize_latent_stats(self, latent_stats: List[Dict]) -> Dict:
        """Latent 통계 요약"""
        if not latent_stats:
            return {}
        
        means = [stat['mean'] for stat in latent_stats]
        stds = [stat['std'] for stat in latent_stats]
        sparsities = [stat['sparsity'] for stat in latent_stats]
        
        return {
            'num_samples': len(latent_stats),
            'mean_statistics': {
                'average': np.mean(means),
                'std': np.std(means),
                'min': np.min(means),
                'max': np.max(means)
            },
            'std_statistics': {
                'average': np.mean(stds),
                'std': np.std(stds),
                'min': np.min(stds),
                'max': np.max(stds)
            },
            'sparsity_statistics': {
                'average': np.mean(sparsities),
                'std': np.std(sparsities),
                'min': np.min(sparsities),
                'max': np.max(sparsities)
            }
        }
    
    def _summarize_results(self, results: List[Dict], total_time: float) -> Dict:
        """결과 요약"""
        
        if not results:
            return {}
        
        # 평균 계산
        avg_snr = np.mean([r.get('snr_db', 0) for r in results])
        avg_si_sdr = np.mean([r.get('si_sdr_db', 0) for r in results])
        avg_mse = np.mean([r.get('mse', float('inf')) for r in results])
        avg_l1 = np.mean([r.get('l1_loss', float('inf')) for r in results])
        avg_inference_time = np.mean([r.get('inference_time', 0) for r in results])
        
        # 압축 비율
        compression_ratio = results[0].get('compression_ratio', 1.0)
        
        summary = {
            'total_files': len(results),
            'total_time': total_time,
            'average_metrics': {
                'snr_db': avg_snr,
                'si_sdr_db': avg_si_sdr,
                'mse': avg_mse,
                'l1_loss': avg_l1,
                'inference_time': avg_inference_time
            },
            'compression_ratio': compression_ratio,
            'model_info': {
                'type': 'ULTRA-OPTIMIZED-CQT-SSM-DCAE',
                'sample_rate': self.sample_rate,
                'device': str(self.device),
                'save_latents': self.save_latents
            }
        }
        
        return summary
    
    def _create_visualizations(self, results: List[Dict], output_dir: Path, latent_stats: List[Dict] = None):
        """결과 시각화 (Latent 포함)"""
        
        if latent_stats:
            # Latent 정보 포함 시각화
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            
            # SNR 분포
            snr_values = [r.get('snr_db', 0) for r in results]
            axes[0, 0].hist(snr_values, bins=20, alpha=0.7, color='blue')
            axes[0, 0].set_xlabel('SNR (dB)')
            axes[0, 0].set_ylabel('빈도')
            axes[0, 0].set_title('SNR 분포')
            axes[0, 0].grid(True, alpha=0.3)
            
            # SI-SDR 분포
            si_sdr_values = [r.get('si_sdr_db', 0) for r in results]
            axes[0, 1].hist(si_sdr_values, bins=20, alpha=0.7, color='green')
            axes[0, 1].set_xlabel('SI-SDR (dB)')
            axes[0, 1].set_ylabel('빈도')
            axes[0, 1].set_title('SI-SDR 분포')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 추론 시간 분포
            inference_times = [r.get('inference_time', 0) for r in results]
            axes[1, 0].hist(inference_times, bins=20, alpha=0.7, color='red')
            axes[1, 0].set_xlabel('추론 시간 (초)')
            axes[1, 0].set_ylabel('빈도')
            axes[1, 0].set_title('추론 시간 분포')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 오디오 길이 vs SNR
            audio_lengths = [r.get('audio_length', 0) for r in results]
            axes[1, 1].scatter(audio_lengths, snr_values, alpha=0.6)
            axes[1, 1].set_xlabel('오디오 길이 (초)')
            axes[1, 1].set_ylabel('SNR (dB)')
            axes[1, 1].set_title('오디오 길이 vs SNR')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Latent 희소성 분포
            sparsities = [stat['sparsity'] for stat in latent_stats]
            axes[2, 0].hist(sparsities, bins=20, alpha=0.7, color='purple')
            axes[2, 0].set_xlabel('Latent 희소성')
            axes[2, 0].set_ylabel('빈도')
            axes[2, 0].set_title('Latent 희소성 분포')
            axes[2, 0].grid(True, alpha=0.3)
            
            # Latent 평균값 분포
            latent_means = [stat['mean'] for stat in latent_stats]
            axes[2, 1].hist(latent_means, bins=20, alpha=0.7, color='orange')
            axes[2, 1].set_xlabel('Latent 평균값')
            axes[2, 1].set_ylabel('빈도')
            axes[2, 1].set_title('Latent 평균값 분포')
            axes[2, 1].grid(True, alpha=0.3)
        else:
            # 기본 시각화
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            snr_values = [r.get('snr_db', 0) for r in results]
            plt.hist(snr_values, bins=20, alpha=0.7, color='blue')
            plt.xlabel('SNR (dB)')
            plt.ylabel('빈도')
            plt.title('SNR 분포')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            si_sdr_values = [r.get('si_sdr_db', 0) for r in results]
            plt.hist(si_sdr_values, bins=20, alpha=0.7, color='green')
            plt.xlabel('SI-SDR (dB)')
            plt.ylabel('빈도')
            plt.title('SI-SDR 분포')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 3)
            inference_times = [r.get('inference_time', 0) for r in results]
            plt.hist(inference_times, bins=20, alpha=0.7, color='red')
            plt.xlabel('추론 시간 (초)')
            plt.ylabel('빈도')
            plt.title('추론 시간 분포')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            audio_lengths = [r.get('audio_length', 0) for r in results]
            plt.scatter(audio_lengths, snr_values, alpha=0.6)
            plt.xlabel('오디오 길이 (초)')
            plt.ylabel('SNR (dB)')
            plt.title('오디오 길이 vs SNR')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'test_analysis_with_latents.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 시각화 저장: {output_dir / 'test_analysis_with_latents.png'}")
    
    def benchmark_performance(self, duration_seconds: float = 10.0, num_trials: int = 5) -> Dict:
        """성능 벤치마크"""
        print(f"⚡ 성능 벤치마크: {duration_seconds}초 오디오, {num_trials}회 반복")
        
        # 테스트 오디오 생성
        test_audio = torch.randn(1, 2, int(duration_seconds * self.sample_rate))
        test_audio = test_audio.to(self.device)
        
        times = []
        memory_usage = []
        
        for i in range(num_trials):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                start_memory = torch.cuda.memory_allocated(self.device)
            
            start_time = time.time()
            
            with torch.no_grad():
                _ = self.model(test_audio, return_loss=False)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated(self.device)
                memory_usage.append((end_memory - start_memory) / 1024**2)  # MB
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        benchmark_results = {
            'audio_duration': duration_seconds,
            'num_trials': num_trials,
            'average_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'real_time_factor': np.mean(times) / duration_seconds,
        }
        
        if memory_usage:
            benchmark_results.update({
                'average_memory_mb': np.mean(memory_usage),
                'peak_memory_mb': np.max(memory_usage)
            })
        
        return benchmark_results


def main():
    parser = argparse.ArgumentParser(description='DCAE 모델 테스트 (Latent 저장 포함)')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='체크포인트 디렉토리 경로')
    parser.add_argument('--audio_path', type=str,
                       help='테스트할 오디오 파일 또는 디렉토리')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--sample_rate', type=int, default=44100,
                       help='샘플레이트')
    parser.add_argument('--max_files', type=int, default=10,
                       help='테스트할 최대 파일 수')
    parser.add_argument('--benchmark', action='store_true',
                       help='성능 벤치마크 실행')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='사용할 디바이스')
    parser.add_argument('--no_save_latents', action='store_true',
                       help='Latent representation 저장하지 않음')
    
    args = parser.parse_args()
    
    # 테스터 초기화
    tester = DCAEModelTester(
        checkpoint_path=args.checkpoint,
        sample_rate=args.sample_rate,
        device=args.device,
        save_latents=not args.no_save_latents
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 벤치마크
    if args.benchmark:
        print("\n" + "="*50)
        print("🏃‍♂️ 성능 벤치마크")
        print("="*50)
        
        benchmark_results = tester.benchmark_performance()
        
        print(f"⏱️  평균 추론 시간: {benchmark_results['average_time']:.3f}초")
        print(f"🚀 실시간 팩터: {benchmark_results['real_time_factor']:.2f}x")
        if 'average_memory_mb' in benchmark_results:
            print(f"💾 평균 메모리 사용: {benchmark_results['average_memory_mb']:.1f}MB")
        
        # 벤치마크 결과 저장
        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2)
    
    # 오디오 테스트
    if args.audio_path:
        print("\n" + "="*50)
        print("🎵 오디오 테스트")
        print("="*50)
        
        audio_path = Path(args.audio_path)
        
        if audio_path.is_file():
            # 단일 파일 테스트
            audio = tester.load_audio(str(audio_path))
            result, original, reconstructed, latent = tester.test_single_audio(audio)
            
            if result:
                print(f"✅ 테스트 완료: {audio_path.name}")
                print(f"🎯 SNR: {result['snr_db']:.2f}dB")
                print(f"🎯 SI-SDR: {result['si_sdr_db']:.2f}dB")
                print(f"⏱️  추론 시간: {result['inference_time']:.3f}초")
                
                if latent is not None:
                    print(f"🧠 Latent 형태: {result['latent_shape']}")
                    print(f"🧠 Latent 희소성: {result['latent_sparsity']:.3f}")
                
                # 결과 저장
                tester._save_comparison(
                    original, reconstructed,
                    output_dir / f"{audio_path.stem}_comparison.wav",
                    result
                )
                
                # Latent 저장
                if latent is not None and not args.no_save_latents:
                    latent_path = output_dir / f"{audio_path.stem}_latent.pt"
                    torch.save(latent, latent_path)
                    print(f"🧠 Latent 저장: {latent_path}")
                    
                    # Latent 시각화
                    tester._save_latent_visualization(
                        latent, 
                        output_dir / f"{audio_path.stem}_latent.png"
                    )
                
                with open(output_dir / 'single_test_result.json', 'w') as f:
                    json.dump(result, f, indent=2)
            
        elif audio_path.is_dir():
            # 디렉토리 테스트
            summary = tester.test_directory(
                str(audio_path), 
                str(output_dir),
                max_files=args.max_files
            )
            
            if summary:
                print(f"✅ 디렉토리 테스트 완료: {summary['total_files']}개 파일")
                print(f"🎯 평균 SNR: {summary['average_metrics']['snr_db']:.2f}dB")
                print(f"🎯 평균 SI-SDR: {summary['average_metrics']['si_sdr_db']:.2f}dB")
                print(f"⏱️  총 시간: {summary['total_time']:.2f}초")
                
                if 'latent_statistics' in summary:
                    print(f"🧠 Latent 평균 희소성: {summary['latent_statistics']['sparsity_statistics']['average']:.3f}")
        else:
            print(f"❌ 경로 없음: {audio_path}")
    
    print(f"\n📁 결과 저장: {output_dir}")
    if not args.no_save_latents:
        print(f"🧠 Latent 저장: {output_dir / 'latents'}")
    print("🎉 테스트 완료!")


if __name__ == '__main__':
    main()