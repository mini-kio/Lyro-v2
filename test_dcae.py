#!/usr/bin/env python3
"""
DCAE 인코딩/디코딩 테스트 스크립트
프리트레인된 DCAE 모델로 1.mp3 파일을 청킹 오버랩하여 테스트
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import time
from pathlib import Path
import logging

# 현재 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.dcae import create_dcae_model
from utils.audio import AudioProcessor, AudioConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DCAEChunkedProcessor:
    """청킹과 오버랩을 사용한 DCAE 처리기"""
    
    def __init__(self, dcae_model, chunk_length=10.0, overlap_ratio=0.25, sample_rate=44100):
        self.dcae_model = dcae_model
        self.chunk_length = chunk_length  # 초
        self.overlap_ratio = overlap_ratio
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_length * sample_rate)
        self.overlap_samples = int(self.chunk_samples * overlap_ratio)
        self.step_samples = self.chunk_samples - self.overlap_samples
        
        logger.info(f"청크 길이: {chunk_length}초 ({self.chunk_samples} 샘플)")
        logger.info(f"오버랩: {overlap_ratio*100}% ({self.overlap_samples} 샘플)")
        logger.info(f"스텝 크기: {self.step_samples} 샘플")
    
    def encode_chunked(self, audio):
        """청킹된 오디오를 인코딩"""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (T,) -> (1, T)
        if audio.dim() == 2 and audio.shape[0] > 2:
            audio = audio[:2]  # 최대 스테레오
        
        total_samples = audio.shape[-1]
        logger.info(f"전체 오디오 길이: {total_samples} 샘플 ({total_samples/self.sample_rate:.2f}초)")
        
        if total_samples <= self.chunk_samples:
            # 단일 청크로 처리
            logger.info("단일 청크로 처리")
            with torch.no_grad():
                audio_batch = audio.unsqueeze(0)  # (C, T) -> (1, C, T)
                latents, _ = self.dcae_model.encode(audio_batch)
                return latents, [(0, total_samples)]
        
        # 멀티 청크 처리
        chunk_positions = []
        latent_chunks = []
        
        start = 0
        chunk_idx = 0
        
        while start < total_samples:
            end = min(start + self.chunk_samples, total_samples)
            
            # 청크 추출
            chunk = audio[..., start:end]
            
            # 마지막 청크가 너무 작으면 패딩
            if chunk.shape[-1] < self.chunk_samples:
                pad_length = self.chunk_samples - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_length))
            
            logger.info(f"청크 {chunk_idx+1}: 샘플 {start}-{end} (길이: {chunk.shape[-1]})")
            
            # 인코딩
            with torch.no_grad():
                chunk_batch = chunk.unsqueeze(0)  # (C, T) -> (1, C, T)
                chunk_latents, _ = self.dcae_model.encode(chunk_batch)
                latent_chunks.append(chunk_latents)
                chunk_positions.append((start, end))
            
            start += self.step_samples
            chunk_idx += 1
            
            if start >= total_samples:
                break
        
        # 잠재 벡터들을 연결
        all_latents = torch.cat(latent_chunks, dim=0)  # (num_chunks, C, T)
        
        logger.info(f"총 {len(latent_chunks)}개 청크 처리됨")
        logger.info(f"잠재 벡터 shape: {all_latents.shape}")
        
        return all_latents, chunk_positions
    
    def decode_chunked(self, latents, chunk_positions, target_length):
        """청킹된 잠재 벡터를 디코딩하고 오버랩 영역을 병합"""
        logger.info("청킹된 잠재 벡터 디코딩 시작")
        
        if latents.dim() == 3 and latents.shape[0] == 1:
            # 단일 청크
            with torch.no_grad():
                decoded = self.dcae_model.decode(latents)
                if decoded.dim() == 3:
                    decoded = decoded.squeeze(0)  # (1, C, T) -> (C, T)
                return decoded[..., :target_length]
        
        # 멀티 청크 처리
        decoded_chunks = []
        
        for i, chunk_latents in enumerate(latents):
            logger.info(f"청크 {i+1} 디코딩 중...")
            
            with torch.no_grad():
                chunk_latents_batch = chunk_latents.unsqueeze(0)  # (C, T) -> (1, C, T)
                decoded_chunk = self.dcae_model.decode(chunk_latents_batch)
                
                if decoded_chunk.dim() == 3:
                    decoded_chunk = decoded_chunk.squeeze(0)  # (1, C, T) -> (C, T)
                
                decoded_chunks.append(decoded_chunk)
        
        # 오버랩 영역 병합
        logger.info("오버랩 영역 병합 중...")
        return self._merge_overlapped_chunks(decoded_chunks, chunk_positions, target_length)
    
    def _merge_overlapped_chunks(self, chunks, positions, target_length):
        """오버랩된 청크들을 병합"""
        # 최종 오디오 버퍼 생성
        channels = chunks[0].shape[0]
        final_audio = torch.zeros(channels, target_length, device=chunks[0].device)
        weight_sum = torch.zeros(target_length, device=chunks[0].device)
        
        for i, (chunk, (start_pos, end_pos)) in enumerate(zip(chunks, positions)):
            chunk_length = min(chunk.shape[-1], end_pos - start_pos)
            actual_end = min(start_pos + chunk_length, target_length)
            
            if start_pos >= target_length:
                continue
            
            # 오버랩 가중치 계산 (코사인 윈도우)
            chunk_weights = torch.ones(chunk_length, device=chunk.device)
            
            if i > 0:  # 첫 번째 청크가 아닌 경우 앞쪽 페이드인
                fade_samples = min(self.overlap_samples, chunk_length // 2)
                fade_in = 0.5 * (1 - torch.cos(torch.linspace(0, np.pi, fade_samples, device=chunk.device)))
                chunk_weights[:fade_samples] = fade_in
            
            if i < len(chunks) - 1:  # 마지막 청크가 아닌 경우 뒤쪽 페이드아웃
                fade_samples = min(self.overlap_samples, chunk_length // 2)
                fade_out = 0.5 * (1 + torch.cos(torch.linspace(0, np.pi, fade_samples, device=chunk.device)))
                chunk_weights[-fade_samples:] = fade_out
            
            # 가중 합산
            write_length = min(chunk_length, actual_end - start_pos)
            final_audio[:, start_pos:start_pos + write_length] += chunk[:, :write_length] * chunk_weights[:write_length]
            weight_sum[start_pos:start_pos + write_length] += chunk_weights[:write_length]
        
        # 가중치 정규화
        weight_sum = torch.clamp(weight_sum, min=1e-8)
        final_audio = final_audio / weight_sum.unsqueeze(0)
        
        return final_audio


def load_test_audio(file_path):
    """테스트 오디오 로드"""
    logger.info(f"오디오 로드 중: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        return None, None
    
    try:
        # torchaudio로 로드
        audio, sr = torchaudio.load(file_path)
        logger.info(f"원본 오디오: {audio.shape}, 샘플레이트: {sr}")
        
        # 44.1kHz로 리샘플
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            audio = resampler(audio)
            sr = 44100
            logger.info(f"리샘플링 후: {audio.shape}, 샘플레이트: {sr}")
        
        # 스테레오 변환
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]
        
        logger.info(f"최종 오디오: {audio.shape}")
        return audio, sr
        
    except Exception as e:
        logger.error(f"오디오 로드 실패: {e}")
        return None, None


def save_audio(audio, path, sample_rate=44100):
    """오디오 저장"""
    logger.info(f"오디오 저장 중: {path}")
    
    try:
        # 정규화
        audio = torch.clamp(audio, -1.0, 1.0)
        
        # 저장
        torchaudio.save(path, audio.cpu(), sample_rate)
        logger.info(f"저장 완료: {path}")
        
    except Exception as e:
        logger.error(f"오디오 저장 실패: {e}")


def calculate_metrics(original, reconstructed):
    """재구성 품질 메트릭 계산"""
    # 길이 맞춤
    min_length = min(original.shape[-1], reconstructed.shape[-1])
    orig = original[..., :min_length]
    recon = reconstructed[..., :min_length]
    
    # MSE
    mse = torch.mean((orig - recon) ** 2).item()
    
    # SNR
    signal_power = torch.mean(orig ** 2)
    noise_power = torch.mean((orig - recon) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()
    
    # 상관관계
    orig_flat = orig.flatten()
    recon_flat = recon.flatten()
    correlation = torch.corrcoef(torch.stack([orig_flat, recon_flat]))[0, 1].item()
    
    return {
        'mse': mse,
        'snr_db': snr,
        'correlation': correlation
    }


def main():
    """메인 테스트 함수"""
    logger.info("🎵 DCAE 인코딩/디코딩 테스트 시작")
    
    # 1. 테스트 파일 확인
    test_file = "1.mp3"
    if not os.path.exists(test_file):
        logger.error(f"테스트 파일을 찾을 수 없습니다: {test_file}")
        logger.info("1.mp3 파일을 현재 디렉토리에 넣어주세요.")
        return False
    
    # 2. DCAE 모델 로드
    logger.info("DCAE 모델 로드 중...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"사용 디바이스: {device}")
        
        dcae_model = create_dcae_model(
            model_name="facebook/musicgen-small",  # 작은 모델로 테스트
            sample_rate=44100
        )
        dcae_model.to(device)
        dcae_model.eval()
        
        logger.info("DCAE 모델 로드 완료")
        
    except Exception as e:
        logger.error(f"DCAE 모델 로드 실패: {e}")
        return False
    
    # 3. 오디오 로드
    audio, sr = load_test_audio(test_file)
    if audio is None:
        return False
    
    audio = audio.to(device)
    original_length = audio.shape[-1]
    
    # 4. 청킹 처리기 생성
    processor = DCAEChunkedProcessor(
        dcae_model=dcae_model,
        chunk_length=10.0,  # 10초 청크
        overlap_ratio=0.25,  # 25% 오버랩
        sample_rate=sr
    )
    
    # 5. 인코딩 테스트
    logger.info("\n🔥 인코딩 시작...")
    start_time = time.time()
    
    try:
        latents, chunk_positions = processor.encode_chunked(audio)
        encode_time = time.time() - start_time
        
        logger.info(f"인코딩 완료 (소요시간: {encode_time:.2f}초)")
        logger.info(f"잠재 벡터 shape: {latents.shape}")
        
        # 압축률 계산
        original_size = audio.numel() * 4  # float32
        latent_size = latents.numel() * 4
        compression_ratio = original_size / latent_size
        
        logger.info(f"압축률: {compression_ratio:.2f}x")
        
    except Exception as e:
        logger.error(f"인코딩 실패: {e}")
        return False
    
    # 6. 디코딩 테스트
    logger.info("\n🔥 디코딩 시작...")
    start_time = time.time()
    
    try:
        reconstructed = processor.decode_chunked(latents, chunk_positions, original_length)
        decode_time = time.time() - start_time
        
        logger.info(f"디코딩 완료 (소요시간: {decode_time:.2f}초)")
        logger.info(f"재구성 오디오 shape: {reconstructed.shape}")
        
    except Exception as e:
        logger.error(f"디코딩 실패: {e}")
        return False
    
    # 7. 품질 평가
    logger.info("\n📊 품질 평가...")
    try:
        metrics = calculate_metrics(audio.cpu(), reconstructed.cpu())
        
        logger.info(f"MSE: {metrics['mse']:.6f}")
        logger.info(f"SNR: {metrics['snr_db']:.2f} dB")
        logger.info(f"상관관계: {metrics['correlation']:.4f}")
        
        # 품질 판정
        if metrics['snr_db'] > 20:
            quality = "우수"
        elif metrics['snr_db'] > 15:
            quality = "양호"
        elif metrics['snr_db'] > 10:
            quality = "보통"
        else:
            quality = "불량"
        
        logger.info(f"재구성 품질: {quality}")
        
    except Exception as e:
        logger.error(f"품질 평가 실패: {e}")
    
    # 8. 결과 저장
    logger.info("\n💾 결과 저장...")
    try:
        # 원본 저장 (참조용)
        save_audio(audio.cpu(), "original.wav", sr)
        
        # 재구성 결과 저장
        save_audio(reconstructed.cpu(), "reconstructed.wav", sr)
        
        # 차이 저장 (잔여 신호)
        difference = audio.cpu() - reconstructed.cpu()
        save_audio(difference * 10, "difference_x10.wav", sr)  # 10배 증폭
        
        logger.info("모든 결과 파일 저장 완료")
        logger.info("- original.wav: 원본 오디오")
        logger.info("- reconstructed.wav: 재구성된 오디오")
        logger.info("- difference_x10.wav: 차이 신호 (10배 증폭)")
        
    except Exception as e:
        logger.error(f"결과 저장 실패: {e}")
    
    # 9. 성능 요약
    logger.info("\n📈 성능 요약:")
    logger.info(f"총 처리 시간: {encode_time + decode_time:.2f}초")
    logger.info(f"실시간 배율: {(original_length/sr)/(encode_time + decode_time):.2f}x")
    logger.info(f"청크 수: {len(chunk_positions)}")
    logger.info(f"오버랩 비율: {processor.overlap_ratio*100}%")
    
    logger.info("\n🎉 DCAE 테스트 완료!")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
