# lyro/dataset/preprocess.py
"""
LYRO Data Preprocessing Script
오디오 분리, 메타데이터 생성, 품질 검증
"""

import os
import sys
import json
import argparse
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from functools import partial
import logging
import hashlib
import subprocess

# 오디오 분리를 위한 Demucs 임포트 (선택적)
try:
    import demucs.separate
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    print("Warning: Demucs not available. Audio separation will be skipped.")

# 가사 정렬을 위한 MFA (선택적)
try:
    from montreal_forced_aligner import align
    MFA_AVAILABLE = True
except ImportError:
    MFA_AVAILABLE = False


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """오디오 처리 클래스"""
    
    def __init__(
        self,
        target_sr: int = 44100,
        min_duration: float = 10.0,
        max_duration: float = 600.0,
        loudness_target: float = -14.0  # LUFS
    ):
        self.target_sr = target_sr
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.loudness_target = loudness_target
        
    def process_audio_file(self, audio_path: Path) -> Dict:
        """단일 오디오 파일 처리"""
        try:
            # 오디오 로드
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
            
            # 모노를 스테레오로 변환
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            elif audio.shape[0] > 2:
                audio = audio[:2]  # 처음 2채널만 사용
                
            # 리샘플링
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                
            # 길이 확인
            duration = audio.shape[1] / self.target_sr
            if duration < self.min_duration:
                return {'status': 'too_short', 'duration': duration}
            if duration > self.max_duration:
                return {'status': 'too_long', 'duration': duration}
                
            # 라우드니스 정규화
            audio = self.normalize_loudness(audio)
            
            # 무음 제거
            audio = self.trim_silence(audio)
            
            # 체크섬 계산
            checksum = self.calculate_checksum(audio)
            
            return {
                'status': 'success',
                'duration': audio.shape[1] / self.target_sr,
                'channels': audio.shape[0],
                'sample_rate': self.target_sr,
                'checksum': checksum,
                'audio': audio
            }
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """라우드니스 정규화 (LUFS)"""
        # 간단한 RMS 기반 정규화 (실제로는 LUFS 측정 필요)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 10 ** (self.loudness_target / 20)
            audio = audio * (target_rms / rms)
        return audio
    
    def trim_silence(self, audio: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
        """무음 구간 제거"""
        # 스테레오를 모노로 변환하여 무음 감지
        mono = audio.mean(axis=0)
        
        # 무음이 아닌 구간 찾기
        threshold = 10 ** (threshold_db / 20)
        non_silent = np.abs(mono) > threshold
        
        # 시작과 끝 찾기
        indices = np.where(non_silent)[0]
        if len(indices) > 0:
            start = indices[0]
            end = indices[-1] + 1
            return audio[:, start:end]
        
        return audio
    
    def calculate_checksum(self, audio: np.ndarray) -> str:
        """오디오 체크섬 계산"""
        return hashlib.md5(audio.tobytes()).hexdigest()
    
    def separate_stems(self, audio_path: Path, output_dir: Path) -> Dict:
        """Demucs를 사용한 음원 분리"""
        if not DEMUCS_AVAILABLE:
            return {'status': 'demucs_not_available'}
            
        try:
            # Demucs 실행
            cmd = [
                'python', '-m', 'demucs.separate',
                '-n', 'htdemucs',  # 모델 선택
                '-d', 'cuda' if torch.cuda.is_available() else 'cpu',
                '--two-stems', 'vocals',  # 보컬/반주 분리
                '-o', str(output_dir),
                str(audio_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # 결과 파일 찾기
            stem_dir = output_dir / 'htdemucs' / audio_path.stem
            vocal_path = stem_dir / 'vocals.wav'
            inst_path = stem_dir / 'no_vocals.wav'
            
            if vocal_path.exists() and inst_path.exists():
                return {
                    'status': 'success',
                    'vocal_path': vocal_path,
                    'inst_path': inst_path
                }
            else:
                return {'status': 'output_not_found'}
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Demucs error: {e.stderr}")
            return {'status': 'error', 'error': str(e)}
        except Exception as e:
            logger.error(f"Separation error: {e}")
            return {'status': 'error', 'error': str(e)}


class MetadataGenerator:
    """메타데이터 생성 및 관리"""
    
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.metadata = []
        
    def scan_dataset(self) -> List[Dict]:
        """데이터셋 스캔 및 메타데이터 생성"""
        # 오디오 파일 찾기
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        audio_files = []
        
        audio_dir = self.dataset_root / 'audio' / 'full'
        if audio_dir.exists():
            for ext in audio_extensions:
                audio_files.extend(audio_dir.glob(f'*{ext}'))
                
        logger.info(f"Found {len(audio_files)} audio files")
        
        # 각 오디오 파일에 대한 메타데이터 생성
        for audio_path in tqdm(audio_files, desc="Scanning audio files"):
            metadata_item = self.create_metadata_item(audio_path)
            if metadata_item:
                self.metadata.append(metadata_item)
                
        return self.metadata
    
    def create_metadata_item(self, audio_path: Path) -> Optional[Dict]:
        """단일 오디오 파일의 메타데이터 생성"""
        try:
            # 기본 정보
            item = {
                'id': audio_path.stem,
                'audio_path': str(audio_path.relative_to(self.dataset_root))
            }
            
            # 가사 파일 찾기
            lyric_path = self.dataset_root / 'lyrics' / f'{audio_path.stem}.txt'
            if lyric_path.exists():
                item['lyric_path'] = str(lyric_path.relative_to(self.dataset_root))
            else:
                item['lyric_path'] = None
                
            # 분리된 트랙 찾기
            inst_path = self.dataset_root / 'audio' / 'instrumental' / f'{audio_path.stem}_inst.wav'
            vocal_path = self.dataset_root / 'audio' / 'vocal' / f'{audio_path.stem}_vocal.wav'
            
            if inst_path.exists():
                item['inst_path'] = str(inst_path.relative_to(self.dataset_root))
            else:
                item['inst_path'] = None
                
            if vocal_path.exists():
                item['vocal_path'] = str(vocal_path.relative_to(self.dataset_root))
            else:
                item['vocal_path'] = None
                
            # 장르 추론 (파일명 또는 디렉토리 구조에서)
            item['genre'] = self.infer_genre(audio_path)
            
            # 제목 추론
            item['title'] = self.clean_title(audio_path.stem)
            
            return item
            
        except Exception as e:
            logger.error(f"Error creating metadata for {audio_path}: {e}")
            return None
            
    def infer_genre(self, audio_path: Path) -> List[str]:
        """장르 추론"""
        # 간단한 규칙 기반 추론
        filename_lower = audio_path.stem.lower()
        
        genre_keywords = {
            'pop': ['pop'],
            'rock': ['rock'],
            'jazz': ['jazz'],
            'classical': ['classical', 'symphony', 'sonata'],
            'electronic': ['electronic', 'edm', 'techno'],
            'hip-hop': ['hiphop', 'hip-hop', 'rap'],
            'folk': ['folk'],
            'blues': ['blues'],
            'metal': ['metal'],
            'indie': ['indie']
        }
        
        genres = []
        for genre, keywords in genre_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                genres.append(genre.capitalize())
                
        return genres if genres else ['Unknown']
    
    def clean_title(self, filename: str) -> str:
        """파일명에서 제목 추출"""
        # 숫자, 특수문자 제거
        title = filename
        
        # 일반적인 패턴 제거
        patterns = [
            r'^\d+[\s\-_\.]*',  # 앞의 숫자
            r'[\(\[].*?[\)\]]',  # 괄호 내용
            r'_inst$', r'_vocal$',  # 접미사
            r'\.mp3$', r'\.wav$'  # 확장자
        ]
        
        import re
        for pattern in patterns:
            title = re.sub(pattern, '', title)
            
        # 언더스코어를 공백으로
        title = title.replace('_', ' ').replace('-', ' ')
        
        # 연속 공백 제거
        title = ' '.join(title.split())
        
        return title.strip()
    
    def save_metadata(self, output_path: Path):
        """메타데이터 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.metadata:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        logger.info(f"Saved {len(self.metadata)} metadata items to {output_path}")
        
    def split_metadata(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Train/Val/Test 분할"""
        np.random.shuffle(self.metadata)
        
        total = len(self.metadata)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_data = self.metadata[:train_size]
        val_data = self.metadata[train_size:train_size + val_size]
        test_data = self.metadata[train_size + val_size:]
        
        return train_data, val_data, test_data


class QualityFilter:
    """데이터 품질 필터링"""
    
    def __init__(
        self,
        min_snr: float = 20.0,  # dB
        min_complexity: float = 0.3,
        max_clipping_ratio: float = 0.01
    ):
        self.min_snr = min_snr
        self.min_complexity = min_complexity
        self.max_clipping_ratio = max_clipping_ratio
        
    def check_quality(self, audio: np.ndarray, sr: int) -> Dict:
        """오디오 품질 검사"""
        quality_metrics = {}
        
        # SNR 추정
        quality_metrics['snr'] = self.estimate_snr(audio)
        
        # 스펙트럴 복잡도
        quality_metrics['complexity'] = self.calculate_spectral_complexity(audio, sr)
        
        # 클리핑 비율
        quality_metrics['clipping_ratio'] = self.calculate_clipping_ratio(audio)
        
        # 전체 품질 판정
        quality_metrics['pass'] = (
            quality_metrics['snr'] >= self.min_snr and
            quality_metrics['complexity'] >= self.min_complexity and
            quality_metrics['clipping_ratio'] <= self.max_clipping_ratio
        )
        
        return quality_metrics
    
    def estimate_snr(self, audio: np.ndarray) -> float:
        """SNR 추정 (간단한 방법)"""
        # 신호 파워
        signal_power = np.mean(audio ** 2)
        
        # 노이즈 추정 (고주파 필터링 후)
        from scipy import signal
        sos = signal.butter(10, 0.9, 'high', output='sos')
        noise = signal.sosfilt(sos, audio.mean(axis=0))
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 100.0  # 매우 높은 SNR
            
        return snr
    
    def calculate_spectral_complexity(self, audio: np.ndarray, sr: int) -> float:
        """스펙트럴 복잡도 계산"""
        # STFT
        D = librosa.stft(audio.mean(axis=0))
        S = np.abs(D)
        
        # 스펙트럴 엔트로피
        S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-10)
        entropy = -np.sum(S_norm * np.log(S_norm + 1e-10), axis=0)
        
        # 정규화된 평균 엔트로피
        max_entropy = np.log(S.shape[0])
        normalized_entropy = np.mean(entropy) / max_entropy
        
        return float(normalized_entropy)
    
    def calculate_clipping_ratio(self, audio: np.ndarray) -> float:
        """클리핑 비율 계산"""
        threshold = 0.99
        clipped_samples = np.sum(np.abs(audio) > threshold)
        total_samples = audio.size
        
        return clipped_samples / total_samples


def process_single_file(args: Tuple[Path, Path, AudioProcessor]) -> Dict:
    """단일 파일 처리 (멀티프로세싱용)"""
    audio_path, output_root, processor = args
    
    result = {
        'audio_path': audio_path,
        'status': 'pending'
    }
    
    try:
        # 오디오 처리
        audio_result = processor.process_audio_file(audio_path)
        
        if audio_result['status'] != 'success':
            result['status'] = audio_result['status']
            return result
            
        # 처리된 오디오 저장
        processed_dir = output_root / 'audio' / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = processed_dir / f'{audio_path.stem}_processed.wav'
        sf.write(output_path, audio_result['audio'].T, processor.target_sr)
        
        result.update({
            'status': 'success',
            'processed_path': output_path,
            'duration': audio_result['duration'],
            'checksum': audio_result['checksum']
        })
        
        # 음원 분리 (옵션)
        if DEMUCS_AVAILABLE:
            sep_result = processor.separate_stems(audio_path, output_root / 'audio')
            if sep_result['status'] == 'success':
                result['vocal_path'] = sep_result['vocal_path']
                result['inst_path'] = sep_result['inst_path']
                
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        result['status'] = 'error'
        result['error'] = str(e)
        
    return result


def main():
    parser = argparse.ArgumentParser(description='LYRO Dataset Preprocessing')
    
    # 경로 설정
    parser.add_argument('--dataset_root', type=str, default='dataset/',
                        help='Dataset root directory')
    parser.add_argument('--output_root', type=str, default=None,
                        help='Output root directory (default: same as dataset_root)')
    
    # 처리 옵션
    parser.add_argument('--separate_stems', action='store_true',
                        help='Separate vocals and instrumentals using Demucs')
    parser.add_argument('--quality_filter', action='store_true',
                        help='Apply quality filtering')
    parser.add_argument('--generate_metadata', action='store_true',
                        help='Generate metadata files')
    
    # 오디오 설정
    parser.add_argument('--target_sr', type=int, default=44100,
                        help='Target sample rate')
    parser.add_argument('--min_duration', type=float, default=10.0,
                        help='Minimum audio duration in seconds')
    parser.add_argument('--max_duration', type=float, default=600.0,
                        help='Maximum audio duration in seconds')
    
    # 품질 설정
    parser.add_argument('--min_snr', type=float, default=20.0,
                        help='Minimum SNR in dB')
    parser.add_argument('--min_complexity', type=float, default=0.3,
                        help='Minimum spectral complexity')
    
    # 데이터 분할
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation data ratio')
    
    # 성능 설정
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip already processed files')
    
    args = parser.parse_args()
    
    # 경로 설정
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root) if args.output_root else dataset_root
    
    # 프로세서 초기화
    audio_processor = AudioProcessor(
        target_sr=args.target_sr,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    
    quality_filter = QualityFilter(
        min_snr=args.min_snr,
        min_complexity=args.min_complexity
    ) if args.quality_filter else None
    
    # 1. 오디오 파일 찾기
    logger.info("Scanning for audio files...")
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    audio_files = []
    
    audio_dir = dataset_root / 'audio' / 'full'
    if audio_dir.exists():
        for ext in audio_extensions:
            audio_files.extend(audio_dir.glob(f'*{ext}'))
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # 2. 멀티프로세싱으로 오디오 처리
    if args.num_workers > 1:
        logger.info(f"Processing audio files with {args.num_workers} workers...")
        
        process_args = [(f, output_root, audio_processor) for f in audio_files]
        
        with mp.Pool(args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file, process_args),
                total=len(audio_files),
                desc="Processing audio"
            ))
    else:
        # 단일 프로세스
        results = []
        for audio_file in tqdm(audio_files, desc="Processing audio"):
            result = process_single_file((audio_file, output_root, audio_processor))
            results.append(result)
    
    # 3. 결과 정리
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    logger.info(f"Successfully processed: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if failed:
        logger.warning("Failed files:")
        for f in failed[:10]:  # 처음 10개만 표시
            logger.warning(f"  {f['audio_path']}: {f['status']}")
    
    # 4. 품질 필터링 (옵션)
    if quality_filter and successful:
        logger.info("Applying quality filter...")
        
        high_quality = []
        low_quality = []
        
        for item in tqdm(successful, desc="Quality filtering"):
            audio, sr = librosa.load(item['processed_path'], sr=None, mono=False)
            quality = quality_filter.check_quality(audio, sr)
            
            if quality['pass']:
                high_quality.append(item)
            else:
                low_quality.append(item)
                
        logger.info(f"High quality: {len(high_quality)}")
        logger.info(f"Low quality: {len(low_quality)}")
        
        # 고품질 데이터만 사용
        successful = high_quality
    
    # 5. 메타데이터 생성
    if args.generate_metadata:
        logger.info("Generating metadata...")
        
        metadata_gen = MetadataGenerator(dataset_root)
        metadata = metadata_gen.scan_dataset()
        
        # 처리 결과 반영
        processed_map = {r['audio_path'].stem: r for r in successful}
        
        for item in metadata:
            audio_id = Path(item['audio_path']).stem
            if audio_id in processed_map:
                result = processed_map[audio_id]
                item['processed'] = True
                item['duration'] = result.get('duration')
                item['checksum'] = result.get('checksum')
            else:
                item['processed'] = False
        
        # 처리된 항목만 필터링
        metadata = [m for m in metadata if m.get('processed', False)]
        
        # Train/Val/Test 분할
        train_data, val_data, test_data = metadata_gen.split_metadata(
            args.train_ratio,
            args.val_ratio
        )
        
        # 메타데이터 저장
        metadata_dir = output_root / 'metadata'
        metadata_dir.mkdir(exist_ok=True)
        
        # 전체 메타데이터
        with open(metadata_dir / 'metadata.jsonl', 'w', encoding='utf-8') as f:
            for item in metadata:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # Train/Val/Test 메타데이터
        for split_name, split_data in [
            ('train', train_data),
            ('val', val_data),
            ('test', test_data)
        ]:
            with open(metadata_dir / f'{split_name}_metadata.jsonl', 'w', encoding='utf-8') as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(split_data)} items to {split_name}_metadata.jsonl")
        
        # 고품질 데이터 선별 (Stage E용)
        if quality_filter:
            gold_data = [m for m in train_data if m.get('duration', 0) > 60]  # 1분 이상
            with open(metadata_dir / 'gold_metadata.jsonl', 'w', encoding='utf-8') as f:
                for item in gold_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(gold_data)} gold items")
    
    # 6. 통계 출력
    logger.info("\nDataset Statistics:")
    logger.info(f"Total files: {len(audio_files)}")
    logger.info(f"Processed: {len(successful)}")
    
    if successful:
        durations = [r['duration'] for r in successful if 'duration' in r]
        if durations:
            logger.info(f"Total duration: {sum(durations)/3600:.2f} hours")
            logger.info(f"Average duration: {np.mean(durations):.2f} seconds")
            logger.info(f"Min/Max duration: {min(durations):.2f}/{max(durations):.2f} seconds")
    
    logger.info("\nPreprocessing completed!")


if __name__ == '__main__':
    main()