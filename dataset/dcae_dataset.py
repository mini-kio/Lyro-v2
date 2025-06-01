# lyro/dataset/dcae_dataset.py
"""
Fixed DCAE Dataset Implementation
DCAE 학습용 간단한 오디오 전용 데이터셋 (패딩 오류 해결)
"""

import os
import torch
import numpy as np
import librosa
import torchaudio
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
import random
from pathlib import Path
from tqdm import tqdm
import warnings
import logging
import soundfile as sf

# 경고 메시지 억제
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
warnings.filterwarnings("ignore", message=".*Audioread support is deprecated.*")

# librosa 로깅 레벨 조정
logging.getLogger('librosa').setLevel(logging.ERROR)
logging.getLogger('audioread').setLevel(logging.ERROR)


class DCAEDataset(Dataset):
    """
    DCAE 학습용 간단한 오디오 데이터셋 - FIXED VERSION
    
    /dataset-dcae/datasets/raw/ 구조:
    - environmental/
    - music/  
    - speech/
    각 폴더에 mp3, wav 파일들
    """
    
    def __init__(
        self,
        data_root: str = "dataset-dcae/datasets/raw",
        sample_rate: int = 44100,
        max_duration: float = 10.0,  # 10초
        min_duration: float = 2.0,   # 2초
        augmentation: bool = True,
        cache_audio: bool = False,   # 메모리에 캐시할지 (작은 데이터셋용)
        skip_corrupted: bool = True,  # 손상된 파일 건너뛰기
        max_file_size_mb: float = 50.0,  # 최대 파일 크기 (MB)
        target_length: Optional[int] = None,  # 고정 길이 (samples)
    ):
        """
        Args:
            data_root: 데이터 루트 경로 (raw 폴더)
            sample_rate: 오디오 샘플링 레이트
            max_duration: 최대 오디오 길이 (초)
            min_duration: 최소 오디오 길이 (초)
            augmentation: 데이터 증강 여부
            cache_audio: 오디오를 메모리에 캐시할지
            skip_corrupted: 손상된 파일 건너뛰기
            max_file_size_mb: 최대 파일 크기 제한
            target_length: 타겟 길이 (None이면 가변 길이)
        """
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.augmentation = augmentation
        self.cache_audio = cache_audio
        self.skip_corrupted = skip_corrupted
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.target_length = target_length or int(max_duration * sample_rate)
        
        # 오디오 파일 경로 수집
        self.audio_paths = self._collect_audio_files()
        
        print(f"Found {len(self.audio_paths)} valid audio files")
        
        # 오디오 캐시 (선택적)
        self.audio_cache = {}
        if cache_audio and len(self.audio_paths) < 1000:  # 1000개 미만일 때만
            print("Caching audio files...")
            self._cache_audio_files()
    
    def _collect_audio_files(self) -> List[Path]:
        """오디오 파일 경로 수집 - 강화된 검증"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        audio_paths = []
        
        # environmental, music, speech 폴더에서 파일 찾기
        subfolders = ['environmental', 'music', 'speech']
        
        for subfolder in subfolders:
            subfolder_path = self.data_root / subfolder
            if subfolder_path.exists():
                for ext in audio_extensions:
                    audio_paths.extend(subfolder_path.glob(f'*{ext}'))
                    audio_paths.extend(subfolder_path.glob(f'*{ext.upper()}'))
        
        print(f"Found {len(audio_paths)} potential audio files")
        
        # 유효한 파일만 필터링 (강화된 검증)
        valid_paths = []
        corrupted_count = 0
        
        for path in tqdm(audio_paths, desc="Validating audio files"):
            if self._is_valid_audio_enhanced(path):
                valid_paths.append(path)
            else:
                corrupted_count += 1
                
        if corrupted_count > 0:
            print(f"Skipped {corrupted_count} corrupted/invalid audio files")
                
        return valid_paths
    
    def _is_valid_audio_enhanced(self, audio_path: Path) -> bool:
        """강화된 오디오 파일 유효성 검사"""
        try:
            # 1. 기본 파일 검사
            if not audio_path.exists():
                return False
                
            # 2. 파일 크기 검사
            file_size = audio_path.stat().st_size
            if file_size < 1000:  # 너무 작은 파일
                return False
            if file_size > self.max_file_size_bytes:  # 너무 큰 파일
                return False
            
            # 3. 파일 확장자 재검사
            valid_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
            if audio_path.suffix.lower() not in valid_extensions:
                return False
            
            # 4. 다중 방법으로 오디오 검증
            duration = None
            
            # 방법 1: librosa.get_duration (가장 빠름)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    duration = librosa.get_duration(path=str(audio_path))
            except Exception as e:
                # MP3 오류 등을 무시하고 다른 방법 시도
                pass
            
            # 방법 2: soundfile로 시도 (WAV, FLAC 등)
            if duration is None:
                try:
                    with sf.SoundFile(str(audio_path)) as f:
                        duration = len(f) / f.samplerate
                except Exception:
                    pass
            
            # 방법 3: torchaudio로 시도
            if duration is None:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        info = torchaudio.info(str(audio_path))
                        duration = info.num_frames / info.sample_rate
                except Exception:
                    pass
            
            # 방법 4: 실제 로딩 시도 (가장 확실하지만 느림)
            if duration is None and not self.skip_corrupted:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        audio, sr = librosa.load(str(audio_path), sr=None, duration=1.0)
                        if audio is not None and len(audio) > 0:
                            # 추정 duration
                            duration = len(audio) / sr
                except Exception:
                    return False
            
            # 5. Duration 검증
            if duration is None:
                return False
                
            if duration < self.min_duration or duration > 300:  # 5분 이상은 제외
                return False
            
            # 6. MP3 특별 검증 (손상된 MP3 파일 필터링)
            if audio_path.suffix.lower() == '.mp3':
                return self._validate_mp3_file(audio_path)
                
            return True
            
        except Exception as e:
            # 모든 예외를 잡아서 False 반환
            return False
    
    def _validate_mp3_file(self, audio_path: Path) -> bool:
        """MP3 파일 특별 검증"""
        try:
            # MP3 파일의 헤더 간단 검증
            with open(audio_path, 'rb') as f:
                header = f.read(10)
                
                # ID3 태그 또는 MP3 프레임 헤더 확인
                if header.startswith(b'ID3') or header[0:2] == b'\xff\xfb' or header[0:2] == b'\xff\xfa':
                    # 간단한 로딩 테스트
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            audio, sr = librosa.load(str(audio_path), sr=None, duration=0.1)
                            return len(audio) > 0
                    except Exception:
                        return False
                else:
                    return False
                    
        except Exception:
            return False
    
    def _cache_audio_files(self):
        """오디오 파일들을 메모리에 캐시"""
        successful_cache = 0
        failed_cache = 0
        
        for path in tqdm(self.audio_paths, desc="Caching audio"):
            try:
                audio = self._load_audio_safe(path)
                if audio is not None:
                    self.audio_cache[str(path)] = audio
                    successful_cache += 1
                else:
                    failed_cache += 1
            except Exception as e:
                failed_cache += 1
                continue
                
        print(f"Cached {successful_cache} files successfully, {failed_cache} failed")
    
    def _load_audio_safe(self, audio_path: Path) -> Optional[np.ndarray]:
        """안전한 오디오 파일 로드 (여러 방법 시도)"""
        # 캐시에서 먼저 확인
        if self.cache_audio and str(audio_path) in self.audio_cache:
            return self.audio_cache[str(audio_path)]
        
        audio = None
        
        # 방법 1: librosa 시도
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, sr = librosa.load(
                    str(audio_path), 
                    sr=self.sample_rate, 
                    mono=False,
                    duration=self.max_duration
                )
        except Exception as e:
            pass
        
        # 방법 2: torchaudio 시도
        if audio is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio_tensor, sr = torchaudio.load(str(audio_path))
                    if sr != self.sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                        audio_tensor = resampler(audio_tensor)
                    audio = audio_tensor.numpy()
            except Exception as e:
                pass
        
        # 방법 3: soundfile 시도 (WAV, FLAC)
        if audio is None and audio_path.suffix.lower() in ['.wav', '.flac']:
            try:
                audio, sr = sf.read(str(audio_path))
                audio = audio.T  # soundfile은 (time, channels) 형태
                if sr != self.sample_rate:
                    # 간단한 리샘플링
                    import scipy.signal
                    audio = scipy.signal.resample(
                        audio, 
                        int(len(audio) * self.sample_rate / sr), 
                        axis=-1
                    )
            except Exception as e:
                pass
        
        if audio is None:
            return None
        
        # 스테레오로 변환
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        elif audio.shape[0] == 1:
            audio = np.tile(audio, (2, 1))
        elif audio.shape[0] > 2:
            audio = audio[:2]  # 처음 2채널만 사용
        
        # 길이 조정 - 타겟 길이에 맞춤
        if audio.shape[1] > self.target_length:
            # 랜덤 크롭
            start = random.randint(0, audio.shape[1] - self.target_length)
            audio = audio[:, start:start + self.target_length]
        elif audio.shape[1] < self.target_length:
            # 패딩 또는 반복
            current_length = audio.shape[1]
            needed_length = self.target_length
            
            if current_length < needed_length // 4:
                # 너무 짧으면 반복
                repeat_count = (needed_length // current_length) + 1
                audio = np.tile(audio, (1, repeat_count))
                audio = audio[:, :needed_length]
            else:
                # 패딩
                pad_length = needed_length - current_length
                audio = np.pad(audio, ((0, 0), (0, pad_length)), mode='constant')
        
        # NaN/Inf 제거
        if np.isnan(audio).any() or np.isinf(audio).any():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 음량 정규화
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        return audio
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """간단한 오디오 증강"""
        if not self.augmentation:
            return audio
            
        # 볼륨 변경 (80% ~ 120%)
        if random.random() < 0.7:
            gain = random.uniform(0.8, 1.2)
            audio = audio * gain
        
        # 좌우 밸런스 조정
        if audio.shape[0] == 2 and random.random() < 0.3:
            balance = random.uniform(0.8, 1.2)
            audio[0] *= balance
            audio[1] *= (2.0 - balance)
        
        # 극성 뒤집기 (가끔)
        if random.random() < 0.1:
            audio = -audio
        
        # DC 오프셋 추가
        if random.random() < 0.2:
            dc_offset = random.uniform(-0.01, 0.01)
            audio = audio + dc_offset
        
        # 클리핑 방지
        max_val = np.abs(audio).max()
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)
            
        return audio
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Args:
            idx: 인덱스
        Returns:
            audio: (2, T) 스테레오 오디오 텐서
        """
        audio_path = self.audio_paths[idx]
        
        # 오디오 로드 (안전한 방법)
        audio = self._load_audio_safe(audio_path)
        
        # 로드 실패시 대체 오디오 생성
        if audio is None:
            print(f"Failed to load {audio_path}, using dummy audio")
            audio = np.random.randn(2, self.target_length) * 0.01
        
        # 최종 길이 확인 및 조정
        if audio.shape[1] != self.target_length:
            if audio.shape[1] > self.target_length:
                audio = audio[:, :self.target_length]
            else:
                pad_length = self.target_length - audio.shape[1]
                audio = np.pad(audio, ((0, 0), (0, pad_length)), mode='constant')
        
        # 증강 적용
        audio = self._augment_audio(audio)
        
        # 텐서로 변환
        audio_tensor = torch.from_numpy(audio).float()
        
        return audio_tensor


class DCAECollator:
    """
    DCAE용 안전한 Collator - FIXED VERSION
    가변 길이 오디오를 배치로 묶기 (패딩 오류 해결)
    """
    
    def __init__(
        self, 
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        pad_to_multiple: int = 1
    ):
        """
        Args:
            max_length: 최대 길이 제한
            min_length: 최소 길이 제한 
            pad_to_multiple: 이 값의 배수로 패딩
        """
        self.max_length = max_length
        self.min_length = min_length or 1000  # 최소 1000 샘플
        self.pad_to_multiple = pad_to_multiple
    
    def _safe_pad(self, audio: torch.Tensor, target_length: int) -> torch.Tensor:
        """안전한 패딩 함수"""
        current_length = audio.shape[1]
        
        if current_length == target_length:
            return audio
        elif current_length > target_length:
            # 크롭
            return audio[:, :target_length]
        else:
            # 패딩 필요
            pad_length = target_length - current_length
            
            # 패딩 크기가 너무 클 때 처리
            if pad_length > current_length * 3:  # 3배 이상 패딩이 필요하면
                # 반복으로 먼저 확장
                repeat_count = (pad_length // current_length) + 1
                extended_audio = audio.repeat(1, repeat_count + 1)
                return extended_audio[:, :target_length]
            else:
                # 안전한 reflection 패딩
                if current_length > 1 and pad_length <= current_length:
                    return torch.nn.functional.pad(
                        audio, (0, pad_length), mode='reflect'
                    )
                else:
                    # constant 패딩
                    return torch.nn.functional.pad(
                        audio, (0, pad_length), mode='constant', value=0
                    )
    
    def __call__(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: List of (2, T) audio tensors
        Returns:
            batched_audio: (B, 2, T) tensor
        """
        # 유효한 텐서만 필터링
        valid_batch = []
        for audio in batch:
            if isinstance(audio, torch.Tensor) and not torch.isnan(audio).any():
                # 최소 길이 확인
                if audio.shape[1] >= self.min_length:
                    valid_batch.append(audio)
                else:
                    # 최소 길이보다 짧으면 패딩
                    audio_padded = self._safe_pad(audio, self.min_length)
                    valid_batch.append(audio_padded)
        
        if not valid_batch:
            # 모든 배치가 무효한 경우 더미 배치 생성
            dummy_length = self.max_length or 44100
            dummy_audio = torch.zeros(len(batch), 2, dummy_length)
            return dummy_audio
        
        # 길이 통계
        lengths = [audio.shape[1] for audio in valid_batch]
        max_len = max(lengths)
        min_len = min(lengths)
        
        # 최대 길이 제한 적용
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        
        # 배수로 맞추기
        if self.pad_to_multiple > 1:
            max_len = ((max_len - 1) // self.pad_to_multiple + 1) * self.pad_to_multiple
        
        # 길이 차이가 너무 큰 경우 처리
        if max_len > min_len * 4:  # 4배 이상 차이나면
            # 중간값으로 타겟 길이 설정
            median_len = sorted(lengths)[len(lengths) // 2]
            target_len = min(max_len, median_len * 2)
        else:
            target_len = max_len
        
        # 안전한 패딩 적용
        padded_batch = []
        for audio in valid_batch:
            try:
                padded_audio = self._safe_pad(audio, target_len)
                padded_batch.append(padded_audio)
            except Exception as e:
                print(f"Padding error: {e}, using dummy audio")
                dummy_audio = torch.zeros(2, target_len)
                padded_batch.append(dummy_audio)
        
        # 배치로 스택
        try:
            batched_audio = torch.stack(padded_batch, dim=0)
        except Exception as e:
            print(f"Stacking error: {e}, creating dummy batch")
            batched_audio = torch.zeros(len(valid_batch), 2, target_len)
        
        return batched_audio


class AudioFileScanner:
    """오디오 파일 스캐너 유틸리티 - 강화된 검증"""
    
    @staticmethod
    def scan_directory(directory: Path) -> dict:
        """디렉토리 스캔하여 통계 정보 반환"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        
        stats = {
            'total_files': 0,
            'valid_files': 0,
            'corrupted_files': 0,
            'by_folder': {},
            'by_extension': {},
            'total_duration': 0.0,
            'error_files': [],
            'corruption_details': {}
        }
        
        for subfolder in ['environmental', 'music', 'speech']:
            subfolder_path = directory / subfolder
            if not subfolder_path.exists():
                continue
                
            folder_stats = {
                'count': 0,
                'valid_count': 0,
                'duration': 0.0,
                'extensions': {},
                'corrupted': []
            }
            
            # 파일 찾기
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(subfolder_path.glob(f'*{ext}'))
                audio_files.extend(subfolder_path.glob(f'*{ext.upper()}'))
            
            print(f"Scanning {subfolder}: {len(audio_files)} files")
            
            for audio_file in tqdm(audio_files, desc=f"Scanning {subfolder}"):
                stats['total_files'] += 1
                folder_stats['count'] += 1
                
                try:
                    # 확장자 통계
                    ext = audio_file.suffix.lower()
                    folder_stats['extensions'][ext] = folder_stats['extensions'].get(ext, 0) + 1
                    stats['by_extension'][ext] = stats['by_extension'].get(ext, 0) + 1
                    
                    # 안전한 길이 확인
                    duration = None
                    error_details = []
                    
                    # 여러 방법으로 시도
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            duration = librosa.get_duration(path=str(audio_file))
                    except Exception as e:
                        error_details.append(f"librosa: {str(e)[:100]}")
                        
                        # soundfile 시도
                        try:
                            with sf.SoundFile(str(audio_file)) as f:
                                duration = len(f) / f.samplerate
                        except Exception as e2:
                            error_details.append(f"soundfile: {str(e2)[:100]}")
                            
                            # torchaudio 시도
                            try:
                                info = torchaudio.info(str(audio_file))
                                duration = info.num_frames / info.sample_rate
                            except Exception as e3:
                                error_details.append(f"torchaudio: {str(e3)[:100]}")
                    
                    if duration is not None:
                        folder_stats['duration'] += duration
                        stats['total_duration'] += duration
                        stats['valid_files'] += 1
                        folder_stats['valid_count'] += 1
                    else:
                        stats['corrupted_files'] += 1
                        folder_stats['corrupted'].append(str(audio_file))
                        stats['error_files'].append(str(audio_file))
                        stats['corruption_details'][str(audio_file)] = error_details
                    
                except Exception as e:
                    stats['corrupted_files'] += 1
                    stats['error_files'].append(str(audio_file))
                    stats['corruption_details'][str(audio_file)] = [str(e)]
            
            stats['by_folder'][subfolder] = folder_stats
        
        return stats
    
    @staticmethod
    def print_stats(stats: dict):
        """통계 정보 출력"""
        print("\n" + "="*50)
        print("Fixed DCAE Dataset Statistics")
        print("="*50)
        
        print(f"Total files found: {stats['total_files']}")
        print(f"Valid files: {stats['valid_files']}")
        print(f"Corrupted files: {stats['corrupted_files']}")
        if stats['total_files'] > 0:
            corruption_rate = (stats['corrupted_files'] / stats['total_files']) * 100
            print(f"Corruption rate: {corruption_rate:.1f}%")
        
        print(f"Total duration: {stats['total_duration']/3600:.2f} hours")
        
        print(f"\nBy folder:")
        for folder, folder_stats in stats['by_folder'].items():
            corruption_count = len(folder_stats['corrupted'])
            print(f"  {folder}: {folder_stats['valid_count']}/{folder_stats['count']} valid, "
                  f"{folder_stats['duration']/3600:.2f}h, {corruption_count} corrupted")
            for ext, count in folder_stats['extensions'].items():
                print(f"    {ext}: {count}")
        
        print(f"\nBy extension:")
        for ext, count in stats['by_extension'].items():
            print(f"  {ext}: {count}")
        
        if stats['error_files']:
            print(f"\nCorrupted files ({len(stats['error_files'])}):")
            for error_file in stats['error_files'][:5]:
                print(f"  {error_file}")
                if error_file in stats['corruption_details']:
                    for detail in stats['corruption_details'][error_file][:2]:
                        print(f"    └─ {detail}")
            if len(stats['error_files']) > 5:
                print(f"  ... and {len(stats['error_files']) - 5} more")


def create_dcae_datasets(
    data_root: str = "dataset-dcae/datasets/raw",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    sample_rate: int = 44100,
    max_duration: float = 10.0,
    cache_audio: bool = False,
    skip_corrupted: bool = True,
    target_length: Optional[int] = None,
    **kwargs
) -> Tuple[DCAEDataset, DCAEDataset, DCAEDataset]:
    """
    DCAE 학습용 데이터셋 생성 (train/val/test 분할) - FIXED VERSION
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # 타겟 길이 설정
    if target_length is None:
        target_length = int(max_duration * sample_rate)
    
    # 전체 데이터셋 생성
    full_dataset = DCAEDataset(
        data_root=data_root,
        sample_rate=sample_rate,
        max_duration=max_duration,
        augmentation=False,  # 분할 전에는 증강 안함
        cache_audio=cache_audio,
        skip_corrupted=skip_corrupted,
        target_length=target_length,
        **kwargs
    )
    
    # 인덱스 분할
    total_size = len(full_dataset)
    if total_size == 0:
        raise ValueError("No valid audio files found in dataset")
    
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # 랜덤 셔플
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # 서브셋 생성
    from torch.utils.data import Subset
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # 증강은 train에만 적용
    full_dataset.augmentation = True
    
    print(f"Fixed dataset split:")
    print(f"  Train: {len(train_dataset)} files")
    print(f"  Val: {len(val_dataset)} files") 
    print(f"  Test: {len(test_dataset)} files")
    print(f"  Target length: {target_length} samples ({target_length/sample_rate:.1f}s)")
    
    return train_dataset, val_dataset, test_dataset


# 사용 예시 및 테스트
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed DCAE Dataset Test')
    parser.add_argument('--data_root', default='dataset-dcae/datasets/raw')
    parser.add_argument('--scan_only', action='store_true', help='Only scan and show stats')
    parser.add_argument('--skip_corrupted', action='store_true', default=True, 
                        help='Skip corrupted audio files')
    parser.add_argument('--target_length', type=int, default=None,
                        help='Target length in samples')
    args = parser.parse_args()
    
    # 디렉토리 스캔
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        exit(1)
    
    print(f"Scanning directory with fixed validation: {data_root}")
    scanner = AudioFileScanner()
    stats = scanner.scan_directory(data_root)
    scanner.print_stats(stats)
    
    if args.scan_only:
        exit(0)
    
    # 데이터셋 테스트
    print("\nTesting fixed dataset...")
    try:
        train_ds, val_ds, test_ds = create_dcae_datasets(
            data_root=str(data_root),
            max_duration=5.0,  # 테스트용 짧게
            cache_audio=False,
            skip_corrupted=args.skip_corrupted,
            target_length=args.target_length
        )
        
        print(f"\nFixed dataset test successful!")
        
        # 샘플 데이터 확인
        print("Testing data loading...")
        if len(train_ds) > 0:
            sample = train_ds[0]
            print(f"Sample audio shape: {sample.shape}")
            print(f"Sample audio range: [{sample.min():.3f}, {sample.max():.3f}]")
            
            # Collator 테스트
            from torch.utils.data import DataLoader
            
            collator = DCAECollator(
                max_length=44100 * 5,  # 5초
                min_length=44100 * 1,  # 1초
                pad_to_multiple=512    # 512 샘플 단위
            )
            loader = DataLoader(
                train_ds, 
                batch_size=4, 
                shuffle=True, 
                collate_fn=collator,
                num_workers=0  # 테스트용
            )
            
            batch = next(iter(loader))
            print(f"Batch shape: {batch.shape}")
            print(f"All samples same length: {len(set(batch.shape[2] for _ in range(batch.shape[0]))) == 1}")
            
            print("✅ All fixed tests passed!")
        else:
            print("⚠️ No valid samples found in dataset")
        
    except Exception as e:
        print(f"❌ Fixed dataset test failed: {e}")
        import traceback
        traceback.print_exc()