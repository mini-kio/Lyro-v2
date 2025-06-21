# lyro/dataset/dcae_dataset.py
"""
DDP Compatible S6-SSM Compression Optimized DCAE Dataset Implementation
STEREO ENHANCED: 스테레오 형태 보장 및 채널 처리 개선
SIMPLIFIED: 음질 평가 완전 제거, fast_mode에서 간단한 오류 검사만 수행
FIXED: libmpg123 등 오류 파일 미리 제거로 안정성 향상
"""

import os
import torch
import numpy as np
import librosa
import torchaudio
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Dict, Any
import random
from pathlib import Path
from tqdm import tqdm
import warnings
import logging
import soundfile as sf
import threading
import time
from collections import deque, defaultdict
import multiprocessing
from functools import lru_cache
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import psutil

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
warnings.filterwarnings("ignore", message=".*Audioread support is deprecated.*")

# Configure logging
logging.getLogger('librosa').setLevel(logging.ERROR)
logging.getLogger('audioread').setLevel(logging.ERROR)


def safe_str(obj):
    """Safely convert object to string for DDP compatibility"""
    try:
        if isinstance(obj, (str, Path)):
            text = str(obj)
            return text.encode('utf-8', errors='replace').decode('utf-8')
        else:
            return str(obj)
    except Exception:
        return "<invalid_unicode>"


def safe_print(*args, **kwargs):
    """DDP compatible safe print"""
    try:
        safe_args = [safe_str(arg) for arg in args]
        print(*safe_args, **kwargs)
    except UnicodeEncodeError:
        try:
            safe_args = [str(arg).encode('ascii', errors='replace').decode('ascii') for arg in args]
            print(*safe_args, **kwargs)
        except Exception:
            print("<Unicode encoding error in print>")


def ensure_stereo_audio(audio: np.ndarray) -> np.ndarray:
    """
    STEREO ENHANCED: 오디오를 스테레오 형태로 변환
    Input: (T,) 모노 또는 (C, T) 멀티채널
    Output: (2, T) 스테레오
    """
    if audio.ndim == 1:  # 모노 (T,)
        # 스테레오로 복제
        audio = np.stack([audio, audio], axis=0)  # (2, T)
    elif audio.ndim == 2:  # 멀티채널 (C, T)
        if audio.shape[0] == 1:  # 모노 채널
            audio = np.tile(audio, (2, 1))  # (2, T)
        elif audio.shape[0] > 2:  # 멀티채널
            audio = audio[:2, :]  # 첫 2채널만 사용
        # audio.shape[0] == 2인 경우는 그대로 유지
    else:
        # 예상치 못한 차원: 안전한 기본값으로 변환
        if audio.size > 0:
            audio_flat = audio.flatten()
            audio = np.stack([audio_flat, audio_flat], axis=0)
        else:
            audio = np.zeros((2, 44100))  # 기본 스테레오 무음
    
    return audio


class DDPCompatibleFileManager:
    """
    DDP Compatible file manager - SIMPLIFIED VERSION
    음질 평가 제거, 간단한 오류 검사만 수행
    """
    
    def __init__(
        self, 
        enable_deletion: bool = True, 
        backup_corrupted: bool = False,
        max_workers: Optional[int] = None,
        fast_mode: bool = False
    ):
        self.enable_deletion = enable_deletion
        self.backup_corrupted = backup_corrupted
        self.fast_mode = fast_mode
        
        # Thread safety for DDP
        self.lock = threading.Lock()
        self._logged_files = set()
        
        # SIMPLIFIED: Static results storage for DDP
        self.corrupted_files = set()
        self.deletion_log = []
        
        # Static parallel processing for DDP (V100 optimized)
        num_workers = os.cpu_count() or 1
        self.max_workers = max_workers or max(2, num_workers // 4)  # DDP optimized
        
        # Create backup directory
        if self.backup_corrupted:
            self.backup_dir = Path("corrupted_files_backup")
            self.backup_dir.mkdir(exist_ok=True)
    
    def _is_libmpg123_error(self, error_msg: str) -> bool:
        """
        libmpg123 오류 패턴 감지 (확장된 패턴)
        part2_3_length 오류 등 포함
        """
        error_lower = error_msg.lower()
        libmpg123_patterns = [
            'dequantization failed', 'part2_3_length', 'layer3',
            'one-frame stream', 'cannot read next header',
            'mpg123', 'mp3 decode', 'invalid',
            'too large for available bit count',
            'INT123_do_layer3', 'libmpg123', 'bit count',
            'available bit', 'src/libmpg123',
            'part2_3_length', 'bit count'
        ]
        return any(pattern in error_lower for pattern in libmpg123_patterns)
    
    def quick_validate_file(self, file_path: Path) -> bool:
        """
        SIMPLIFIED: 빠른 파일 검증 - 음질 평가 없음
        libmpg123 등 오류 파일 미리 감지
        """
        try:
            # 기본 파일 검사
            if not file_path.exists():
                return False
            
            file_size = file_path.stat().st_size
            if file_size < 1000:  # 1KB 최소
                return False
            
            # fast_mode에서는 여기서 중단
            if self.fast_mode:
                return True
            
            # 간단한 헤더 검사로 오류 파일 감지
            try:
                # librosa duration check (빠름)
                duration = librosa.get_duration(path=str(file_path))
                if duration <= 0 or duration > 300:  # 5분 최대
                    return False
            except Exception as e:
                error_msg = str(e).lower()
                # libmpg123 관련 오류 감지 (확장된 패턴)
                if any(keyword in error_msg for keyword in [
                    'dequantization failed', 'part2_3_length', 'layer3',
                    'one-frame stream', 'cannot read next header',
                    'mpg123', 'mp3 decode', 'invalid mp3',
                    'too large for available bit count',
                    'INT123_do_layer3', 'libmpg123', 'bit count',
                    'part2_3_length', 'available bit'
                ]):
                    return False
                return False
            
            # 매우 간단한 로딩 테스트 (0.1초만)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    audio, sr = librosa.load(str(file_path), sr=None, duration=0.1)
                    
                    if len(audio) == 0:
                        return False
                    
                    if np.isnan(audio).any() or np.isinf(audio).any():
                        return False
            except Exception as e:
                if self._is_libmpg123_error(str(e)):
                    return False
                return False
            
            return True
            
        except Exception:
            return False
    
    def validate_files_batch(self, file_paths: List[Path]) -> Dict[Path, bool]:
        """
        SIMPLIFIED: 배치 파일 검증 - 음질 평가 없음
        """
        results = {}
        
        if self.fast_mode:
            # fast_mode에서는 기본 검사만
            for file_path in file_paths:
                results[file_path] = self.quick_validate_file(file_path)
            return results
        
        def validate_single_file(file_path):
            return file_path, self.quick_validate_file(file_path)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(validate_single_file, fp): fp for fp in file_paths}
            
            for future in as_completed(future_to_file):
                try:
                    file_path, is_valid = future.result()
                    results[file_path] = is_valid
                    
                    if not is_valid:
                        self.mark_corrupted(file_path, "Failed validation")
                        
                except Exception:
                    file_path = future_to_file[future]
                    results[file_path] = False
                    self.mark_corrupted(file_path, "Validation exception")
        
        return results
    
    def mark_corrupted(self, file_path: Path, error_msg: str = ""):
        """Mark file as corrupted - DDP compatible thread-safe"""
        with self.lock:
            file_key = str(file_path)
            if file_key in self._logged_files:
                return
            self._logged_files.add(file_key)
            
            if file_path in self.corrupted_files:
                return
            
            self.corrupted_files.add(file_path)
            
            if self.enable_deletion:
                try:
                    if self.backup_corrupted and file_path.exists():
                        backup_path = self.backup_dir / file_path.name
                        file_path.rename(backup_path)
                        action = f"moved to backup"
                    else:
                        if file_path.exists():
                            file_path.unlink()
                            action = "deleted"
                        else:
                            action = "already missing"
                    
                    log_entry = {
                        'file': safe_str(file_path),
                        'error': safe_str(error_msg),
                        'action': action,
                        'timestamp': time.time()
                    }
                    self.deletion_log.append(log_entry)
                    
                except Exception:
                    pass
    
    def is_corrupted(self, file_path: Path) -> bool:
        """Check if file is marked as corrupted - DDP compatible"""
        return file_path in self.corrupted_files
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simplified statistics - DDP compatible"""
        return {
            'total_corrupted': len(self.corrupted_files),
            'total_deleted': len(self.deletion_log),
            'deletion_enabled': self.enable_deletion,
            'backup_enabled': self.backup_corrupted,
            'max_workers': self.max_workers,
            'fast_mode': self.fast_mode,
            'ddp_compatible': True,
            'v100_optimized': True,
            'simplified': True,  # 음질 평가 제거됨
            'stereo_enhanced': True  # STEREO ENHANCED
        }
    
    def save_logs(self, log_dir: Path):
        """Save deletion logs - DDP compatible"""
        log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save deletion log only
            deletion_log_path = log_dir / "ddp_s6_ssm_stereo_deletion_log.json"
            with open(deletion_log_path, 'w', encoding='utf-8', errors='replace') as f:
                json.dump(self.deletion_log, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


class DDPCompatibleAudioValidator:
    """
    DDP compatible audio validator - SIMPLIFIED VERSION
    음질 평가 제거, 간단한 검증만
    """
    
    def __init__(self, file_manager: DDPCompatibleFileManager):
        self.file_manager = file_manager
        self.validation_cache = {}
        self.cache_lock = threading.Lock()
    
    def validate_audio_batch(self, audio_paths: List[Path]) -> Dict[Path, bool]:
        """
        SIMPLIFIED: 배치 검증 - 음질 평가 없음
        """
        return self.file_manager.validate_files_batch(audio_paths)
    
    def validate_and_filter(self, audio_path: Path) -> bool:
        """
        SIMPLIFIED: 검증 및 필터링
        """
        with self.cache_lock:
            cache_key = safe_str(audio_path)
            if cache_key in self.validation_cache:
                return self.validation_cache[cache_key]
        
        # Check if already marked
        if self.file_manager.is_corrupted(audio_path):
            with self.cache_lock:
                self.validation_cache[cache_key] = False
            return False
        
        # Perform simple validation
        is_valid = self.file_manager.quick_validate_file(audio_path)
        
        if not is_valid:
            self.file_manager.mark_corrupted(audio_path, "Failed validation")
            
        with self.cache_lock:
            self.validation_cache[cache_key] = is_valid
        return is_valid


class StereoEnhancedS6SSMDataset(Dataset):
    """
    STEREO ENHANCED DDP Compatible S6-SSM Compression Optimized DCAE Dataset
    SIMPLIFIED: 음질 평가 제거, 간단한 오류 검사만
    STEREO ENHANCED: 스테레오 형태 보장 및 채널 처리 개선
    """
    
    def __init__(
        self,
        data_root: str = "dataset-dcae/datasets/raw",
        sample_rate: int = 44100,
        max_duration: float = 10.0,
        min_duration: float = 1.0,
        augmentation: bool = True,
        cache_audio: bool = False,
        skip_corrupted: bool = True,
        max_file_size_mb: float = 30.0,  # V100 optimized
        target_length: Optional[int] = None,
        auto_delete_corrupted: bool = True,
        backup_corrupted: bool = False,
        max_retries: int = 3,
        # V100 memory optimization
        memory_efficient_loading: bool = True,
        intelligent_caching: bool = True,
        cache_size_limit_mb: float = 500.0,
        # DDP compatible normalization
        preserve_peak_db: float = -1.0,
        normalize_for_compression: bool = True,
        # V100 optimized parallel processing
        max_workers: Optional[int] = None,
        batch_size_validation: int = 50,
        # DDP compatible fast mode
        fast_mode: bool = False,
        skip_validation: bool = False,
        use_cached_list: bool = True,
        # STEREO ENHANCED parameters
        ensure_stereo: bool = True,
        stereo_processing_mode: str = 'independent'
    ):
        """
        STEREO ENHANCED DDP Compatible S6-SSM Compression Optimized Dataset
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
        self.max_retries = max_retries
        self.batch_size_validation = batch_size_validation
        
        # STEREO ENHANCED parameters
        self.ensure_stereo = ensure_stereo
        self.stereo_processing_mode = stereo_processing_mode
        
        # SIMPLIFIED: fast mode settings
        self.fast_mode = fast_mode
        self.skip_validation = skip_validation
        self.use_cached_list = use_cached_list
        
        # SIMPLIFIED: fast mode auto settings
        if self.fast_mode:
            self.skip_validation = True
            self.auto_delete_corrupted = False
            safe_print("🚀 DDP STEREO Fast Mode: 음질 평가 제거, libmpg123 오류는 런타임 감지!")
        
        # SIMPLIFIED: basic parameters only
        self.memory_efficient_loading = memory_efficient_loading
        self.intelligent_caching = intelligent_caching
        self.preserve_peak_db = preserve_peak_db
        self.normalize_for_compression = normalize_for_compression
        
        # SIMPLIFIED: enhanced file manager
        self.file_manager = DDPCompatibleFileManager(
            enable_deletion=auto_delete_corrupted and not self.fast_mode,
            backup_corrupted=backup_corrupted,
            max_workers=max_workers,
            fast_mode=self.fast_mode
        )
        
        # SIMPLIFIED: enhanced validator
        self.validator = DDPCompatibleAudioValidator(self.file_manager)
        
        # SIMPLIFIED: collect and validate audio files
        if self.fast_mode or self.skip_validation:
            self.audio_paths = self._collect_audio_files_fast()
        else:
            self.audio_paths = self._collect_and_validate_audio_files_parallel()
        
        safe_print(f"✅ DDP STEREO S6-SSM Dataset: {len(self.audio_paths)} files ready")
        
        # SIMPLIFIED: statistics
        if not self.fast_mode:
            stats = self.file_manager.get_stats()
            if stats['total_corrupted'] > 0:
                safe_print(f"🗑️  Processed {stats['total_corrupted']} corrupted files")
                if stats['deletion_enabled']:
                    safe_print(f"✅ Cleaned {stats['total_deleted']} files")
        
        # SIMPLIFIED: intelligent caching setup
        self.audio_cache = {}
        self.cache_usage = defaultdict(int)
        self.cache_size_bytes = 0
        self.cache_size_limit_bytes = int(cache_size_limit_mb * 1024 * 1024)
        
        if (cache_audio and len(self.audio_paths) < 500 and  # V100 optimized
            self.intelligent_caching and not self.fast_mode):
            safe_print("Setting up V100 STEREO optimized intelligent caching...")
            self._setup_intelligent_cache()
        
        # SIMPLIFIED: save logs
        if not self.fast_mode:
            stats = self.file_manager.get_stats() 
            if stats['total_deleted'] > 0:
                log_dir = self.data_root / "ddp_s6_ssm_stereo_compression_logs"
                self.file_manager.save_logs(log_dir)
                safe_print(f"📝 DDP STEREO S6-SSM logs saved to: {safe_str(log_dir)}")
    
    def _collect_audio_files_fast(self) -> List[Path]:
        """DDP compatible fast mode file collection"""
        safe_print("🚀 DDP STEREO Fast Mode: Quick file collection...")
        
        # SIMPLIFIED: cached file list check
        cache_file = self.data_root / "ddp_s6_ssm_stereo_file_cache.json"
        if self.use_cached_list and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_paths = json.load(f)
                
                audio_paths = []
                for path_str in cached_paths:
                    path = Path(path_str)
                    if path.exists() and path.stat().st_size > 1000:  # 간단한 크기 체크만
                        audio_paths.append(path)
                
                if len(audio_paths) > 50:
                    safe_print(f"📋 Using cached STEREO file list: {len(audio_paths)} files")
                    return audio_paths
            except Exception:
                pass
        
        # SIMPLIFIED: quick file collection
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        audio_paths = []
        
        subfolders = ['environmental', 'music', 'speech']
        
        for subfolder in subfolders:
            subfolder_path = self.data_root / subfolder
            if not subfolder_path.exists():
                continue
            
            # SIMPLIFIED: quick file search
            for ext in audio_extensions:
                try:
                    folder_files = list(subfolder_path.glob(f'*{ext}'))
                    folder_files.extend(subfolder_path.glob(f'*{ext.upper()}'))
                    
                    # SIMPLIFIED: basic file size check only
                    for file_path in folder_files:
                        try:
                            if file_path.stat().st_size > 1000:  # 1KB 최소
                                audio_paths.append(file_path)
                        except Exception:
                            continue
                except Exception:
                    continue
            
            safe_print(f"📁 STEREO {subfolder}: {len([p for p in audio_paths if subfolder in str(p)])} files")
        
        # SIMPLIFIED: save file list cache
        try:
            with open(cache_file, 'w') as f:
                json.dump([str(p) for p in audio_paths], f)
        except Exception:
            pass
        
        return audio_paths
    
    def _collect_and_validate_audio_files_parallel(self) -> List[Path]:
        """DDP compatible parallel file collection and validation"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        audio_paths = []
        
        safe_print(f"🔍 DDP STEREO S6-SSM: Collecting files in: {safe_str(self.data_root)}")
        
        subfolders = ['environmental', 'music', 'speech']
        total_found = 0
        
        for subfolder in subfolders:
            subfolder_path = self.data_root / subfolder
            
            if not subfolder_path.exists():
                safe_print(f"⚠️  Subfolder not found: {subfolder}")
                continue
            
            # Find all audio files
            folder_files = []
            for ext in audio_extensions:
                try:
                    folder_files.extend(subfolder_path.glob(f'*{ext}'))
                    folder_files.extend(subfolder_path.glob(f'*{ext.upper()}'))
                except Exception:
                    continue
            
            total_found += len(folder_files)
            safe_print(f"📁 STEREO {subfolder}: {len(folder_files)} files found")
            
            # SIMPLIFIED: parallel validation
            if folder_files:
                safe_print(f"🔍 DDP STEREO S6-SSM validation: {len(folder_files)} files in {subfolder}...")
                
                valid_files = self._validate_files_in_batches(folder_files, subfolder)
                audio_paths.extend(valid_files)
        
        safe_print(f"📊 DDP STEREO S6-SSM: Found {total_found}, Valid: {len(audio_paths)}")
        return audio_paths
    
    def _validate_files_in_batches(self, file_paths: List[Path], subfolder: str) -> List[Path]:
        """DDP compatible file validation in parallel batches"""
        valid_files = []
        
        # Process files in batches
        batches = [file_paths[i:i + self.batch_size_validation] 
                  for i in range(0, len(file_paths), self.batch_size_validation)]
        
        with tqdm(total=len(file_paths), desc=f"Validating STEREO {subfolder}", leave=False) as pbar:
            for batch in batches:
                try:
                    # Filter invalid Unicode files
                    valid_batch = []
                    for file_path in batch:
                        safe_path_str = safe_str(file_path)
                        if "<invalid_unicode>" not in safe_path_str:
                            valid_batch.append(file_path)
                    
                    if not valid_batch:
                        pbar.update(len(batch))
                        continue
                    
                    # SIMPLIFIED: parallel validation
                    validation_results = self.validator.validate_audio_batch(valid_batch)
                    
                    # Collect valid files
                    for file_path, is_valid in validation_results.items():
                        if is_valid:
                            valid_files.append(file_path)
                    
                    pbar.update(len(batch))
                    
                except Exception:
                    pbar.update(len(batch))
                    continue
        
        return valid_files
    
    def _setup_intelligent_cache(self):
        """DDP compatible intelligent caching setup (V100 optimized)"""
        cache_candidates = []
        
        # Sample first 50 files for V100 DDP
        for path in self.audio_paths[:50]:
            try:
                # Quick analysis for caching decision
                file_size = path.stat().st_size
                
                # Prefer smaller files for V100 caching
                if file_size < self.max_file_size_bytes // 3:
                    priority = 1.0 / (file_size + 1)
                    cache_candidates.append((path, priority))
            except Exception:
                continue
        
        # Sort by priority and cache
        cache_candidates.sort(key=lambda x: x[1], reverse=True)
        
        cached_count = 0
        for path, priority in cache_candidates:
            if self.cache_size_bytes > self.cache_size_limit_bytes:
                break
            
            try:
                audio = self._load_audio_safe(path)
                if audio is not None:
                    cache_key = safe_str(path)
                    self.audio_cache[cache_key] = audio
                    
                    # Estimate cache size
                    audio_size = audio.nbytes if hasattr(audio, 'nbytes') else len(audio) * 8
                    self.cache_size_bytes += audio_size
                    cached_count += 1
            except Exception:
                continue
        
        safe_print(f"🧠 V100 STEREO intelligent cache: {cached_count} files cached")
    
    def _is_libmpg123_error(self, error_msg: str) -> bool:
        """
        libmpg123 오류 패턴 감지 (확장된 패턴)
        part2_3_length 오류 등 포함
        """
        error_lower = error_msg.lower()
        libmpg123_patterns = [
            'dequantization failed', 'part2_3_length', 'layer3',
            'one-frame stream', 'cannot read next header',
            'mpg123', 'mp3 decode', 'invalid',
            'too large for available bit count',
            'INT123_do_layer3', 'libmpg123', 'bit count',
            'available bit', 'src/libmpg123',
            'part2_3_length', 'bit count'
        ]
        return any(pattern in error_lower for pattern in libmpg123_patterns)
    
    def _load_audio_safe(self, audio_path: Path) -> Optional[np.ndarray]:
        """
        STEREO ENHANCED DDP compatible memory-efficient audio loading (V100 optimized)
        SIMPLIFIED: 음질 평가 제거
        """
        cache_key = safe_str(audio_path)
        
        # Check intelligent cache
        if self.intelligent_caching and cache_key in self.audio_cache:
            self.cache_usage[cache_key] += 1
            return self.audio_cache[cache_key]
        
        # Check file manager status
        if self.file_manager.is_corrupted(audio_path):
            return None
        
        audio = None
        
        # Memory-efficient loading for V100 DDP
        if self.memory_efficient_loading:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    audio, sr = librosa.load(
                        str(audio_path), 
                        sr=self.sample_rate, 
                        mono=False,
                        duration=min(self.max_duration, 10.0)  # V100 optimized duration
                    )
            except Exception as e:
                error_msg = str(e)
                if self._is_libmpg123_error(error_msg):
                    self.file_manager.mark_corrupted(audio_path, error_msg)
                    safe_print(f"🗑️ libmpg123 오류 감지: {audio_path.name} - {error_msg[:50]}...")
                    return None
        
        # Fallback loading
        if audio is None:
            try:
                audio_tensor, sr = torchaudio.load(str(audio_path))
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    audio_tensor = resampler(audio_tensor)
                audio = audio_tensor.numpy()
            except Exception as e:
                self.file_manager.mark_corrupted(audio_path, f"TorchaAudio: {str(e)}")
                return None
        
        if audio is None:
            self.file_manager.mark_corrupted(audio_path, "All loading methods failed")
            return None
        
        # STEREO ENHANCED: 스테레오 형태 보장
        try:
            # STEREO ENHANCED: Ensure stereo format
            if self.ensure_stereo:
                audio = ensure_stereo_audio(audio)
            else:
                # Ensure stereo format for compatibility
                if audio.ndim == 1:
                    audio = np.stack([audio, audio], axis=0)
                elif audio.shape[0] == 1:
                    audio = np.tile(audio, (2, 1))
                elif audio.shape[0] > 2:
                    audio = audio[:2]
            
            # Length adjustment
            current_length = audio.shape[1]
            target_length = self.target_length
            
            if current_length > target_length:
                # Smart cropping
                start_options = [0, current_length - target_length]
                if current_length > target_length * 2:
                    start_options.append((current_length - target_length) // 2)
                
                start = random.choice(start_options)
                audio = audio[:, start:start + target_length]
            elif current_length < target_length:
                # Smart padding/repetition
                if current_length < target_length // 4:
                    repeat_count = (target_length // current_length) + 1
                    audio = np.tile(audio, (1, repeat_count))
                    audio = audio[:, :target_length]
                else:
                    pad_length = target_length - current_length
                    audio = np.pad(audio, ((0, 0), (0, pad_length)), mode='constant')
            
            # Basic quality checks
            if np.isnan(audio).any() or np.isinf(audio).any():
                self.file_manager.mark_corrupted(audio_path, "Contains NaN or Inf")
                return None
            
            # SIMPLIFIED: basic normalization
            if self.normalize_for_compression:
                max_val = np.abs(audio).max()
                if max_val > 0:
                    target_peak = 10 ** (self.preserve_peak_db / 20)
                    audio = audio / max_val * target_peak
                else:
                    self.file_manager.mark_corrupted(audio_path, "Silent audio")
                    return None
            
            return audio
            
        except Exception as e:
            self.file_manager.mark_corrupted(audio_path, f"Post-processing: {str(e)}")
            return None
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        STEREO ENHANCED DDP compatible get item
        Output: (2, T) 스테레오 텐서
        """
        for retry in range(self.max_retries):
            try:
                if idx >= len(self.audio_paths):
                    idx = random.randint(0, len(self.audio_paths) - 1)
                audio_path = self.audio_paths[idx]
                # Load audio
                audio = self._load_audio_safe(audio_path)
                
                if audio is None:
                    # Find another valid file
                    available_indices = [
                        i for i, p in enumerate(self.audio_paths) 
                        if not self.file_manager.is_corrupted(p)
                    ]
                    
                    if available_indices:
                        idx = random.choice(available_indices)
                        continue
                    else:
                        # Generate dummy stereo audio
                        audio = self._generate_dummy_stereo_audio()
                        break
                
                # Final length verification
                if audio.shape[1] != self.target_length:
                    if audio.shape[1] > self.target_length:
                        audio = audio[:, :self.target_length]
                    else:
                        pad_length = self.target_length - audio.shape[1]
                        audio = np.pad(audio, ((0, 0), (0, pad_length)), mode='constant')
                
                # STEREO ENHANCED: Final stereo check
                if self.ensure_stereo and audio.shape[0] != 2:
                    audio = ensure_stereo_audio(audio)
                
                # Convert to tensor
                audio_tensor = torch.from_numpy(audio).float()
                
                return audio_tensor
                
            except Exception as e:
                # 런타임 오류 패턴 감지 및 파일 마킹
                error_msg = str(e)
                if self._is_libmpg123_error(error_msg):
                    # libmpg123 오류로 확인된 파일 마킹
                    self.file_manager.mark_corrupted(audio_path, error_msg)
                    safe_print(f"🗑️ 런타임 libmpg123 오류 감지: {audio_path.name}")
                
                if retry < self.max_retries - 1:
                    idx = random.randint(0, len(self.audio_paths) - 1)
                    continue
                else:
                    # Generate dummy stereo audio
                    audio = self._generate_dummy_stereo_audio()
                    return torch.from_numpy(audio).float()
    
    def _generate_dummy_stereo_audio(self) -> np.ndarray:
        """STEREO ENHANCED dummy audio generation"""
        # Generate pink noise for both channels
        audio_l = np.random.randn(self.target_length) * 0.02
        audio_r = np.random.randn(self.target_length) * 0.02
        
        # Add harmonic content with stereo difference
        t = np.linspace(0, self.target_length / self.sample_rate, self.target_length)
        harmonic_l = 0.05 * np.sin(2 * np.pi * 440 * t)  # A4 note
        harmonic_r = 0.05 * np.sin(2 * np.pi * 440 * t + np.pi/4)  # Phase shifted
        
        audio_l += harmonic_l
        audio_r += harmonic_r
        
        # Stack to stereo
        audio = np.stack([audio_l, audio_r], axis=0)
        
        # Ensure good dynamic range
        audio = audio * 0.7  # Headroom
        
        return audio
    
    def get_stats(self) -> Dict[str, Any]:
        """DDP compatible simplified statistics with stereo info"""
        stats = self.file_manager.get_stats()
        
        # Add dataset-specific stats
        stats.update({
            'total_files': len(self.audio_paths),
            'target_length': self.target_length,
            'memory_efficient_loading': self.memory_efficient_loading,
            'intelligent_caching': self.intelligent_caching,
            'cache_size_mb': self.cache_size_bytes / (1024 * 1024),
            'cache_hit_rate': (
                sum(self.cache_usage.values()) / len(self.cache_usage)
                if self.cache_usage else 0
            ),
            'batch_size_validation': self.batch_size_validation,
            'ddp_compatible': True,
            'v100_optimized': True,
            'simplified': True,  # 음질 평가 제거됨
            'stereo_enhanced': True,  # STEREO ENHANCED
            'ensure_stereo': self.ensure_stereo,
            'stereo_processing_mode': self.stereo_processing_mode,
        })
        
        return stats


class StereoEnhancedCollator:
    """
    STEREO ENHANCED collator with guaranteed stereo format
    """
    
    def __init__(
        self, 
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        pad_to_multiple: int = 512,
        filter_corrupted: bool = True,
        ensure_stereo: bool = True  # STEREO ENHANCED
    ):
        """
        STEREO ENHANCED Collator
        """
        self.max_length = max_length
        self.min_length = min_length or 1000
        self.pad_to_multiple = pad_to_multiple
        self.filter_corrupted = filter_corrupted
        self.ensure_stereo = ensure_stereo
    
    def _is_valid_audio(self, audio: torch.Tensor) -> bool:
        """STEREO ENHANCED: 기본 오디오 검증"""
        try:
            if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
                return False
            
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                return False
            
            return True
        except Exception:
            return False
    
    def _safe_pad(self, audio: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        STEREO ENHANCED: 스테레오 보장 패딩
        """
        # Ensure stereo format first
        if self.ensure_stereo and audio.dim() >= 2:
            if audio.shape[0] != 2:
                if audio.shape[0] == 1:
                    audio = audio.repeat(2, 1)
                elif audio.shape[0] > 2:
                    audio = audio[:2]
        
        current_length = audio.shape[-1]
        
        if current_length == target_length:
            return audio
        elif current_length > target_length:
            return audio[..., :target_length]
        else:
            pad_length = target_length - current_length
            
            if pad_length <= current_length and current_length > 1:
                # Reflection padding
                return torch.nn.functional.pad(
                    audio, (0, pad_length), mode='reflect'
                )
            else:
                # Zero padding
                return torch.nn.functional.pad(
                    audio, (0, pad_length), mode='constant', value=0
                )
    
    def __call__(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """
        STEREO ENHANCED: 스테레오 보장 collate 함수
        """
        # Filter valid audio
        valid_batch = []
        
        for audio in batch:
            if self.filter_corrupted:
                if not self._is_valid_audio(audio):
                    continue
            
            # STEREO ENHANCED: Ensure stereo format
            if self.ensure_stereo:
                if audio.dim() == 1:  # (T,) -> (2, T)
                    audio = audio.unsqueeze(0).repeat(2, 1)
                elif audio.dim() == 2 and audio.shape[0] != 2:
                    if audio.shape[0] == 1:
                        audio = audio.repeat(2, 1)
                    elif audio.shape[0] > 2:
                        audio = audio[:2]
            
            # Length validation
            if audio.shape[-1] >= self.min_length:
                valid_batch.append(audio)
            else:
                # Pad to minimum length
                audio_padded = self._safe_pad(audio, self.min_length)
                valid_batch.append(audio_padded)
        
        if not valid_batch:
            # Create dummy stereo batch
            dummy_length = self.max_length or 44100
            dummy_audio = torch.randn(len(batch), 2, dummy_length) * 0.02
            
            # Add harmonic content with stereo difference
            t = torch.linspace(0, 1, dummy_length)
            harmonic = 0.05 * torch.sin(2 * torch.pi * 440 * t)
            dummy_audio[:, 0] += harmonic
            dummy_audio[:, 1] += harmonic * 0.8  # Phase difference
            
            return dummy_audio
        
        # Determine target length
        lengths = [audio.shape[-1] for audio in valid_batch]
        target_length = max(lengths)
        
        # Apply max length limit
        if self.max_length is not None:
            target_length = min(target_length, self.max_length)
        
        # Ensure multiple alignment
        if self.pad_to_multiple > 1:
            target_length = ((target_length - 1) // self.pad_to_multiple + 1) * self.pad_to_multiple
        
        # Apply padding
        padded_batch = []
        for audio in valid_batch:
            try:
                padded_audio = self._safe_pad(audio, target_length)
                padded_batch.append(padded_audio)
            except Exception:
                # Fallback padding
                if audio.shape[-1] > target_length:
                    padded_audio = audio[..., :target_length]
                else:
                    pad_length = target_length - audio.shape[-1]
                    padded_audio = torch.nn.functional.pad(audio, (0, pad_length))
                
                # Ensure stereo for fallback
                if self.ensure_stereo and padded_audio.dim() >= 2 and padded_audio.shape[0] != 2:
                    if padded_audio.shape[0] == 1:
                        padded_audio = padded_audio.repeat(2, 1)
                    elif padded_audio.shape[0] > 2:
                        padded_audio = padded_audio[:2]
                        
                padded_batch.append(padded_audio)
        
        # Stack into batch
        try:
            batched_audio = torch.stack(padded_batch, dim=0)
            
            # Final stereo verification
            if self.ensure_stereo and batched_audio.dim() >= 3 and batched_audio.shape[1] != 2:
                safe_print(f"⚠️ Batch shape mismatch: {batched_audio.shape}, fixing to stereo...")
                # Fix batch to stereo
                if batched_audio.shape[1] == 1:
                    batched_audio = batched_audio.repeat(1, 2, 1)
                elif batched_audio.shape[1] > 2:
                    batched_audio = batched_audio[:, :2, :]
                    
        except Exception:
            # Ultimate fallback: create stereo zeros
            batch_size = len(valid_batch)
            batched_audio = torch.zeros(batch_size, 2, target_length)
        
        return batched_audio


def create_s6_ssm_compression_datasets(
    data_root: str = "dataset-dcae/datasets/raw",
    train_ratio: float = 0.82,
    val_ratio: float = 0.18,
    sample_rate: int = 44100,
    max_duration: float = 10.0,
    target_length: Optional[int] = None,
    auto_delete_corrupted: bool = True,
    max_workers: Optional[int] = None,
    # DDP compatible fast mode options
    fast_mode: bool = False,
    skip_validation: bool = False,
    augmentation: bool = False,
    # DDP compatible additional options
    cache_audio: bool = False,
    skip_corrupted: bool = True,
    min_duration: float = 0.5,
    use_cached_list: bool = True,
    train_split: float = 0.82,
    # STEREO ENHANCED options
    ensure_stereo: bool = True,
    stereo_processing_mode: str = 'independent',
    **kwargs
) -> Tuple[StereoEnhancedS6SSMDataset, StereoEnhancedS6SSMDataset, StereoEnhancedS6SSMDataset]:
    """
    Create STEREO ENHANCED DDP compatible S6-SSM compression optimized datasets
    """
    if target_length is None:
        target_length = int(max_duration * sample_rate)
    
    mode_info = "DDP STEREO FAST MODE (libmpg123 런타임감지)" if fast_mode else "DDP STEREO PARALLEL (libmpg123 사전감지)" 
    safe_print(f"🚀 Creating DDP STEREO S6-SSM Compression datasets ({mode_info})")
    
    if fast_mode:
        safe_print(f"⚡ DDP STEREO Fast Mode: 음질 평가 완전 제거, 기본 검사만")
        safe_print(f"🔍 libmpg123 오류 감지: 기본 파일 크기 체크만")
        safe_print(f"🗑️  Auto-delete: Disabled")
    else:
        safe_print(f"🔍 Simple validation: libmpg123 part2_3_length 등 오류 파일 제거")
        safe_print(f"🗑️  Auto-delete corrupted: {auto_delete_corrupted}")
        safe_print(f"⚡ Max workers: {max_workers or 'Auto'}")
    
    safe_print(f"🔧 DDP Compatible: ✅ Enabled")
    safe_print(f"💾 V100 Optimized: ✅ Enabled")
    safe_print(f"🎯 Simplified Mode: ✅ 음질 평가 제거됨")
    safe_print(f"🛡️ libmpg123 Error Detection: ✅ part2_3_length 등 감지")
    safe_print(f"🎵 STEREO Enhanced: ✅ Enabled")
    safe_print(f"🔊 Stereo Processing: {stereo_processing_mode}")
    safe_print(f"📻 Ensure Stereo: {ensure_stereo}")
    
    # Create STEREO ENHANCED DDP dataset
    duplicate_args = ['augmentation', 'cache_audio', 'skip_corrupted', 'min_duration', 
                     'use_cached_list', 'train_split', 'ensure_stereo', 'stereo_processing_mode']
    for arg in duplicate_args:
        if arg in kwargs:
            kwargs.pop(arg)
    
    full_dataset = StereoEnhancedS6SSMDataset(
        data_root=data_root,
        sample_rate=sample_rate,
        max_duration=max_duration,
        min_duration=min_duration,
        augmentation=augmentation,
        cache_audio=cache_audio,
        skip_corrupted=skip_corrupted,
        target_length=target_length,
        auto_delete_corrupted=auto_delete_corrupted,
        memory_efficient_loading=True,
        intelligent_caching=True,
        max_workers=max_workers,
        fast_mode=fast_mode,
        skip_validation=skip_validation,
        ensure_stereo=ensure_stereo,
        stereo_processing_mode=stereo_processing_mode,
        **kwargs
    )
    
    # Dataset splitting
    total_size = len(full_dataset)
    if total_size == 0:
        raise ValueError("No valid audio files found for DDP STEREO S6-SSM compression")
    
    # Use train_split
    actual_train_ratio = train_split if train_split != 0.82 else train_ratio
    train_size = int(total_size * actual_train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # Random split
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subsets
    from torch.utils.data import Subset
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Enable augmentation for training
    full_dataset.augmentation = True
    
    # Print statistics
    stats = full_dataset.get_stats()
    
    mode_suffix = " (DDP STEREO FAST - libmpg123 런타임감지)" if fast_mode else " (DDP STEREO PARALLEL - libmpg123 사전감지)"
    safe_print(f"✅ DDP STEREO S6-SSM Compression dataset split{mode_suffix}:")
    safe_print(f"  📚 Train: {len(train_dataset)} files")
    safe_print(f"  📊 Val: {len(val_dataset)} files")
    safe_print(f"  🧪 Test: {len(test_dataset)} files")
    safe_print(f"  🎯 Target length: {target_length} samples ({target_length/sample_rate:.1f}s)")
    safe_print(f"  🔧 DDP Compatible: {stats.get('ddp_compatible', True)}")
    safe_print(f"  💾 V100 Optimized: {stats.get('v100_optimized', True)}")
    safe_print(f"  🎯 Simplified: {stats.get('simplified', True)} (음질 평가 제거됨)")
    safe_print(f"  🎵 STEREO Enhanced: {stats.get('stereo_enhanced', True)}")
    safe_print(f"  🔊 Ensure Stereo: {stats.get('ensure_stereo', True)}")
    safe_print(f"  📻 Stereo Mode: {stats.get('stereo_processing_mode', 'independent')}")
    
    if not fast_mode:
        safe_print(f"  🗑️  Corrupted processed: {stats['total_corrupted']}")
        safe_print(f"  ⚡ Parallel workers: {stats['max_workers']}")
        
        if stats.get('deletion_enabled', False):
            safe_print(f"  ✅ Files cleaned: {stats['total_deleted']}")
    else:
        safe_print(f"  ⚡ DDP STEREO Fast Mode: 기본 검사만으로 빠른 시작 (libmpg123 런타임 감지)")
    
    return train_dataset, val_dataset, test_dataset


# ==================== Backward Compatibility Aliases ====================

# Aliases for backward compatibility
DCAEDataset = StereoEnhancedS6SSMDataset
DCAECollator = StereoEnhancedCollator

# Alternative aliases
DCAE_Dataset = StereoEnhancedS6SSMDataset
DCAE_Collator = StereoEnhancedCollator

# Additional compatibility
DDPCompatibleS6SSMDataset = StereoEnhancedS6SSMDataset