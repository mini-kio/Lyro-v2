# lyro/dataset/dcae_dataset.py
"""
DDP Compatible S6-SSM Compression Optimized DCAE Dataset Implementation
FIXED: All DDP autograd hooks conflicts and progressive parameter changes removed
OPTIMIZED: V100 16GB x4 environment compatibility with static processing
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


class DDPCompatibleFileManager:
    """
    DDP Compatible file manager for S6-SSM compression optimization
    FIXED: Static processing without dynamic parameter changes during DDP training
    """
    
    def __init__(
        self, 
        enable_deletion: bool = True, 
        backup_corrupted: bool = False,
        compression_quality_filter: bool = True,
        min_dynamic_range_db: float = 20.0,  # FIXED: Static threshold
        max_compression_artifacts: float = 0.1,  # FIXED: Static threshold
        max_workers: Optional[int] = None
    ):
        self.enable_deletion = enable_deletion
        self.backup_corrupted = backup_corrupted
        self.compression_quality_filter = compression_quality_filter
        self.min_dynamic_range_db = min_dynamic_range_db
        self.max_compression_artifacts = max_compression_artifacts
        
        # Thread safety for DDP
        self.lock = threading.Lock()
        self._logged_files = set()
        
        # FIXED: Static results storage for DDP
        self.corrupted_files = set()
        self.low_quality_files = set()
        self.deletion_log = []
        self.quality_log = []
        
        # FIXED: Static parallel processing for DDP (V100 optimized)
        self.max_workers = max_workers or min(4, (os.cpu_count() or 1) + 1)  # V100 conservative
        
        # Create backup directory
        if self.backup_corrupted:
            self.backup_dir = Path("corrupted_files_backup")
            self.backup_dir.mkdir(exist_ok=True)
        
        # FIXED: Static quality analysis cache for DDP
        self.quality_cache = {}
        self.cache_file = Path("ddp_s6_ssm_quality_cache.json")
        self._load_quality_cache()
    
    def _load_quality_cache(self):
        """Load quality analysis cache - DDP compatible"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.quality_cache = json.load(f)
        except Exception:
            self.quality_cache = {}
    
    def _save_quality_cache(self):
        """Save quality analysis cache - DDP compatible"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.quality_cache, f)
        except Exception:
            pass
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get file hash for caching - DDP compatible"""
        try:
            stat = file_path.stat()
            hash_input = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception:
            return str(file_path)
    
    def analyze_audio_quality_batch(self, file_paths: List[Path]) -> Dict[Path, Dict[str, float]]:
        """
        DDP compatible parallel batch analysis of audio quality
        """
        results = {}
        
        def analyze_single_file(file_path):
            return file_path, self.analyze_audio_quality_single(file_path)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(analyze_single_file, fp): fp for fp in file_paths}
            
            for future in as_completed(future_to_file):
                try:
                    file_path, quality_metrics = future.result()
                    results[file_path] = quality_metrics
                except Exception:
                    file_path = future_to_file[future]
                    results[file_path] = self._default_quality_metrics()
        
        return results
    
    def analyze_audio_quality_single(self, file_path: Path) -> Dict[str, float]:
        """
        DDP compatible audio quality analysis (single file)
        """
        file_hash = self._get_file_hash(file_path)
        
        # Check cache first
        if file_hash in self.quality_cache:
            return self.quality_cache[file_hash]
        
        quality_metrics = self._default_quality_metrics()
        
        try:
            # FIXED: Static load segment for DDP analysis
            audio, sr = librosa.load(str(file_path), sr=44100, duration=2.0)  # V100 optimized
            
            if len(audio) == 0:
                return quality_metrics
            
            # FIXED: Static dynamic range analysis for DDP
            rms_values = librosa.feature.rms(y=audio, hop_length=512)[0]
            if len(rms_values) > 0:
                dynamic_range = 20 * np.log10(np.max(rms_values) / (np.mean(rms_values) + 1e-10))
                quality_metrics['dynamic_range_db'] = float(dynamic_range)
            
            # FIXED: Static compression artifacts detection for DDP
            diff = np.diff(audio)
            sudden_changes = np.sum(np.abs(diff) > 0.1) / len(diff)
            quality_metrics['compression_artifacts'] = float(sudden_changes)
            
            # FIXED: Static harmonic content analysis for DDP
            try:
                harmonic, percussive = librosa.effects.hpss(audio)
                harmonic_energy = np.mean(harmonic ** 2)
                total_energy = np.mean(audio ** 2)
                quality_metrics['harmonic_content'] = float(harmonic_energy / (total_energy + 1e-10))
            except Exception:
                quality_metrics['harmonic_content'] = 0.5
            
            # FIXED: Static frequency balance analysis for DDP
            try:
                stft = librosa.stft(audio)
                magnitude = np.abs(stft)
                
                freq_bins = magnitude.shape[0]
                low_energy = np.mean(magnitude[:freq_bins//3])
                mid_energy = np.mean(magnitude[freq_bins//3:2*freq_bins//3])
                high_energy = np.mean(magnitude[2*freq_bins//3:])
                
                total_energy = low_energy + mid_energy + high_energy
                if total_energy > 0:
                    balance_score = 1.0 - np.std([low_energy, mid_energy, high_energy]) / total_energy
                    quality_metrics['frequency_balance'] = float(balance_score)
            except Exception:
                quality_metrics['frequency_balance'] = 0.5
            
            # FIXED: Static overall suitability for compression for DDP
            quality_metrics['suitable_for_compression'] = (
                quality_metrics['dynamic_range_db'] >= self.min_dynamic_range_db and
                quality_metrics['compression_artifacts'] <= self.max_compression_artifacts and
                quality_metrics['harmonic_content'] >= 0.1 and
                quality_metrics['frequency_balance'] >= 0.3
            )
            
        except Exception:
            quality_metrics = self._default_quality_metrics()
        
        # Cache the results
        self.quality_cache[file_hash] = quality_metrics
        return quality_metrics
    
    def _default_quality_metrics(self) -> Dict[str, float]:
        """Default quality metrics for failed analysis - DDP compatible"""
        return {
            'dynamic_range_db': 0.0,
            'compression_artifacts': 1.0,
            'harmonic_content': 0.0,
            'frequency_balance': 0.0,
            'suitable_for_compression': False
        }
    
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
    
    def mark_low_quality(self, file_path: Path, quality_metrics: Dict[str, float]):
        """Mark file as low quality - DDP compatible thread-safe"""
        with self.lock:
            file_key = str(file_path)
            if file_key in self._logged_files:
                return
            self._logged_files.add(file_key)
            
            self.low_quality_files.add(file_path)
            
            log_entry = {
                'file': safe_str(file_path),
                'quality_metrics': quality_metrics,
                'timestamp': time.time()
            }
            self.quality_log.append(log_entry)
    
    def is_corrupted(self, file_path: Path) -> bool:
        """Check if file is marked as corrupted - DDP compatible"""
        return file_path in self.corrupted_files
    
    def is_low_quality(self, file_path: Path) -> bool:
        """Check if file is marked as low quality - DDP compatible"""
        return file_path in self.low_quality_files
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics - DDP compatible"""
        return {
            'total_corrupted': len(self.corrupted_files),
            'total_low_quality': len(self.low_quality_files),
            'total_deleted': len(self.deletion_log),
            'deletion_enabled': self.enable_deletion,
            'backup_enabled': self.backup_corrupted,
            'quality_filter_enabled': self.compression_quality_filter,
            'min_dynamic_range_db': self.min_dynamic_range_db,
            'max_compression_artifacts': self.max_compression_artifacts,
            'max_workers': self.max_workers,
            'ddp_compatible': True,
            'v100_optimized': True
        }
    
    def save_logs(self, log_dir: Path):
        """Save deletion and quality logs - DDP compatible"""
        log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save deletion log
            deletion_log_path = log_dir / "ddp_s6_ssm_deletion_log.json"
            with open(deletion_log_path, 'w', encoding='utf-8', errors='replace') as f:
                json.dump(self.deletion_log, f, indent=2, ensure_ascii=False)
            
            # Save quality log
            quality_log_path = log_dir / "ddp_s6_ssm_quality_log.json"
            with open(quality_log_path, 'w', encoding='utf-8', errors='replace') as f:
                json.dump(self.quality_log, f, indent=2, ensure_ascii=False)
            
            # Save quality cache
            self._save_quality_cache()
        except Exception:
            pass


class DDPCompatibleAudioValidator:
    """
    DDP compatible audio validator for S6-SSM compression optimization
    """
    
    def __init__(self, file_manager: DDPCompatibleFileManager):
        self.file_manager = file_manager
        self.validation_cache = {}
        self.cache_lock = threading.Lock()
    
    def validate_audio_batch(self, audio_paths: List[Path]) -> Dict[Path, bool]:
        """
        DDP compatible parallel batch validation
        """
        results = {}
        
        def validate_single_file(file_path):
            return file_path, self.validate_audio_deeply(file_path)
        
        with ThreadPoolExecutor(max_workers=self.file_manager.max_workers) as executor:
            future_to_file = {executor.submit(validate_single_file, fp): fp for fp in audio_paths}
            
            for future in as_completed(future_to_file):
                try:
                    file_path, (is_valid, error_msg, quality_metrics) = future.result()
                    
                    if not is_valid:
                        if quality_metrics and "quality" in error_msg.lower():
                            self.file_manager.mark_low_quality(file_path, quality_metrics)
                        else:
                            self.file_manager.mark_corrupted(file_path, error_msg)
                    
                    results[file_path] = is_valid
                    
                except Exception:
                    file_path = future_to_file[future]
                    self.file_manager.mark_corrupted(file_path, "Validation exception")
                    results[file_path] = False
        
        return results
    
    def validate_audio_deeply(self, audio_path: Path) -> Tuple[bool, str, Dict[str, float]]:
        """
        DDP compatible deep validation with quality analysis
        """
        try:
            # FIXED: Static basic file checks for DDP
            if not audio_path.exists():
                return False, "File does not exist", {}
            
            file_size = audio_path.stat().st_size
            if file_size < 1000:  # Fixed 1KB minimum
                return False, "File too small", {}
            
            # FIXED: Static audio format validation for DDP
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    duration = librosa.get_duration(path=str(audio_path))
                    
                    if duration <= 0 or duration > 300:  # Fixed 5 minutes max (V100 optimized)
                        return False, f"Invalid duration: {duration}", {}
            except Exception as e:
                return False, f"Duration check failed: {str(e)}", {}
            
            # FIXED: Static quality analysis for DDP
            quality_metrics = {}
            if self.file_manager.compression_quality_filter:
                quality_metrics = self.file_manager.analyze_audio_quality_single(audio_path)
                
                if not quality_metrics.get('suitable_for_compression', False):
                    return False, "Low quality for compression", quality_metrics
            
            # FIXED: Static actual loading test for DDP
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    audio, sr = librosa.load(str(audio_path), sr=None, duration=0.1)
                    
                    if len(audio) == 0:
                        return False, "Empty audio data", quality_metrics
                    
                    if np.isnan(audio).any() or np.isinf(audio).any():
                        return False, "Contains NaN or Inf values", quality_metrics
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in [
                    'dequantization failed', 'part2_3_length', 'layer3',
                    'one-frame stream', 'cannot read next header'
                ]):
                    return False, f"Audio corruption detected: {str(e)}", quality_metrics
                else:
                    return False, f"Audio loading failed: {str(e)}", quality_metrics
            
            return True, "Valid", quality_metrics
            
        except Exception as e:
            return False, f"Validation error: {str(e)}", {}
    
    def validate_and_filter(self, audio_path: Path) -> bool:
        """
        DDP compatible validate and filter
        """
        with self.cache_lock:
            cache_key = safe_str(audio_path)
            if cache_key in self.validation_cache:
                return self.validation_cache[cache_key]
        
        # Check if already marked
        if (self.file_manager.is_corrupted(audio_path) or 
            self.file_manager.is_low_quality(audio_path)):
            with self.cache_lock:
                self.validation_cache[cache_key] = False
            return False
        
        # Perform validation
        is_valid, error_msg, quality_metrics = self.validate_audio_deeply(audio_path)
        
        if not is_valid:
            if quality_metrics and "quality" in error_msg.lower():
                self.file_manager.mark_low_quality(audio_path, quality_metrics)
            else:
                self.file_manager.mark_corrupted(audio_path, error_msg)
            
            with self.cache_lock:
                self.validation_cache[cache_key] = False
            return False
        
        with self.cache_lock:
            self.validation_cache[cache_key] = True
        return True


class DDPCompatibleS6SSMDataset(Dataset):
    """
    DDP Compatible S6-SSM Compression Optimized DCAE Dataset
    FIXED: Static processing for DDP compatibility and V100 optimization
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
        max_file_size_mb: float = 30.0,  # V100 optimized: reduced from 50MB
        target_length: Optional[int] = None,
        # DDP compatible S6-SSM compression parameters
        compression_quality_filter: bool = True,
        min_dynamic_range_db: float = 20.0,  # FIXED: Static threshold
        max_compression_artifacts: float = 0.1,  # FIXED: Static threshold
        auto_delete_corrupted: bool = True,
        backup_corrupted: bool = False,
        max_retries: int = 3,  # V100 optimized: reduced from 5
        # V100 memory optimization
        memory_efficient_loading: bool = True,
        intelligent_caching: bool = True,
        cache_size_limit_mb: float = 200.0,  # V100 optimized: reduced from 500MB
        # DDP compatible quality preservation
        preserve_peak_db: float = -1.0,  # FIXED: Static peak level
        normalize_for_compression: bool = True,
        # V100 optimized parallel processing
        max_workers: Optional[int] = None,
        batch_size_validation: int = 50,  # V100 optimized: reduced from 100
        # DDP compatible fast mode
        fast_mode: bool = False,
        skip_validation: bool = False,
        use_cached_list: bool = True
    ):
        """
        DDP Compatible S6-SSM Compression Optimized Dataset
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
        
        # FIXED: Static fast mode settings for DDP
        self.fast_mode = fast_mode
        self.skip_validation = skip_validation
        self.use_cached_list = use_cached_list
        
        # FIXED: Static fast mode auto settings for DDP
        if self.fast_mode:
            self.skip_validation = True
            self.compression_quality_filter = False
            self.auto_delete_corrupted = False
            safe_print("🚀 DDP Fast Mode: Skipping validation for quick startup!")
        
        # FIXED: Static S6-SSM compression parameters for DDP
        self.compression_quality_filter = compression_quality_filter and not self.fast_mode
        self.memory_efficient_loading = memory_efficient_loading
        self.intelligent_caching = intelligent_caching
        self.preserve_peak_db = preserve_peak_db
        self.normalize_for_compression = normalize_for_compression
        
        # FIXED: Static enhanced file manager for DDP
        self.file_manager = DDPCompatibleFileManager(
            enable_deletion=auto_delete_corrupted and not self.fast_mode,
            backup_corrupted=backup_corrupted,
            compression_quality_filter=self.compression_quality_filter,
            min_dynamic_range_db=min_dynamic_range_db,
            max_compression_artifacts=max_compression_artifacts,
            max_workers=max_workers
        )
        
        # FIXED: Static enhanced validator for DDP
        self.validator = DDPCompatibleAudioValidator(self.file_manager)
        
        # FIXED: Static collect and validate audio files for DDP
        if self.fast_mode or self.skip_validation:
            self.audio_paths = self._collect_audio_files_fast()
        else:
            self.audio_paths = self._collect_and_validate_audio_files_parallel()
        
        safe_print(f"✅ DDP S6-SSM Dataset: {len(self.audio_paths)} files ready")
        
        # FIXED: Static enhanced statistics for DDP
        if not self.fast_mode:
            stats = self.file_manager.get_stats()
            if stats['total_corrupted'] > 0 or stats['total_low_quality'] > 0:
                safe_print(f"🗑️  Processed {stats['total_corrupted']} corrupted files")
                safe_print(f"⚠️  Filtered {stats['total_low_quality']} low-quality files")
                if stats['deletion_enabled']:
                    safe_print(f"✅ Cleaned {stats['total_deleted']} files")
        
        # FIXED: Static intelligent caching setup for DDP
        self.audio_cache = {}
        self.cache_usage = defaultdict(int)
        self.cache_size_bytes = 0
        self.cache_size_limit_bytes = int(cache_size_limit_mb * 1024 * 1024)
        
        if (cache_audio and len(self.audio_paths) < 500 and  # V100 optimized
            self.intelligent_caching and not self.fast_mode):
            safe_print("Setting up V100 optimized intelligent caching...")
            self._setup_intelligent_cache()
        
        # FIXED: Static save logs for DDP
        if not self.fast_mode:
            stats = self.file_manager.get_stats() 
            if stats['total_deleted'] > 0 or stats['total_low_quality'] > 0:
                log_dir = self.data_root / "ddp_s6_ssm_compression_logs"
                self.file_manager.save_logs(log_dir)
                safe_print(f"📝 DDP S6-SSM logs saved to: {safe_str(log_dir)}")
    
    def _collect_audio_files_fast(self) -> List[Path]:
        """DDP compatible fast mode file collection"""
        safe_print("🚀 DDP Fast Mode: Quick file collection...")
        
        # FIXED: Static cached file list check for DDP
        cache_file = self.data_root / "ddp_s6_ssm_file_cache.json"
        if self.use_cached_list and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_paths = json.load(f)
                
                audio_paths = []
                for path_str in cached_paths:
                    path = Path(path_str)
                    if path.exists():
                        audio_paths.append(path)
                
                if len(audio_paths) > 50:  # V100 optimized threshold
                    safe_print(f"📋 Using cached file list: {len(audio_paths)} files")
                    return audio_paths
            except Exception:
                pass
        
        # FIXED: Static quick file collection for DDP
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        audio_paths = []
        
        subfolders = ['environmental', 'music', 'speech']
        
        for subfolder in subfolders:
            subfolder_path = self.data_root / subfolder
            if not subfolder_path.exists():
                continue
            
            # FIXED: Static quick file search for DDP
            for ext in audio_extensions:
                try:
                    folder_files = list(subfolder_path.glob(f'*{ext}'))
                    folder_files.extend(subfolder_path.glob(f'*{ext.upper()}'))
                    
                    # FIXED: Static basic file size check for DDP
                    for file_path in folder_files:
                        try:
                            if file_path.stat().st_size > 1000:  # Fixed 1KB minimum
                                audio_paths.append(file_path)
                        except Exception:
                            continue
                except Exception:
                    continue
            
            safe_print(f"📁 {subfolder}: {len([p for p in audio_paths if subfolder in str(p)])} files")
        
        # FIXED: Static save file list cache for DDP
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
        
        safe_print(f"🔍 DDP S6-SSM: Collecting files in: {safe_str(self.data_root)}")
        
        subfolders = ['environmental', 'music', 'speech']
        total_found = 0
        
        for subfolder in subfolders:
            subfolder_path = self.data_root / subfolder
            
            if not subfolder_path.exists():
                safe_print(f"⚠️  Subfolder not found: {subfolder}")
                continue
            
            # FIXED: Static find all audio files for DDP
            folder_files = []
            for ext in audio_extensions:
                try:
                    folder_files.extend(subfolder_path.glob(f'*{ext}'))
                    folder_files.extend(subfolder_path.glob(f'*{ext.upper()}'))
                except Exception:
                    continue
            
            total_found += len(folder_files)
            safe_print(f"📁 {subfolder}: {len(folder_files)} files found")
            
            # FIXED: Static parallel validation for DDP
            if folder_files:
                safe_print(f"🔍 DDP S6-SSM validation: {len(folder_files)} files in {subfolder}...")
                
                valid_files = self._validate_files_in_batches(folder_files, subfolder)
                audio_paths.extend(valid_files)
        
        safe_print(f"📊 DDP S6-SSM: Found {total_found}, Valid: {len(audio_paths)}")
        return audio_paths
    
    def _validate_files_in_batches(self, file_paths: List[Path], subfolder: str) -> List[Path]:
        """DDP compatible file validation in parallel batches"""
        valid_files = []
        
        # FIXED: Static process files in batches for DDP
        batches = [file_paths[i:i + self.batch_size_validation] 
                  for i in range(0, len(file_paths), self.batch_size_validation)]
        
        with tqdm(total=len(file_paths), desc=f"Validating {subfolder}", leave=False) as pbar:
            for batch in batches:
                try:
                    # FIXED: Static filter invalid Unicode files for DDP
                    valid_batch = []
                    for file_path in batch:
                        safe_path_str = safe_str(file_path)
                        if "<invalid_unicode>" not in safe_path_str:
                            valid_batch.append(file_path)
                    
                    if not valid_batch:
                        pbar.update(len(batch))
                        continue
                    
                    # FIXED: Static parallel validation for DDP
                    validation_results = self.validator.validate_audio_batch(valid_batch)
                    
                    # FIXED: Static collect valid files for DDP
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
        
        # FIXED: Static sample first 50 files for V100 DDP
        for path in self.audio_paths[:50]:
            try:
                # FIXED: Static quick analysis for caching decision for DDP
                file_size = path.stat().st_size
                
                # FIXED: Static prefer smaller files for V100 caching
                if file_size < self.max_file_size_bytes // 3:  # V100 optimized
                    priority = 1.0 / (file_size + 1)
                    cache_candidates.append((path, priority))
            except Exception:
                continue
        
        # FIXED: Static sort by priority and cache for DDP
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
                    
                    # FIXED: Static estimate cache size for DDP
                    audio_size = audio.nbytes if hasattr(audio, 'nbytes') else len(audio) * 8
                    self.cache_size_bytes += audio_size
                    cached_count += 1
            except Exception:
                continue
        
        safe_print(f"🧠 V100 intelligent cache: {cached_count} files cached")
    
    def _load_audio_safe(self, audio_path: Path) -> Optional[np.ndarray]:
        """
        DDP compatible memory-efficient audio loading (V100 optimized)
        """
        cache_key = safe_str(audio_path)
        
        # FIXED: Static check intelligent cache for DDP
        if self.intelligent_caching and cache_key in self.audio_cache:
            self.cache_usage[cache_key] += 1
            return self.audio_cache[cache_key]
        
        # FIXED: Static check file manager status for DDP
        if (self.file_manager.is_corrupted(audio_path) or 
            self.file_manager.is_low_quality(audio_path)):
            return None
        
        audio = None
        
        # FIXED: Static memory-efficient loading for V100 DDP
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
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in [
                    'dequantization failed', 'part2_3_length', 'layer3'
                ]):
                    self.file_manager.mark_corrupted(audio_path, str(e))
                    return None
        
        # FIXED: Static fallback loading for DDP
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
        
        # FIXED: Static audio processing for DDP compression optimization
        try:
            # FIXED: Static ensure stereo format for DDP
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            elif audio.shape[0] == 1:
                audio = np.tile(audio, (2, 1))
            elif audio.shape[0] > 2:
                audio = audio[:2]
            
            # FIXED: Static length adjustment for DDP
            current_length = audio.shape[1]
            target_length = self.target_length
            
            if current_length > target_length:
                # FIXED: Static smart cropping for DDP
                start_options = [0, current_length - target_length]
                if current_length > target_length * 2:
                    start_options.append((current_length - target_length) // 2)
                
                start = random.choice(start_options)
                audio = audio[:, start:start + target_length]
            elif current_length < target_length:
                # FIXED: Static smart padding/repetition for DDP
                if current_length < target_length // 4:
                    repeat_count = (target_length // current_length) + 1
                    audio = np.tile(audio, (1, repeat_count))
                    audio = audio[:, :target_length]
                else:
                    pad_length = target_length - current_length
                    audio = np.pad(audio, ((0, 0), (0, pad_length)), mode='constant')
            
            # FIXED: Static quality checks for DDP
            if np.isnan(audio).any() or np.isinf(audio).any():
                self.file_manager.mark_corrupted(audio_path, "Contains NaN or Inf")
                return None
            
            # FIXED: Static compression-optimized normalization for DDP
            if self.normalize_for_compression:
                max_val = np.abs(audio).max()
                if max_val > 0:
                    target_peak = 10 ** (self.preserve_peak_db / 20)
                    audio = audio / max_val * target_peak
                else:
                    self.file_manager.mark_corrupted(audio_path, "Silent audio")
                    return None
            
            # FIXED: Static final quality check for DDP
            if self.compression_quality_filter:
                rms_values = np.sqrt(np.mean(audio ** 2, axis=1))
                peak_values = np.max(np.abs(audio), axis=1)
                
                for channel in range(audio.shape[0]):
                    if peak_values[channel] > 0:
                        dynamic_range = 20 * np.log10(peak_values[channel] / (rms_values[channel] + 1e-10))
                        if dynamic_range < self.file_manager.min_dynamic_range_db:
                            quality_metrics = {'dynamic_range_db': dynamic_range}
                            self.file_manager.mark_low_quality(audio_path, quality_metrics)
                            return None
            
            return audio
            
        except Exception as e:
            self.file_manager.mark_corrupted(audio_path, f"Post-processing: {str(e)}")
            return None
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        DDP compatible get item with S6-SSM compression optimization
        """
        for retry in range(self.max_retries):
            try:
                if idx >= len(self.audio_paths):
                    idx = random.randint(0, len(self.audio_paths) - 1)
                
                audio_path = self.audio_paths[idx]
                
                # FIXED: Static load audio for DDP
                audio = self._load_audio_safe(audio_path)
                
                if audio is None:
                    # FIXED: Static find another valid file for DDP
                    available_indices = [
                        i for i, p in enumerate(self.audio_paths) 
                        if not (self.file_manager.is_corrupted(p) or self.file_manager.is_low_quality(p))
                    ]
                    
                    if available_indices:
                        idx = random.choice(available_indices)
                        continue
                    else:
                        # FIXED: Static generate dummy audio for DDP
                        audio = self._generate_compression_friendly_dummy()
                        break
                
                # FIXED: Static final length verification for DDP
                if audio.shape[1] != self.target_length:
                    if audio.shape[1] > self.target_length:
                        audio = audio[:, :self.target_length]
                    else:
                        pad_length = self.target_length - audio.shape[1]
                        audio = np.pad(audio, ((0, 0), (0, pad_length)), mode='constant')
                
                # FIXED: Static convert to tensor for DDP
                audio_tensor = torch.from_numpy(audio).float()
                
                return audio_tensor
                
            except Exception:
                if retry < self.max_retries - 1:
                    idx = random.randint(0, len(self.audio_paths) - 1)
                    continue
                else:
                    # FIXED: Static generate compression-friendly dummy for DDP
                    audio = self._generate_compression_friendly_dummy()
                    return torch.from_numpy(audio).float()
    
    def _generate_compression_friendly_dummy(self) -> np.ndarray:
        """DDP compatible compression-friendly dummy audio generation"""
        # FIXED: Static generate pink noise for DDP
        audio = np.random.randn(2, self.target_length) * 0.02
        
        # FIXED: Static add harmonic content for DDP
        t = np.linspace(0, self.target_length / self.sample_rate, self.target_length)
        harmonic = 0.05 * np.sin(2 * np.pi * 440 * t)  # Fixed A4 note
        audio[0] += harmonic
        audio[1] += harmonic * 0.8  # Fixed stereo difference
        
        # FIXED: Static ensure good dynamic range for DDP
        audio = audio * 0.7  # Fixed headroom
        
        return audio
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """DDP compatible comprehensive compression statistics"""
        stats = self.file_manager.get_stats()
        
        # FIXED: Static add dataset-specific stats for DDP
        stats.update({
            'total_files': len(self.audio_paths),
            'target_length': self.target_length,
            'compression_quality_filter': self.compression_quality_filter,
            'memory_efficient_loading': self.memory_efficient_loading,
            'intelligent_caching': self.intelligent_caching,
            'cache_size_mb': self.cache_size_bytes / (1024 * 1024),
            'cache_hit_rate': (
                sum(self.cache_usage.values()) / len(self.cache_usage)
                if self.cache_usage else 0
            ),
            'batch_size_validation': self.batch_size_validation,
            'ddp_compatible': True,
            'v100_optimized': True
        })
        
        return stats


class DDPCompatibleCollator:
    """
    DDP Compatible collator for S6-SSM compression optimization
    FIXED: Static processing for DDP compatibility
    """
    
    def __init__(
        self, 
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        pad_to_multiple: int = 512,
        filter_corrupted: bool = True,
        # DDP compatible S6-SSM compression specific
        compression_aware_batching: bool = True,
        quality_threshold: float = 0.1,  # FIXED: Static quality threshold
        dynamic_length_adjustment: bool = True
    ):
        """
        DDP Compatible S6-SSM Compression Collator
        """
        self.max_length = max_length
        self.min_length = min_length or 1000  # Fixed minimum
        self.pad_to_multiple = pad_to_multiple
        self.filter_corrupted = filter_corrupted
        self.compression_aware_batching = compression_aware_batching
        self.quality_threshold = quality_threshold
        self.dynamic_length_adjustment = dynamic_length_adjustment
    
    def _is_compression_friendly(self, audio: torch.Tensor) -> bool:
        """DDP compatible compression-friendly check"""
        try:
            if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
                return False
            
            if torch.isnan(audio).any() or torch.isinf(audio).any():
                return False
            
            # FIXED: Static check dynamic range for DDP
            rms = torch.sqrt(torch.mean(audio ** 2))
            peak = torch.max(torch.abs(audio))
            
            if peak > 0:
                dynamic_range_db = 20 * torch.log10(peak / (rms + 1e-10))
                if dynamic_range_db < 15.0:  # Fixed minimum dynamic range
                    return False
            
            # FIXED: Static check for artifacts for DDP
            diff = torch.diff(audio.mean(0))
            sudden_changes = torch.sum(torch.abs(diff) > 0.1) / len(diff)
            if sudden_changes > 0.05:  # Fixed threshold
                return False
            
            return True
        except Exception:
            return False
    
    def _safe_pad_compression_aware(self, audio: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        DDP compatible compression-aware padding
        """
        current_length = audio.shape[1]
        
        if current_length == target_length:
            return audio
        elif current_length > target_length:
            # FIXED: Static smart cropping for DDP
            if self.dynamic_length_adjustment:
                # FIXED: Static find quietest segment for DDP
                segment_size = current_length - target_length
                min_energy = float('inf')
                best_start = 0
                
                # FIXED: Static check few positions for DDP
                for start in range(0, segment_size + 1, max(1, segment_size // 10)):
                    segment = audio[:, start:start + target_length]
                    energy = torch.mean(segment ** 2)
                    if energy < min_energy:
                        min_energy = energy
                        best_start = start
                
                return audio[:, best_start:best_start + target_length]
            else:
                return audio[:, :target_length]
        else:
            # FIXED: Static smart padding for DDP
            pad_length = target_length - current_length
            
            if pad_length <= current_length and current_length > 1:
                # FIXED: Static reflection padding for DDP
                return torch.nn.functional.pad(
                    audio, (0, pad_length), mode='reflect'
                )
            else:
                # FIXED: Static zero padding with fade for DDP
                padded = torch.nn.functional.pad(
                    audio, (0, pad_length), mode='constant', value=0
                )
                
                # FIXED: Static apply fade for DDP
                if current_length > 100:
                    fade_length = min(100, current_length // 10)
                    fade = torch.linspace(1, 0, fade_length)
                    padded[:, current_length - fade_length:current_length] *= fade
                
                return padded
    
    def __call__(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """
        DDP compatible compression-aware collate function
        """
        # FIXED: Static filter for compression quality for DDP
        valid_batch = []
        
        for audio in batch:
            if self.filter_corrupted:
                if not self._is_compression_friendly(audio):
                    continue
            
            # FIXED: Static length validation for DDP
            if audio.shape[1] >= self.min_length:
                valid_batch.append(audio)
            else:
                # FIXED: Static pad to minimum length for DDP
                audio_padded = self._safe_pad_compression_aware(audio, self.min_length)
                valid_batch.append(audio_padded)
        
        if not valid_batch:
            # FIXED: Static create compression-friendly dummy batch for DDP
            dummy_length = self.max_length or 44100  # Fixed default
            dummy_audio = torch.randn(len(batch), 2, dummy_length) * 0.02
            
            # FIXED: Static add harmonic content for DDP
            t = torch.linspace(0, 1, dummy_length)
            harmonic = 0.05 * torch.sin(2 * torch.pi * 440 * t)  # Fixed frequency
            dummy_audio[:, 0] += harmonic
            dummy_audio[:, 1] += harmonic * 0.8  # Fixed stereo
            
            return dummy_audio
        
        # FIXED: Static compression-aware length determination for DDP
        lengths = [audio.shape[1] for audio in valid_batch]
        
        if self.compression_aware_batching:
            # FIXED: Static choose compression-friendly length for DDP
            target_candidates = []
            for length in lengths:
                # FIXED: Static round to compression-friendly sizes for DDP
                friendly_length = ((length - 1) // 1024 + 1) * 1024  # Fixed 1024 blocks
                target_candidates.append(friendly_length)
            
            # FIXED: Static use median for DDP
            target_length = sorted(target_candidates)[len(target_candidates) // 2]
        else:
            target_length = max(lengths)
        
        # FIXED: Static apply max length limit for DDP
        if self.max_length is not None:
            target_length = min(target_length, self.max_length)
        
        # FIXED: Static ensure multiple alignment for DDP
        if self.pad_to_multiple > 1:
            target_length = ((target_length - 1) // self.pad_to_multiple + 1) * self.pad_to_multiple
        
        # FIXED: Static apply compression-aware padding for DDP
        padded_batch = []
        for audio in valid_batch:
            try:
                padded_audio = self._safe_pad_compression_aware(audio, target_length)
                padded_batch.append(padded_audio)
            except Exception:
                # FIXED: Static fallback padding for DDP
                if audio.shape[1] > target_length:
                    padded_audio = audio[:, :target_length]
                else:
                    pad_length = target_length - audio.shape[1]
                    padded_audio = torch.nn.functional.pad(audio, (0, pad_length))
                padded_batch.append(padded_audio)
        
        # FIXED: Static stack into batch for DDP
        try:
            batched_audio = torch.stack(padded_batch, dim=0)
        except Exception:
            batched_audio = torch.zeros(len(valid_batch), 2, target_length)
        
        return batched_audio


def create_s6_ssm_compression_datasets(
    data_root: str = "dataset-dcae/datasets/raw",
    train_ratio: float = 0.82,
    val_ratio: float = 0.18,
    sample_rate: int = 44100,
    max_duration: float = 10.0,
    target_length: Optional[int] = None,
    compression_quality_filter: bool = True,
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
    **kwargs
) -> Tuple[DDPCompatibleS6SSMDataset, DDPCompatibleS6SSMDataset, DDPCompatibleS6SSMDataset]:
    """
    Create DDP compatible S6-SSM compression optimized datasets
    """
    if target_length is None:
        target_length = int(max_duration * sample_rate)
    
    mode_info = "DDP FAST MODE" if fast_mode else "DDP PARALLEL" 
    safe_print(f"🚀 Creating DDP S6-SSM Compression datasets ({mode_info})")
    
    if fast_mode:
        safe_print(f"⚡ DDP Fast Mode: Skip validation, quick startup")
        safe_print(f"🎯 Compression quality filter: Disabled")
        safe_print(f"🗑️  Auto-delete: Disabled")
    else:
        safe_print(f"🎯 Compression quality filter: {compression_quality_filter}")
        safe_print(f"🗑️  Auto-delete corrupted: {auto_delete_corrupted}")
        safe_print(f"⚡ Max workers: {max_workers or 'Auto'}")
    
    safe_print(f"🔧 DDP Compatible: ✅ Enabled")
    safe_print(f"💾 V100 Optimized: ✅ Enabled")
    
    # FIXED: Static create DDP dataset for compatibility
    duplicate_args = ['augmentation', 'cache_audio', 'skip_corrupted', 'min_duration', 
                     'use_cached_list', 'train_split']
    for arg in duplicate_args:
        if arg in kwargs:
            kwargs.pop(arg)
    
    full_dataset = DDPCompatibleS6SSMDataset(
        data_root=data_root,
        sample_rate=sample_rate,
        max_duration=max_duration,
        min_duration=min_duration,
        augmentation=augmentation,
        cache_audio=cache_audio,
        skip_corrupted=skip_corrupted,
        target_length=target_length,
        compression_quality_filter=compression_quality_filter,
        auto_delete_corrupted=auto_delete_corrupted,
        memory_efficient_loading=True,
        intelligent_caching=True,
        max_workers=max_workers,
        fast_mode=fast_mode,
        skip_validation=skip_validation,
        **kwargs
    )
    
    # FIXED: Static dataset splitting for DDP
    total_size = len(full_dataset)
    if total_size == 0:
        raise ValueError("No valid audio files found for DDP S6-SSM compression")
    
    # FIXED: Static use train_split for DDP
    actual_train_ratio = train_split if train_split != 0.82 else train_ratio
    train_size = int(total_size * actual_train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # FIXED: Static random split for DDP
    indices = list(range(total_size))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # FIXED: Static create subsets for DDP
    from torch.utils.data import Subset
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # FIXED: Static enable augmentation for training for DDP
    full_dataset.augmentation = True
    
    # FIXED: Static print statistics for DDP
    compression_stats = full_dataset.get_compression_stats()
    
    mode_suffix = " (DDP FAST)" if fast_mode else " (DDP PARALLEL)"
    safe_print(f"✅ DDP S6-SSM Compression dataset split{mode_suffix}:")
    safe_print(f"  📚 Train: {len(train_dataset)} files")
    safe_print(f"  📊 Val: {len(val_dataset)} files")
    safe_print(f"  🧪 Test: {len(test_dataset)} files")
    safe_print(f"  🎯 Target length: {target_length} samples ({target_length/sample_rate:.1f}s)")
    safe_print(f"  🔧 DDP Compatible: {compression_stats.get('ddp_compatible', True)}")
    safe_print(f"  💾 V100 Optimized: {compression_stats.get('v100_optimized', True)}")
    
    if not fast_mode:
        safe_print(f"  🗑️  Corrupted processed: {compression_stats['total_corrupted']}")
        safe_print(f"  ⚠️  Low quality filtered: {compression_stats['total_low_quality']}")
        safe_print(f"  ⚡ Parallel workers: {compression_stats['max_workers']}")
        
        if compression_stats.get('deletion_enabled', False):
            safe_print(f"  ✅ Files cleaned: {compression_stats['total_deleted']}")
    else:
        safe_print(f"  ⚡ DDP Fast Mode: Validation skipped for quick startup")
    
    return train_dataset, val_dataset, test_dataset


# ==================== Backward Compatibility Aliases ====================

# FIXED: Static aliases for DDP compatibility
DCAEDataset = DDPCompatibleS6SSMDataset
DCAECollator = DDPCompatibleCollator

# FIXED: Static alternative aliases for DDP compatibility
DCAE_Dataset = DDPCompatibleS6SSMDataset
DCAE_Collator = DDPCompatibleCollator