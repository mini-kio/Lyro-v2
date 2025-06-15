# lyro/dataset/dataset.py
"""Dataset utilities with optional TTS support."""

import os
import json
import torch
import numpy as np
import librosa
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import random
from pathlib import Path
import sys

# LYRO 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the enhanced augmentation from training_utils
try:
    from dcae.training_utils import MixScaleAugmentation
    AUGMENTATION_AVAILABLE = True
except ImportError:
    print("Warning: MixScaleAugmentation not available. Using basic augmentation.")
    AUGMENTATION_AVAILABLE = False


class LyroDataset(Dataset):
    """
    Enhanced LYRO 메인 데이터셋 클래스 with TTS support and advanced augmentation
    
    메타데이터를 읽어서 오디오, 가사, TTS, 스타일 정보를 로드하고
    학습에 필요한 형태로 전처리하며, 고급 데이터 증강을 적용합니다.
    """
    
    def __init__(
        self,
        metadata_path: str,
        dataset_root: str = "dataset/",
        sample_rate: int = 44100,
        max_duration: float = 300.0,  # 5분
        task_ratios: Dict[str, float] = None,
        augmentation: bool = True,
        use_processed: bool = True,
        use_tts: bool = True,  # TTS 사용 여부
        tts_ratio: float = 0.3,  # TTS 데이터 사용 비율
        # T-1: Enhanced Augmentation Parameters
        augmentation_config: Optional[Dict] = None
    ):
        """
        Args:
            metadata_path: metadata.jsonl 파일 경로
            dataset_root: 데이터셋 루트 디렉토리
            sample_rate: 오디오 샘플링 레이트
            max_duration: 최대 오디오 길이 (초)
            task_ratios: 태스크별 샘플링 비율
            augmentation: 데이터 증강 여부
            use_processed: 전처리된 데이터 사용 여부
            use_tts: TTS 데이터 사용 여부
            tts_ratio: TTS 데이터 사용 비율
            augmentation_config: 증강 설정 (T-1)
        """
        self.metadata_path = Path(metadata_path)
        self.dataset_root = Path(dataset_root)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.augmentation = augmentation
        self.use_processed = use_processed
        self.use_tts = use_tts
        self.tts_ratio = tts_ratio
        
        # 기본 태스크 비율 설정 (TTS 추가)
        self.task_ratios = task_ratios or {
            'SONG': 0.50,
            'INST': 0.15,
            'COVER': 0.15,
            'TTS': 0.20  # TTS 태스크 추가
        }
        
        # 메타데이터 로드
        self.items = self._load_metadata()
        
        # 태스크별 인덱스 구성
        self.task_indices = self._organize_by_task()
        
        # 전처리된 데이터 경로
        self.processed_dir = self.dataset_root / "processed"
        
        # T-1: Enhanced Augmentation Setup
        if self.augmentation and AUGMENTATION_AVAILABLE:
            augmentation_config = augmentation_config or {}
            
            # Default enhanced augmentation settings
            default_config = {
                'sample_rate': self.sample_rate,
                'augmentation_prob': 0.8,
                'gain_range': (-3.0, 3.0),
                'pitch_range': (-0.5, 0.5),
                'tempo_range': (0.9, 1.1),
                'noise_level': 0.005,
                'reverb_prob': 0.3,
                'eq_prob': 0.4
            }
            
            # Override with user config
            default_config.update(augmentation_config)
            
            self.mix_scale_augmentation = MixScaleAugmentation(**default_config)
            
            # Additional lightweight augmentations for variety
            self.quick_augmentations = self._setup_quick_augmentations()
        else:
            self.mix_scale_augmentation = None
            self.quick_augmentations = None if not self.augmentation else self._setup_quick_augmentations()
        
        # 학습/평가 모드
        self._eval_mode = False
        
    def _setup_quick_augmentations(self):
        """Setup lightweight augmentations for fast processing"""
        return {
            'polarity_invert': lambda x: -x if random.random() < 0.1 else x,
            'channel_swap': lambda x: torch.stack([x[1], x[0]]) if x.shape[0] == 2 and random.random() < 0.1 else x,
            'mono_to_stereo': lambda x: x.repeat(2, 1) if x.shape[0] == 1 and random.random() < 0.2 else x,
            'stereo_balance': self._create_stereo_balance_aug(),
            'dc_offset': lambda x: x + (random.uniform(-0.001, 0.001) if random.random() < 0.3 else 0),
        }
    
    def _create_stereo_balance_aug(self):
        """Create stereo balance augmentation function"""
        def stereo_balance(audio):
            if audio.shape[0] == 2 and random.random() < 0.4:
                balance = random.uniform(0.7, 1.3)  # Left/right balance
                audio[0] *= balance
                audio[1] *= (2.0 - balance)
            return audio
        return stereo_balance
        
    def _load_metadata(self) -> List[Dict]:
        """metadata.jsonl 파일 로드"""
        items = []
        
        # 메타데이터 파일이 절대 경로가 아닌 경우 dataset_root 기준으로 처리
        if not self.metadata_path.is_absolute():
            metadata_path = self.dataset_root / self.metadata_path
        else:
            metadata_path = self.metadata_path
            
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # 경로 검증
                if self._validate_paths(item):
                    items.append(item)
                    
        print(f"Loaded {len(items)} valid items from {metadata_path}")
        return items
    
    def _validate_paths(self, item: Dict) -> bool:
        """파일 경로 존재 여부 검증"""
        # 기본 오디오 파일 체크
        audio_path = self.dataset_root / item['audio_path']
        if not audio_path.exists():
            return False
            
        # 가사 파일 체크 (선택적)
        if item.get('lyric_path'):
            lyric_path = self.dataset_root / item['lyric_path']
            if not lyric_path.exists():
                return False
        
        # TTS 관련 파일 체크 (선택적)        
        if self.use_tts and item.get('has_tts', False):
            if item.get('tts_audio_path'):
                tts_audio_path = self.dataset_root / item['tts_audio_path']
                if not tts_audio_path.exists():
                    print(f"Warning: TTS audio not found: {tts_audio_path}")
                    # TTS 파일이 없어도 일단 유효한 것으로 처리
                    
            if item.get('transcript_path'):
                transcript_path = self.dataset_root / item['transcript_path']
                if not transcript_path.exists():
                    print(f"Warning: Transcript not found: {transcript_path}")
                    
        return True
    
    def _organize_by_task(self) -> Dict[str, List[int]]:
        """태스크별로 데이터 인덱스 구성"""
        task_indices = {
            'SONG': [],
            'INST': [],
            'COVER': [],
            'TTS': []  # TTS 태스크 추가
        }
        
        for idx, item in enumerate(self.items):
            # TTS 데이터가 있는 경우 TTS 태스크로 분류
            if (self.use_tts and 
                item.get('has_tts', False) and 
                item.get('tts_audio_path') and 
                item.get('transcript_path')):
                task_indices['TTS'].append(idx)
                
            # 가사가 있으면 SONG 태스크
            if item.get('lyric_path'):
                task_indices['SONG'].append(idx)
                # COVER 태스크용으로도 사용 가능
                if random.random() < 0.3:  # 30% 확률로 COVER 데이터로도 사용
                    task_indices['COVER'].append(idx)
            else:
                # 가사가 없으면 INST 태스크
                task_indices['INST'].append(idx)
                
        # 빈 태스크 확인
        for task, indices in task_indices.items():
            if not indices:
                print(f"Warning: No data for task {task}")
                
        return task_indices
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        데이터 로드 및 전처리 with enhanced augmentation and TTS support
        
        Returns:
            Dict containing:
                - audio: (2, T) 스테레오 오디오
                - lyrics: 가사 텍스트 (optional)
                - transcript: TTS 전사본 (optional for TTS task)
                - genre: 장르 리스트
                - task_token: 태스크 타입
                - metadata: 원본 메타데이터
        """
        # 태스크 기반 샘플링
        task = self._sample_task()
        
        if task == 'COVER':
            # COVER 태스크는 참조 오디오도 필요
            return self._get_cover_item(idx)
        elif task == 'TTS':
            # TTS 태스크 처리
            return self._get_tts_item(idx)
        else:
            # SONG 또는 INST 태스크
            if task in self.task_indices and self.task_indices[task]:
                task_idx = random.choice(self.task_indices[task])
                item = self.items[task_idx]
            else:
                # 해당 태스크에 데이터가 없으면 기본 인덱스 사용
                item = self.items[idx]
            
            return self._process_item(item, task)
    
    def _sample_task(self) -> str:
        """태스크 비율에 따라 랜덤 샘플링"""
        rand_val = random.random()
        cumsum = 0.0
        
        for task, ratio in self.task_ratios.items():
            cumsum += ratio
            if rand_val < cumsum:
                return task
                
        return 'SONG'  # 기본값
    
    def _process_item(self, item: Dict, task: str) -> Dict:
        """단일 아이템 처리 with enhanced augmentation"""
        # 오디오 로드
        audio_path = self.dataset_root / item['audio_path']
        
        try:
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=False)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # 더미 오디오 반환
            audio = np.zeros((2, int(self.sample_rate * 1.0)))  # 1초 무음
            
        # 스테레오로 변환
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        elif audio.shape[0] > 2:
            audio = audio[:2]  # 처음 2채널만 사용
            
        # 길이 제한
        max_samples = int(self.max_duration * self.sample_rate)
        if audio.shape[1] > max_samples:
            # 랜덤 크롭
            start = random.randint(0, audio.shape[1] - max_samples)
            audio = audio[:, start:start + max_samples]
            
        # 가사 로드
        lyrics = None
        if item.get('lyric_path') and task != 'INST':
            lyric_path = self.dataset_root / item['lyric_path']
            try:
                with open(lyric_path, 'r', encoding='utf-8') as f:
                    lyrics = f.read().strip()
            except Exception as e:
                print(f"Error loading lyrics {lyric_path}: {e}")
                lyrics = ""
                
        # 장르 정보
        genre = item.get('genre', ['Unknown'])
        if isinstance(genre, str):
            genre = [genre]
            
        # Convert to tensor for augmentation
        audio_tensor = torch.from_numpy(audio).float()
        
        # T-1: Apply Enhanced Augmentation
        if self.augmentation and self.training:
            audio_tensor = self._apply_enhanced_augmentation(audio_tensor)
            
        return {
            'audio': audio_tensor,
            'lyrics': lyrics,
            'transcript': None,  # TTS 태스크가 아니므로 None
            'genre': genre,
            'task_token': f'<TASK={task}>',
            'metadata': item,
            'audio_length': audio_tensor.shape[1]
        }
    
    def _get_tts_item(self, idx: int) -> Dict:
        """TTS 태스크용 데이터 준비"""
        # TTS 데이터가 있는 항목 선택
        if 'TTS' in self.task_indices and self.task_indices['TTS']:
            tts_idx = random.choice(self.task_indices['TTS'])
            item = self.items[tts_idx]
        else:
            # TTS 데이터가 없으면 일반 데이터를 TTS처럼 처리
            print("Warning: No TTS data available, using regular data")
            item = self.items[idx]
            
        # TTS 오디오 로드
        tts_audio_path = None
        if item.get('tts_audio_path'):
            tts_audio_path = self.dataset_root / item['tts_audio_path']
            
        # TTS 오디오가 있으면 사용, 없으면 원본 오디오 사용
        if tts_audio_path and tts_audio_path.exists():
            audio_path = tts_audio_path
        else:
            audio_path = self.dataset_root / item['audio_path']
            
        try:
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=False)
        except Exception as e:
            print(f"Error loading TTS audio {audio_path}: {e}")
            # 더미 오디오 반환
            audio = np.zeros((2, int(self.sample_rate * 1.0)))
            
        # 스테레오로 변환
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        elif audio.shape[0] > 2:
            audio = audio[:2]
            
        # 길이 제한
        max_samples = int(self.max_duration * self.sample_rate)
        if audio.shape[1] > max_samples:
            start = random.randint(0, audio.shape[1] - max_samples)
            audio = audio[:, start:start + max_samples]
            
        # 전사본 로드
        transcript = None
        if item.get('transcript_path'):
            transcript_path = self.dataset_root / item['transcript_path']
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
            except Exception as e:
                print(f"Error loading transcript {transcript_path}: {e}")
                transcript = ""
        
        # 장르 정보
        genre = item.get('genre', ['TTS'])  # TTS 기본 장르
        if isinstance(genre, str):
            genre = [genre]
            
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # TTS에는 다른 증강 전략 적용 (더 보수적)
        if self.augmentation and self.training:
            audio_tensor = self._apply_tts_augmentation(audio_tensor)
            
        return {
            'audio': audio_tensor,
            'lyrics': None,  # TTS는 가사 대신 전사본 사용
            'transcript': transcript,
            'genre': genre,
            'task_token': '<TASK=TTS>',
            'metadata': item,
            'audio_length': audio_tensor.shape[1]
        }
    
    def _apply_tts_augmentation(self, audio: torch.Tensor) -> torch.Tensor:
        """TTS용 보수적 증강 적용"""
        # TTS는 말하기 데이터이므로 음악보다 보수적으로 증강
        if self.quick_augmentations:
            for aug_name, aug_func in self.quick_augmentations.items():
                if aug_name in ['polarity_invert', 'dc_offset'] and random.random() < 0.1:
                    try:
                        audio = aug_func(audio)
                    except Exception as e:
                        print(f"Warning: TTS augmentation {aug_name} failed: {e}")
                        continue
        
        # 간단한 노이즈만 추가
        if random.random() < 0.3:
            noise = torch.randn_like(audio) * 0.002  # 매우 작은 노이즈
            audio = audio + noise
            
        return audio
    
    def _apply_enhanced_augmentation(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply enhanced augmentation pipeline (T-1)
        
        Args:
            audio: (C, T) audio tensor
        Returns:
            augmented_audio: (C, T) enhanced augmented audio
        """
        # Apply quick augmentations first (very fast)
        if self.quick_augmentations:
            for aug_name, aug_func in self.quick_augmentations.items():
                if random.random() < 0.3:  # 30% chance for each quick aug
                    try:
                        audio = aug_func(audio)
                    except Exception as e:
                        # Skip augmentation if it fails
                        print(f"Warning: Quick augmentation {aug_name} failed: {e}")
                        continue
        
        # Apply advanced mix-scale augmentation
        if self.mix_scale_augmentation and AUGMENTATION_AVAILABLE:
            try:
                audio = self.mix_scale_augmentation(audio)
            except Exception as e:
                print(f"Warning: Mix-scale augmentation failed: {e}")
        
        return audio
    
    def _get_cover_item(self, idx: int) -> Dict:
        """COVER 태스크용 데이터 준비 with enhanced processing"""
        # 원본과 타겟 선택
        if 'SONG' in self.task_indices and len(self.task_indices['SONG']) >= 2:
            source_idx = random.choice(self.task_indices['SONG'])
            target_idx = random.choice(self.task_indices['SONG'])
        else:
            # SONG 데이터가 충분하지 않으면 전체 데이터에서 선택
            source_idx = random.randint(0, len(self.items) - 1)
            target_idx = random.randint(0, len(self.items) - 1)
        
        source_item = self.items[source_idx]
        target_item = self.items[target_idx]
        
        # 원본 오디오 로드 (참조용)
        source_data = self._process_item(source_item, 'SONG')
        
        # 타겟 데이터 (새로운 가사나 스타일)
        target_data = self._process_item(target_item, 'SONG')
        
        # ICL 참조 길이 설정 (10초 ~ 180초)
        ref_length = random.uniform(10.0, 180.0)
        ref_samples = int(ref_length * self.sample_rate)
        
        # 참조 오디오 자르기
        if source_data['audio'].shape[1] > ref_samples:
            source_data['audio'] = source_data['audio'][:, :ref_samples]
            
        # T-1: Apply different augmentation strategy for COVER task
        if self.augmentation and self.training and AUGMENTATION_AVAILABLE:
            # More conservative augmentation for reference audio
            if self.mix_scale_augmentation and random.random() < 0.6:  # Lower probability
                # Create a more conservative version of augmentation for reference
                from dcae.training_utils import MixScaleAugmentation
                conservative_aug = MixScaleAugmentation(
                    sample_rate=self.sample_rate,
                    augmentation_prob=0.5,  # Lower probability
                    gain_range=(-1.5, 1.5),  # Smaller range
                    pitch_range=(-0.25, 0.25),  # Smaller range
                    noise_level=0.002,  # Less noise
                    reverb_prob=0.15,  # Less reverb
                    eq_prob=0.2  # Less EQ
                )
                source_data['audio'] = conservative_aug(source_data['audio'])
            
        return {
            'audio': target_data['audio'],  # 타겟 오디오
            'reference_audio': source_data['audio'],  # 참조 오디오
            'reference_length': ref_length,
            'lyrics': target_data['lyrics'],
            'transcript': None,
            'genre': source_data['genre'],  # 원본 스타일 유지
            'task_token': '<TASK=COVER>',
            'metadata': {
                'source': source_item,
                'target': target_item
            }
        }
    
    @property
    def training(self):
        """학습 모드 여부"""
        return not self._eval_mode
    
    def eval(self):
        """평가 모드 설정"""
        self._eval_mode = True
        
    def train(self):
        """학습 모드 설정"""
        self._eval_mode = False
    
    def get_augmentation_stats(self) -> Dict[str, any]:
        """Get augmentation statistics for monitoring"""
        if not self.mix_scale_augmentation:
            return {"augmentation_enabled": False}
        
        stats = {
            "augmentation_enabled": True,
            "augmentation_available": AUGMENTATION_AVAILABLE,
            "use_tts": self.use_tts,
            "tts_ratio": self.tts_ratio,
        }
        
        if AUGMENTATION_AVAILABLE and self.mix_scale_augmentation:
            stats.update({
                "augmentation_prob": self.mix_scale_augmentation.augmentation_prob,
                "gain_range": self.mix_scale_augmentation.gain_range,
                "pitch_range": self.mix_scale_augmentation.pitch_range,
                "reverb_prob": self.mix_scale_augmentation.reverb_prob,
                "eq_prob": self.mix_scale_augmentation.eq_prob,
            })
            
        if self.quick_augmentations:
            stats["quick_augmentations"] = len(self.quick_augmentations)
            
        return stats
    
    def get_task_distribution(self) -> Dict[str, int]:
        """태스크별 데이터 분포 반환"""
        return {task: len(indices) for task, indices in self.task_indices.items()}


class LyroCollator:
    """
    Enhanced 배치 데이터 정리를 위한 Collator with TTS support
    
    가변 길이 오디오와 텍스트를 패딩하고
    배치 형태로 정리합니다.
    """
    
    def __init__(
        self,
        tokenizer=None,
        max_audio_length: int = None,
        max_text_length: int = 512,
        # T-1: Batch-level augmentation options
        batch_mix_prob: float = 0.1  # Probability of mixing samples in batch
    ):
        self.tokenizer = tokenizer
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.batch_mix_prob = batch_mix_prob
        
    def _apply_batch_mixing(self, batch_audios: torch.Tensor) -> torch.Tensor:
        """Apply batch-level mixing augmentation"""
        if random.random() > self.batch_mix_prob:
            return batch_audios
        
        batch_size = batch_audios.shape[0]
        if batch_size < 2:
            return batch_audios
            
        # Mix random pairs in the batch
        mix_indices = torch.randperm(batch_size)
        mix_lambda = torch.rand(batch_size, 1, 1) * 0.3  # Mix ratio 0-30%
        
        mixed_batch = batch_audios.clone()
        for i in range(0, batch_size - 1, 2):
            if i + 1 < batch_size:
                idx1, idx2 = i, i + 1
                lambda_val = mix_lambda[idx1]
                mixed_batch[idx1] = lambda_val * batch_audios[idx1] + (1 - lambda_val) * batch_audios[idx2]
                
        return mixed_batch
        
    def __call__(self, batch: List[Dict]) -> Dict:
        """배치 데이터 정리 with enhanced processing and TTS support"""
        # 오디오 패딩
        audio_lengths = [item['audio'].shape[1] for item in batch]
        max_len = max(audio_lengths) if not self.max_audio_length else \
                  min(max(audio_lengths), self.max_audio_length)
        
        padded_audios = []
        for item in batch:
            audio = item['audio']
            if audio.shape[1] > max_len:
                audio = audio[:, :max_len]
            elif audio.shape[1] < max_len:
                pad_len = max_len - audio.shape[1]
                # Use reflection padding to avoid artifacts (but limit to available length)
                if audio.shape[1] > 1:
                    # Reflection padding
                    pad_mode = 'reflect'
                else:
                    # If audio is too short, use constant padding
                    pad_mode = 'constant'
                audio = torch.nn.functional.pad(audio, (0, pad_len), mode=pad_mode)
            padded_audios.append(audio)
            
        batch_audios = torch.stack(padded_audios)
        
        # T-1: Apply batch-level mixing augmentation
        batch_audios = self._apply_batch_mixing(batch_audios)
        
        # 가사 토큰화 (tokenizer가 있는 경우)
        lyrics_tokens = None
        if self.tokenizer and any(item.get('lyrics') for item in batch):
            lyrics_list = [item.get('lyrics', '') for item in batch]
            lyrics_tokens = self.tokenizer.batch_encode(
                lyrics_list, 
                max_length=self.max_text_length
            )
        
        # TTS 전사본 토큰화
        transcript_tokens = None
        if self.tokenizer and any(item.get('transcript') for item in batch):
            transcript_list = [item.get('transcript', '') for item in batch]
            transcript_tokens = self.tokenizer.batch_encode(
                transcript_list,
                max_length=self.max_text_length
            )
            
        # 배치 구성
        batch_dict = {
            'audio': batch_audios,
            'audio_lengths': torch.tensor(audio_lengths),
            'task_tokens': [item['task_token'] for item in batch],
            'genres': [item['genre'] for item in batch],
        }
        
        if lyrics_tokens is not None:
            batch_dict['lyrics_tokens'] = lyrics_tokens
            
        if transcript_tokens is not None:
            batch_dict['transcript_tokens'] = transcript_tokens
            
        # COVER 태스크 처리
        if any('reference_audio' in item for item in batch):
            ref_audios = []
            ref_lengths = []
            for item in batch:
                if 'reference_audio' in item:
                    ref_audios.append(item['reference_audio'])
                    ref_lengths.append(item['reference_length'])
                else:
                    # 더미 참조 (COVER가 아닌 경우)
                    ref_audios.append(torch.zeros_like(item['audio'][:, :1000]))
                    ref_lengths.append(0.0)
                    
            # Pad reference audios
            max_ref_len = max(ref_audio.shape[1] for ref_audio in ref_audios)
            padded_ref_audios = []
            for ref_audio in ref_audios:
                if ref_audio.shape[1] < max_ref_len:
                    pad_len = max_ref_len - ref_audio.shape[1]
                    if ref_audio.shape[1] > 1:
                        pad_mode = 'reflect'
                    else:
                        pad_mode = 'constant'
                    ref_audio = torch.nn.functional.pad(ref_audio, (0, pad_len), mode=pad_mode)
                padded_ref_audios.append(ref_audio)
                    
            batch_dict['reference_audios'] = torch.stack(padded_ref_audios)
            batch_dict['reference_lengths'] = torch.tensor(ref_lengths)
            
        return batch_dict