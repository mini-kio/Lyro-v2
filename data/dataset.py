# lyro/data/dataset.py
"""
LYRO Dataset - Latent Vector 기반 구조 (수정됨)
Supports task-specific metadata with latent vectors
"""

import os
import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset
from tqdm import tqdm


class LyroDataset(Dataset):
    """
    Unified LYRO dataset with latent-based structure (수정됨)
    Supports SONG, INST, and COVER tasks with latent vectors
    """
    
    def __init__(
        self,
        metadata_path: str,
        dataset_root: str,
        tokenizer: Any,
        latent_channels: int = 16,
        latent_time_steps: int = 128,
        max_text_length: int = 512,
        task_ratios: Dict[str, float] = None,
        augmentation: bool = True,
        cache_latents: bool = False,
        preload_count: int = 0,
        filter_corrupted: bool = True
    ):
        self.metadata_path = Path(metadata_path)
        self.dataset_root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.latent_channels = latent_channels
        self.latent_time_steps = latent_time_steps
        self.max_text_length = max_text_length
        self.augmentation = augmentation
        self.cache_latents = cache_latents
        self.filter_corrupted = filter_corrupted
        
        # Default task ratios
        if task_ratios is None:
            task_ratios = {
                'SONG': 0.6,     # Lyrics + latents (no reference)
                'INST': 0.3,     # Caption + latents (optional reference)
                'COVER': 0.1     # Reference + lyrics/caption + latents
            }
        self.task_ratios = task_ratios
        
        # Load and validate metadata
        self.metadata = self._load_metadata()
        
        # Prepare samples with task validation
        self.samples = self._prepare_samples()
        
        # Latent cache
        self.latent_cache = {} if cache_latents else None
        
        # Preload some samples
        if preload_count > 0:
            self._preload_samples(min(preload_count, len(self.samples)))
            
        print(f"Loaded {len(self.samples)} samples from {self.metadata_path}")
        print(f"Task distribution: {self._get_task_distribution()}")
        
    def _load_metadata(self) -> List[Dict]:
        """Load and validate metadata from JSONL file"""
        metadata = []
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
            
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    
                    # Validate and normalize item
                    if self._validate_metadata_item(item, line_num):
                        metadata.append(item)
                        
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON at line {line_num}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue
                    
        return metadata
    
    def _validate_metadata_item(self, item: Dict, line_num: int) -> bool:
        """
        Validate metadata item according to task-specific rules (수정됨 - Latent 기반)
        
        Task validation rules:
        - SONG: must have lyrics, no reference_path
        - INST: no lyrics, optional reference_path
        - COVER: must have reference_path, lyrics optional (for instrumental covers)
        """
        # Basic required fields (latent_path로 변경)
        if 'id' not in item or 'latent_path' not in item or 'task' not in item:
            if self.filter_corrupted:
                print(f"Warning: Missing required fields at line {line_num}")
                return False
        
        task = item.get('task', '').upper()
        
        # Validate task type
        if task not in ['SONG', 'INST', 'COVER']:
            if self.filter_corrupted:
                print(f"Warning: Invalid task '{task}' at line {line_num}")
                return False
            # Default to INST if task is invalid
            item['task'] = 'INST'
            task = 'INST'
        
        # Task-specific validation
        has_lyrics = bool(item.get('lyrics', '').strip())
        has_reference = bool(item.get('reference_latent_path', '').strip())
        
        if task == 'SONG':
            # SONG: must have lyrics, no reference
            if not has_lyrics:
                if self.filter_corrupted:
                    print(f"Warning: SONG task missing lyrics at line {line_num}")
                    return False
            if has_reference:
                if self.filter_corrupted:
                    print(f"Warning: SONG task should not have reference_latent_path at line {line_num}")
                # Remove reference for SONG task
                item.pop('reference_latent_path', None)
        
        elif task == 'INST':
            # INST: no lyrics, optional reference
            if has_lyrics:
                if self.filter_corrupted:
                    print(f"Warning: INST task should not have lyrics at line {line_num}")
                # Remove lyrics for INST task
                item.pop('lyrics', None)
        
        elif task == 'COVER':
            # COVER: must have reference, lyrics optional
            if not has_reference:
                if self.filter_corrupted:
                    print(f"Warning: COVER task missing reference_latent_path at line {line_num}")
                    return False
        
        # Validate file existence (latent files)
        latent_path = self.dataset_root / item['latent_path']
        if not latent_path.exists():
            if self.filter_corrupted:
                print(f"Warning: Latent file not found: {latent_path}")
                return False
        
        # Validate reference latent existence for COVER tasks
        if task == 'COVER' and has_reference:
            ref_path = self.dataset_root / item['reference_latent_path']
            if not ref_path.exists():
                if self.filter_corrupted:
                    print(f"Warning: Reference latent file not found: {ref_path}")
                    return False
        
        # Ensure caption field exists (empty string if not provided)
        if 'caption' not in item:
            item['caption'] = ''
        
        return True
    

    def _prepare_samples(self) -> List[Dict]:
        """Prepare samples with task-based distribution"""
        # Separate samples by task
        task_samples = {
            'SONG': [],
            'INST': [], 
            'COVER': []
        }
        
        for item in self.metadata:
            task = item['task'].upper()
            
            sample = {
                'id': item['id'],
                'task': task,
                'latent_path': item['latent_path'],
                'lyrics': item.get('lyrics', ''),
                'caption': item.get('caption', ''),
                'genre': item.get('genre', ['unknown']),
                'reference_latent_path': item.get('reference_latent_path'),
                'metadata': item
            }
            
            if task in task_samples:
                task_samples[task].append(sample)
            else:
                # Default to INST for unknown tasks
                task_samples['INST'].append(sample)
        
        # Balance according to task ratios
        total_samples = len(self.metadata)
        balanced_samples = []
        
        for task, ratio in self.task_ratios.items():
            available_samples = task_samples.get(task, [])
            target_count = int(total_samples * ratio)
            
            if not available_samples:
                continue
            
            if len(available_samples) >= target_count:
                # Randomly sample
                sampled = random.sample(available_samples, target_count)
            else:
                # Repeat samples to reach target
                sampled = available_samples * (target_count // len(available_samples))
                remaining = target_count % len(available_samples)
                if remaining > 0:
                    sampled.extend(random.sample(available_samples, remaining))
            
            balanced_samples.extend(sampled)
        
        # Shuffle final samples
        random.shuffle(balanced_samples)
        
        return balanced_samples
    
    def _get_task_distribution(self) -> Dict[str, int]:
        """Get current task distribution"""
        distribution = {}
        for sample in self.samples:
            task = sample['task']
            distribution[task] = distribution.get(task, 0) + 1
        return distribution
    
    def _preload_samples(self, count: int):
        """Preload some samples for faster access"""
        print(f"Preloading {count} latent samples...")
        
        for i in tqdm(range(count), desc="Preloading"):
            try:
                sample = self.samples[i]
                latent_path = self.dataset_root / sample['latent_path']
                
                if latent_path.exists():
                    latents = self._load_latents(latent_path, sample['id'])
                    if self.latent_cache is not None:
                        self.latent_cache[sample['id']] = latents
                        
            except Exception as e:
                print(f"Warning: Failed to preload sample {i}: {e}")
                continue
                
    def _load_latents(self, latent_path: str, sample_id: str) -> torch.Tensor:
        """Load latent vectors with caching support"""
        # Check cache first
        if self.latent_cache is not None and sample_id in self.latent_cache:
            return self.latent_cache[sample_id]
            
        # Load from disk
        full_path = self.dataset_root / latent_path
        
        try:
            # Load latent vectors (.npy or .pt files)
            if full_path.suffix == '.npy':
                latents_np = np.load(full_path)
                latents = torch.from_numpy(latents_np).float()
            elif full_path.suffix == '.pt':
                latents = torch.load(full_path)
            else:
                raise ValueError(f"Unsupported latent file format: {full_path.suffix}")
            
            # Cache if enabled
            if self.latent_cache is not None:
                self.latent_cache[sample_id] = latents
                
            return latents
            
        except Exception as e:
            print(f"Warning: Failed to load latents {full_path}: {e}")
            # Return dummy latents
            return torch.randn(self.latent_channels, self.latent_time_steps)
    
    def _process_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Process latent vectors to standard format"""
        # 차원 정규화
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)  # (C, T) -> (1, C, T)
        elif latents.dim() == 1:
            latents = latents.unsqueeze(0).unsqueeze(0)  # (T,) -> (1, 1, T)
        elif latents.dim() != 3:
            raise ValueError(f"Expected 2D or 3D latent tensor, got {latents.shape}")
        
        # 채널 수 맞춤
        if latents.shape[1] != self.latent_channels:
            if latents.shape[1] < self.latent_channels:
                # 패딩
                pad_channels = self.latent_channels - latents.shape[1]
                latents = torch.nn.functional.pad(latents, (0, 0, 0, pad_channels))
            else:
                # 크롭
                latents = latents[:, :self.latent_channels, :]
        
        # 시간 길이 맞춤
        if latents.shape[2] != self.latent_time_steps:
            if latents.shape[2] > self.latent_time_steps:
                # Random crop or center crop
                if self.augmentation:
                    start_idx = random.randint(0, latents.shape[2] - self.latent_time_steps)
                else:
                    start_idx = (latents.shape[2] - self.latent_time_steps) // 2
                latents = latents[:, :, start_idx:start_idx + self.latent_time_steps]
            else:
                # Pad with zeros
                pad_length = self.latent_time_steps - latents.shape[2]
                latents = torch.nn.functional.pad(latents, (0, pad_length))
            
        # Remove batch dimension if present
        if latents.shape[0] == 1:
            latents = latents.squeeze(0)  # (1, C, T) -> (C, T)
            
        # Augmentation
        if self.augmentation:
            latents = self._augment_latents(latents)
            
        return latents
    
    def _augment_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply latent augmentations"""
        # Random scaling
        if random.random() < 0.3:
            scale_factor = random.uniform(0.8, 1.2)
            latents = latents * scale_factor
            
        # Random noise
        if random.random() < 0.2:
            noise_level = random.uniform(0.01, 0.05)
            noise = torch.randn_like(latents) * noise_level
            latents = latents + noise
            
        # Random channel shuffle (가끔)
        if random.random() < 0.1:
            channel_order = torch.randperm(latents.shape[0])
            latents = latents[channel_order]
        
        return latents
    
    def _process_text(self, text: str, text_type: str = 'lyrics') -> Dict[str, torch.Tensor]:
        """Process text (lyrics or captions) with tokenizer"""
        if not text or text.strip() == '':
            # Empty text
            tokens = torch.zeros(self.max_text_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_text_length, dtype=torch.bool)
            return {
                'tokens': tokens,
                'attention_mask': attention_mask,
                'length': 0
            }
            
        # Tokenize
        if hasattr(self.tokenizer, 'encode'):
            if text_type == 'lyrics':
                token_ids = self.tokenizer.encode_lyrics(text)
            else:
                token_ids = self.tokenizer.encode_caption(text)
        else:
            # Fallback: character-level tokenization
            token_ids = [ord(c) % 1000 for c in text[:self.max_text_length]]
            
        # Convert to tensor and pad
        tokens = torch.zeros(self.max_text_length, dtype=torch.long)
        attention_mask = torch.zeros(self.max_text_length, dtype=torch.bool)
        
        length = min(len(token_ids), self.max_text_length)
        tokens[:length] = torch.tensor(token_ids[:length], dtype=torch.long)
        attention_mask[:length] = True
        
        return {
            'tokens': tokens,
            'attention_mask': attention_mask,
            'length': length
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with task-specific processing (수정됨 - Latent 기반)"""
        if idx >= len(self.samples):
            idx = idx % len(self.samples)
            
        sample = self.samples[idx]
        
        try:
            # Load and process main latents
            latents = self._load_latents(sample['latent_path'], sample['id'])
            latents = self._process_latents(latents)
            
            # Process lyrics (only for SONG and some COVER tasks)
            task = sample['task']
            if task in ['SONG', 'COVER'] and sample['lyrics']:
                lyrics_data = self._process_text(sample['lyrics'], 'lyrics')
            else:
                # Empty lyrics for INST tasks
                lyrics_data = self._process_text('', 'lyrics')
            
            # Process caption (all tasks have captions)
            caption_data = self._process_text(sample['caption'], 'caption')
            
            # Load reference latents (for COVER and optionally INST tasks)
            reference_latents = None
            if sample.get('reference_latent_path'):
                try:
                    ref_latents = self._load_latents(sample['reference_latent_path'], f"{sample['id']}_ref")
                    reference_latents = self._process_latents(ref_latents)
                except Exception as e:
                    print(f"Warning: Failed to load reference latents: {e}")
                    reference_latents = None
            
            return {
                # Latent data
                'latents': latents,
                'reference_latents': reference_latents,
                
                # Text data
                'lyrics_tokens': lyrics_data['tokens'],
                'lyrics_attention_mask': lyrics_data['attention_mask'],
                'lyrics_length': lyrics_data['length'],
                
                'caption_tokens': caption_data['tokens'],
                'caption_attention_mask': caption_data['attention_mask'],
                'caption_length': caption_data['length'],
                
                # Raw text for encoder processing
                'lyrics_text': sample['lyrics'] if task in ['SONG', 'COVER'] else '',
                'caption_text': sample['caption'],
                
                # Task and metadata
                'task': task,
                'genre': sample['genre'],
                'id': sample['id'],
                
                # Original metadata
                'metadata': sample['metadata']
            }
            
        except Exception as e:
            print(f"Warning: Error loading sample {idx} ({sample['id']}): {e}")
            
            # Return dummy sample
            dummy_latents = torch.randn(self.latent_channels, self.latent_time_steps)
            dummy_tokens = torch.zeros(self.max_text_length, dtype=torch.long)
            dummy_mask = torch.zeros(self.max_text_length, dtype=torch.bool)
            
            return {
                'latents': dummy_latents,
                'reference_latents': None,
                'lyrics_tokens': dummy_tokens,
                'lyrics_attention_mask': dummy_mask,
                'lyrics_length': 0,
                'caption_tokens': dummy_tokens,
                'caption_attention_mask': dummy_mask,
                'caption_length': 0,
                'lyrics_text': '',
                'caption_text': 'This is a music piece.',
                'task': 'INST',
                'genre': ['unknown'],
                'id': f'dummy_{idx}',
                'metadata': {}
            }


def create_lyro_datasets(
    train_metadata: str,
    val_metadata: str,
    test_metadata: Optional[str],
    dataset_root: str,
    tokenizer: Any,
    **kwargs
) -> Tuple[LyroDataset, LyroDataset, Optional[LyroDataset]]:
    """
    Create train, validation, and test datasets with latent validation (수정됨)
    
    Args:
        train_metadata: Path to training metadata
        val_metadata: Path to validation metadata  
        test_metadata: Path to test metadata (optional)
        dataset_root: Root directory of dataset
        tokenizer: Text tokenizer
        **kwargs: Additional dataset arguments
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    
    # Training dataset with augmentation
    train_dataset = LyroDataset(
        metadata_path=train_metadata,
        dataset_root=dataset_root,
        tokenizer=tokenizer,
        augmentation=True,
        **kwargs
    )
    
    # Validation dataset without augmentation
    val_dataset = LyroDataset(
        metadata_path=val_metadata,
        dataset_root=dataset_root,
        tokenizer=tokenizer,
        augmentation=False,
        **kwargs
    )
    
    # Test dataset (optional)
    test_dataset = None
    if test_metadata and Path(test_metadata).exists():
        test_dataset = LyroDataset(
            metadata_path=test_metadata,
            dataset_root=dataset_root,
            tokenizer=tokenizer,
            augmentation=False,
            **kwargs
        )
    
    print(f"Created latent-based datasets:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    if test_dataset:
        print(f"  Test: {len(test_dataset)} samples")
        
    return train_dataset, val_dataset, test_dataset