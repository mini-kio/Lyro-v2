# lyro/data/dataset.py
"""
LYRO Dataset - Unified and simplified
Supports lyrics, captions, and reference audio
"""

import os
import json
import random
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset
from tqdm import tqdm


class LyroDataset(Dataset):
    """
    Unified LYRO dataset for music generation
    Supports multiple conditioning modalities
    """
    
    def __init__(
        self,
        metadata_path: str,
        dataset_root: str,
        tokenizer: Any,
        max_audio_length: int = 441000,  # 10 seconds at 44.1kHz
        max_text_length: int = 512,
        task_ratios: Dict[str, float] = None,
        augmentation: bool = True,
        cache_audio: bool = False,
        preload_count: int = 0,
        filter_corrupted: bool = True
    ):
        self.metadata_path = Path(metadata_path)
        self.dataset_root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.augmentation = augmentation
        self.cache_audio = cache_audio
        self.filter_corrupted = filter_corrupted
        
        # Default task ratios
        if task_ratios is None:
            task_ratios = {
                'SONG': 0.6,     # Lyrics + audio
                'INST': 0.2,     # Caption + audio (no lyrics)
                'COVER': 0.2     # Reference + lyrics + audio
            }
        self.task_ratios = task_ratios
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Filter and balance by tasks
        self.samples = self._prepare_samples()
        
        # Audio cache
        self.audio_cache = {} if cache_audio else None
        
        # Preload some samples
        if preload_count > 0:
            self._preload_samples(min(preload_count, len(self.samples)))
            
        print(f"Loaded {len(self.samples)} samples from {self.metadata_path}")
        
    def _load_metadata(self) -> List[Dict]:
        """Load metadata from JSONL file"""
        metadata = []
        
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
            
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    
                    # Validate required fields
                    if 'id' not in item or 'audio_path' not in item:
                        continue
                        
                    # Check if files exist
                    audio_path = self.dataset_root / item['audio_path']
                    if not audio_path.exists():
                        if self.filter_corrupted:
                            continue
                        
                    metadata.append(item)
                    
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON at line {line_num}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue
                    
        return metadata
    
    def _prepare_samples(self) -> List[Dict]:
        """Prepare and balance samples by task"""
        samples = []
        
        # Separate samples by available modalities
        song_samples = []      # Have lyrics
        inst_samples = []      # No lyrics but have captions/genre
        cover_samples = []     # Have reference audio capability
        
        for item in self.metadata:
            sample = {
                'id': item['id'],
                'audio_path': item['audio_path'],
                'lyrics': item.get('lyrics', ''),
                'caption': self._generate_caption(item),
                'genre': item.get('genre', ['unknown']),
                'reference_path': item.get('reference_path'),  # For covers
                'metadata': item
            }
            
            # Categorize by task capability
            has_lyrics = bool(sample['lyrics'])
            has_reference = bool(sample['reference_path'])
            
            if has_reference:
                cover_samples.append(sample)
            elif has_lyrics:
                song_samples.append(sample)
            else:
                inst_samples.append(sample)
                
        # Balance according to task ratios
        total_samples = len(self.metadata)
        target_song = int(total_samples * self.task_ratios.get('SONG', 0.6))
        target_inst = int(total_samples * self.task_ratios.get('INST', 0.2))
        target_cover = int(total_samples * self.task_ratios.get('COVER', 0.2))
        
        # Sample from each category
        samples.extend(self._sample_category(song_samples, target_song, 'SONG'))
        samples.extend(self._sample_category(inst_samples, target_inst, 'INST'))
        samples.extend(self._sample_category(cover_samples, target_cover, 'COVER'))
        
        # Shuffle final samples
        random.shuffle(samples)
        
        return samples
    
    def _sample_category(self, category_samples: List[Dict], target_count: int, task_type: str) -> List[Dict]:
        """Sample from a category to reach target count"""
        if not category_samples:
            return []
            
        if len(category_samples) >= target_count:
            # Randomly sample
            sampled = random.sample(category_samples, target_count)
        else:
            # Repeat samples to reach target
            sampled = category_samples * (target_count // len(category_samples))
            remaining = target_count % len(category_samples)
            if remaining > 0:
                sampled.extend(random.sample(category_samples, remaining))
                
        # Add task type to each sample
        for sample in sampled:
            sample['task'] = task_type
            
        return sampled
    
    def _generate_caption(self, item: Dict) -> str:
        """Generate MusicCaps-style caption from metadata"""
        caption_parts = []
        
        # Genre information
        genres = item.get('genre', [])
        if genres and genres != ['unknown']:
            if len(genres) == 1:
                caption_parts.append(f"This is a {genres[0]} song")
            else:
                caption_parts.append(f"This is a {', '.join(genres[:-1])} and {genres[-1]} song")
        else:
            caption_parts.append("This is a music piece")
            
        # Tempo/mood (if available)
        tempo = item.get('tempo')
        if tempo:
            if tempo > 140:
                caption_parts.append("with a fast tempo")
            elif tempo < 80:
                caption_parts.append("with a slow tempo")
            else:
                caption_parts.append("with a moderate tempo")
                
        # Instruments (if available)
        instruments = item.get('instruments', [])
        if instruments:
            if len(instruments) <= 3:
                caption_parts.append(f"featuring {', '.join(instruments)}")
            else:
                caption_parts.append(f"featuring {', '.join(instruments[:3])} and other instruments")
                
        # Energy/mood
        energy = item.get('energy', 0.5)
        if energy > 0.7:
            caption_parts.append("with high energy")
        elif energy < 0.3:
            caption_parts.append("with low energy")
            
        # Combine parts
        if len(caption_parts) > 1:
            caption = caption_parts[0] + " " + ", ".join(caption_parts[1:]) + "."
        else:
            caption = caption_parts[0] + "."
            
        return caption
    
    def _preload_samples(self, count: int):
        """Preload some samples for faster access"""
        print(f"Preloading {count} samples...")
        
        for i in tqdm(range(count), desc="Preloading"):
            try:
                sample = self.samples[i]
                audio_path = self.dataset_root / sample['audio_path']
                
                if audio_path.exists():
                    audio, sr = torchaudio.load(audio_path)
                    if self.audio_cache is not None:
                        self.audio_cache[sample['id']] = (audio, sr)
                        
            except Exception as e:
                print(f"Warning: Failed to preload sample {i}: {e}")
                continue
                
    def _load_audio(self, audio_path: str, sample_id: str) -> Tuple[torch.Tensor, int]:
        """Load audio with caching support"""
        # Check cache first
        if self.audio_cache is not None and sample_id in self.audio_cache:
            return self.audio_cache[sample_id]
            
        # Load from disk
        full_path = self.dataset_root / audio_path
        
        try:
            audio, sr = torchaudio.load(full_path)
            
            # Cache if enabled
            if self.audio_cache is not None:
                self.audio_cache[sample_id] = (audio, sr)
                
            return audio, sr
            
        except Exception as e:
            print(f"Warning: Failed to load audio {full_path}: {e}")
            # Return dummy audio
            sr = 44100
            audio = torch.randn(2, sr)  # 1 second of noise
            return audio, sr
    
    def _process_audio(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Process audio to standard format"""
        # Resample if needed
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            audio = resampler(audio)
            
        # Ensure stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]
            
        # Normalize length
        current_length = audio.shape[-1]
        if current_length > self.max_audio_length:
            # Random crop or center crop
            if self.augmentation:
                start_idx = random.randint(0, current_length - self.max_audio_length)
            else:
                start_idx = (current_length - self.max_audio_length) // 2
            audio = audio[:, start_idx:start_idx + self.max_audio_length]
        elif current_length < self.max_audio_length:
            # Pad with zeros
            pad_length = self.max_audio_length - current_length
            audio = torch.nn.functional.pad(audio, (0, pad_length))
            
        # Augmentation
        if self.augmentation:
            audio = self._augment_audio(audio)
            
        return audio
    
    def _augment_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentations"""
        # Random gain
        if random.random() < 0.3:
            gain_db = random.uniform(-3, 3)
            gain_linear = 10 ** (gain_db / 20)
            audio = audio * gain_linear
            
        # Random noise
        if random.random() < 0.1:
            noise_level = random.uniform(0.001, 0.01)
            noise = torch.randn_like(audio) * noise_level
            audio = audio + noise
            
        # Clip to reasonable range
        audio = torch.clamp(audio, -1.0, 1.0)
        
        return audio
    
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
            token_ids = self.tokenizer.encode(text, max_length=self.max_text_length)
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
        """Get a single sample"""
        if idx >= len(self.samples):
            idx = idx % len(self.samples)
            
        sample = self.samples[idx]
        
        try:
            # Load and process audio
            audio, sr = self._load_audio(sample['audio_path'], sample['id'])
            audio = self._process_audio(audio, sr)
            
            # Process lyrics
            lyrics_data = self._process_text(sample['lyrics'], 'lyrics')
            
            # Process caption
            caption_data = self._process_text(sample['caption'], 'caption')
            
            # Load reference audio (for COVER task)
            reference_audio = None
            if sample['task'] == 'COVER' and sample.get('reference_path'):
                try:
                    ref_audio, ref_sr = self._load_audio(sample['reference_path'], f"{sample['id']}_ref")
                    reference_audio = self._process_audio(ref_audio, ref_sr)
                except Exception:
                    # Use same audio as reference if loading fails
                    reference_audio = audio.clone()
            
            return {
                # Audio data
                'audio': audio,
                'reference_audio': reference_audio,
                
                # Text data
                'lyrics_tokens': lyrics_data['tokens'],
                'lyrics_attention_mask': lyrics_data['attention_mask'],
                'lyrics_length': lyrics_data['length'],
                
                'caption_tokens': caption_data['tokens'],
                'caption_attention_mask': caption_data['attention_mask'],
                'caption_length': caption_data['length'],
                
                # Raw text for encoder processing
                'lyrics_text': sample['lyrics'],
                'caption_text': sample['caption'],
                
                # Task and metadata
                'task': sample['task'],
                'genre': sample['genre'],
                'id': sample['id'],
                
                # Original metadata
                'metadata': sample['metadata']
            }
            
        except Exception as e:
            print(f"Warning: Error loading sample {idx} ({sample['id']}): {e}")
            
            # Return dummy sample
            dummy_audio = torch.randn(2, self.max_audio_length)
            dummy_tokens = torch.zeros(self.max_text_length, dtype=torch.long)
            dummy_mask = torch.zeros(self.max_text_length, dtype=torch.bool)
            
            return {
                'audio': dummy_audio,
                'reference_audio': None,
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
    Create train, validation, and test datasets
    
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
    
    print(f"Created datasets:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    if test_dataset:
        print(f"  Test: {len(test_dataset)} samples")
        
    return train_dataset, val_dataset, test_dataset