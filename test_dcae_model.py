# test_dcae_model.py - DCAE Model Inference with Chunking + Overlap (FIXED)
"""
DCAE Inference - Fixed version with proper dtype handling
Supports chunking with overlap for processing audio of any length
"""

import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, Optional, List, Any, Tuple
import warnings
from tqdm import tqdm
import glob
import math

# Accelerate imports for model loading
from accelerate import Accelerator
from accelerate.utils import set_seed

# Add project path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DCAE imports
from dcae.model import create_large_dcae_model

warnings.filterwarnings("ignore")


class DCAEInference:
    """DCAE Model Inference Engine with Chunking Support - FIXED VERSION"""
    
    def __init__(self, checkpoint_path: str, device: str = "auto", chunk_duration: float = 2.0, overlap_ratio: float = 0.25, use_fp16: bool = True, skip_weights: bool = False):
        """
        Initialize DCAE inference
        
        Args:
            checkpoint_path: Path to model checkpoint directory
            device: Device to use ("auto", "cuda", "cpu")
            chunk_duration: Duration of each chunk in seconds
            overlap_ratio: Overlap ratio between chunks (0.0 to 0.5)
            use_fp16: Whether to use FP16 precision (False for debugging)
            skip_weights: Skip loading weights (use random initialization for testing)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.chunk_duration = chunk_duration
        self.overlap_ratio = max(0.0, min(0.5, overlap_ratio))
        self.sample_rate = 44100
        self.use_fp16 = use_fp16 and torch.cuda.is_available()  # Only use FP16 on CUDA
        self.skip_weights = skip_weights
        
        # Calculate chunk parameters
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.overlap_samples = int(self.chunk_samples * self.overlap_ratio)
        self.hop_samples = self.chunk_samples - self.overlap_samples
        
        # Auto device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🚀 DCAE Inference Engine Initializing...")
        print(f"💻 Device: {self.device}")
        print(f"🔧 Mixed Precision: {'FP16' if self.use_fp16 else 'FP32'}")
        print(f"📁 Checkpoint: {self.checkpoint_path}")
        print(f"🔧 Chunk Duration: {self.chunk_duration}s ({self.chunk_samples} samples)")
        print(f"🔄 Overlap Ratio: {self.overlap_ratio} ({self.overlap_samples} samples)")
        print(f"⚡ Hop Size: {self.hop_samples} samples")
        if skip_weights:
            print("⚠️ WARNING: Skipping weight loading!")
        
        # Load model
        self._load_model()
        
        print(f"✅ DCAE Inference Engine Ready!")
    
    def _load_model(self):
        """Load trained model from checkpoint with proper dtype handling"""
        try:
            # Load metadata first
            metadata_path = self.checkpoint_path / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                print(f"📊 Model Info:")
                print(f"   - Epoch: {self.metadata.get('epoch', 'Unknown')}")
                print(f"   - Model Type: {self.metadata.get('model_config', {}).get('type', 'Unknown')}")
                print(f"   - Mixed Precision: {self.metadata.get('model_config', {}).get('mixed_precision', 'Unknown')}")
            else:
                print("⚠️ No metadata found, using default configuration")
                self.metadata = {}
            
            # Create model with large configuration - ALWAYS IN FP32 FIRST
            print("🏗️ Creating model in FP32...")
            self.model = create_large_dcae_model(
                sample_rate=self.sample_rate,
                latent_channels=16,
                base_channels=128,
                s6_layers=[3, 4, 4]
            )
            
            # Move to device BEFORE loading weights
            self.model = self.model.to(self.device)
            
            if self.skip_weights:
                print("⚠️ Skipping weight loading - using random initialization!")
            else:
                # Load model weights
                model_path = self.checkpoint_path / 'pytorch_model.bin'
                if not model_path.exists():
                    # Try alternative paths
                    model_files = list(self.checkpoint_path.glob('*.bin'))
                    if model_files:
                        model_path = model_files[0]
                    else:
                        raise FileNotFoundError(f"No model file found in {self.checkpoint_path}")
                
                # Load state dict
                print(f"📥 Loading model weights from {model_path}")
                
                # Load to CPU first to avoid device mismatch
                state_dict = torch.load(model_path, map_location='cpu')
                
                # Debug: Print state dict structure
                print(f"🔍 State dict keys: {list(state_dict.keys())}")
                
                # Handle different state dict formats
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                    print("📦 Found 'model' key in checkpoint")
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                    print("📦 Found 'state_dict' key in checkpoint")
                elif 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                    print("📦 Found 'model_state_dict' key in checkpoint")
                else:
                    print("📦 Using root level as state dict")
                
                # Debug: Print a few sample keys and their types
                sample_keys = list(state_dict.keys())[:3]
                for key in sample_keys:
                    value = state_dict[key]
                    print(f"🔍 Sample key '{key}': type={type(value)}")
                    if isinstance(value, torch.Tensor):
                        print(f"    Tensor shape={value.shape}, dtype={value.dtype}")
                    elif isinstance(value, dict):
                        print(f"    Dict with keys: {list(value.keys())[:3]}...")
                
                # Remove any prefixes from FSDP/DDP training
                cleaned_state_dict = {}
                for key, value in state_dict.items():
                    clean_key = key
                    # Remove common prefixes
                    for prefix in ['module.', '_orig_mod.', 'model.', '_forward_module.']:
                        if clean_key.startswith(prefix):
                            clean_key = clean_key[len(prefix):]
                            break
                    cleaned_state_dict[clean_key] = value
                
                # Convert state dict to target device and dtype
                def process_state_dict(state_dict, target_dtype, target_device):
                    """Recursively process state dict to handle nested structures"""
                    processed_dict = {}
                    for key, value in state_dict.items():
                        if isinstance(value, dict):
                            # Handle nested dictionaries
                            processed_dict[key] = process_state_dict(value, target_dtype, target_device)
                        elif isinstance(value, torch.Tensor):
                            # Handle tensors
                            if value.dtype != target_dtype:
                                value = value.to(dtype=target_dtype)
                            processed_dict[key] = value.to(target_device)
                        else:
                            # Handle other types (keep as is)
                            processed_dict[key] = value
                    return processed_dict
                
                target_dtype = torch.float16 if self.use_fp16 else torch.float32
                target_device = self.device
                
                if self.use_fp16:
                    print("🔧 Converting model weights to FP16...")
                else:
                    print("🔧 Converting model weights to FP32...")
                
                cleaned_state_dict = process_state_dict(cleaned_state_dict, target_dtype, target_device)
                
                # Load weights
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(cleaned_state_dict, strict=False)
                    if missing_keys:
                        print(f"⚠️ Missing keys ({len(missing_keys)}): {missing_keys[:5]}..." if len(missing_keys) > 5 else f"⚠️ Missing keys: {missing_keys}")
                    if unexpected_keys:
                        print(f"⚠️ Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"⚠️ Unexpected keys: {unexpected_keys}")
                    
                    print(f"✅ Model weights loaded successfully!")
                    
                except Exception as e:
                    print(f"⚠️ Error loading state dict: {e}")
                    print("🔧 Attempting to load with different strategy...")
                    
                    # Try to extract only tensor values if nested dicts exist
                    flat_state_dict = {}
                    def flatten_dict(d, parent_key=''):
                        items = []
                        for k, v in d.items():
                            new_key = f"{parent_key}.{k}" if parent_key else k
                            if isinstance(v, dict):
                                items.extend(flatten_dict(v, new_key).items())
                            elif isinstance(v, torch.Tensor):
                                items.append((new_key, v))
                        return dict(items)
                    
                    flat_state_dict = flatten_dict(cleaned_state_dict)
                    
                    # Convert flattened dict to target dtype/device
                    processed_flat_dict = process_state_dict(flat_state_dict, target_dtype, target_device)
                    
                    # Try loading with flattened dict
                    missing_keys, unexpected_keys = self.model.load_state_dict(processed_flat_dict, strict=False)
                    if missing_keys:
                        print(f"⚠️ Missing keys ({len(missing_keys)}): {missing_keys[:5]}..." if len(missing_keys) > 5 else f"⚠️ Missing keys: {missing_keys}")
                    if unexpected_keys:
                        print(f"⚠️ Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"⚠️ Unexpected keys: {unexpected_keys}")
                    
                    print(f"✅ Model weights loaded with fallback strategy!")
            
            # Set model precision AFTER loading weights and FORCE ALL PARAMETERS
            target_dtype = torch.float16 if self.use_fp16 else torch.float32
            
            print(f"🔧 Force converting ALL model parameters to {target_dtype}...")
            def force_model_dtype(module, dtype):
                """Recursively force all parameters and buffers to target dtype"""
                for param in module.parameters():
                    param.data = param.data.to(dtype=dtype)
                for buffer in module.buffers():
                    buffer.data = buffer.data.to(dtype=dtype)
                for child in module.children():
                    force_model_dtype(child, dtype)
            
            # Force all parameters to target dtype
            force_model_dtype(self.model, target_dtype)
            
            # Double check: verify all parameters have correct dtype
            dtype_check_passed = True
            for name, param in self.model.named_parameters():
                if param.dtype != target_dtype:
                    print(f"⚠️ Parameter {name} still has dtype {param.dtype}, forcing to {target_dtype}")
                    param.data = param.data.to(dtype=target_dtype)
                    dtype_check_passed = False
            
            for name, buffer in self.model.named_buffers():
                if buffer.dtype != target_dtype:
                    print(f"⚠️ Buffer {name} still has dtype {buffer.dtype}, forcing to {target_dtype}")
                    buffer.data = buffer.data.to(dtype=target_dtype)
                    dtype_check_passed = False
            
            if dtype_check_passed:
                print(f"✅ All model parameters verified to be {target_dtype}")
            else:
                print(f"⚠️ Some parameters needed additional dtype correction")
            
            self.model.eval()
            
            weight_status = "random initialization" if self.skip_weights else ("FP16" if self.use_fp16 else "FP32")
            print(f"✅ Model loaded successfully with {weight_status}!")
            
            # Display model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"🧠 Model Parameters: {total_params:,}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("🔍 Checkpoint structure debugging:")
            if not self.skip_weights:
                try:
                    # Try to load and inspect the checkpoint
                    debug_state_dict = torch.load(model_path, map_location='cpu')
                    print(f"   Root keys: {list(debug_state_dict.keys())}")
                    for key, value in debug_state_dict.items():
                        print(f"   {key}: {type(value)}")
                        if isinstance(value, dict):
                            print(f"      Sub-keys: {list(value.keys())[:5]}...")
                        elif isinstance(value, torch.Tensor):
                            print(f"      Tensor: {value.shape}, {value.dtype}")
                except Exception as debug_e:
                    print(f"   Debug failed: {debug_e}")
            else:
                print("   Skipped (weights loading disabled)")
            
            import traceback
            traceback.print_exc()
            raise
    
    def _ensure_tensor_dtype(self, tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """Ensure tensor has the correct dtype to match model"""
        target_dtype = torch.float16 if self.use_fp16 else torch.float32
        
        if tensor.dtype != target_dtype:
            # print(f"🔧 Converting {name} from {tensor.dtype} to {target_dtype}")
            tensor = tensor.to(dtype=target_dtype)
        
        return tensor
    
    def _safe_model_forward(self, model_func, input_tensor, operation_name="forward"):
        """Safely execute model forward with dtype checking"""
        target_dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # Ensure input has correct dtype
        input_tensor = self._ensure_tensor_dtype(input_tensor, f"{operation_name}_input")
        
        try:
            # Execute forward pass
            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                output = model_func(input_tensor)
            
            # Ensure output has correct dtype
            output = self._ensure_tensor_dtype(output, f"{operation_name}_output")
            return output
            
        except RuntimeError as e:
            if "dtype" in str(e).lower():
                print(f"⚠️ Dtype error in {operation_name}: {e}")
                print(f"🔧 Input dtype: {input_tensor.dtype}, Target: {target_dtype}")
                
                # Force convert input and try again
                input_tensor = input_tensor.to(dtype=target_dtype)
                
                # Also force all model parameters again
                for param in self.model.parameters():
                    param.data = param.data.to(dtype=target_dtype)
                for buffer in self.model.buffers():
                    buffer.data = buffer.data.to(dtype=target_dtype)
                
                print(f"🔧 Retrying {operation_name} after dtype correction...")
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    output = model_func(input_tensor)
                output = self._ensure_tensor_dtype(output, f"{operation_name}_output_retry")
                return output
            else:
                raise
    
    def load_audio(self, audio_path: str, max_duration: Optional[float] = None) -> torch.Tensor:
        """
        Load and preprocess audio file with proper dtype handling
        
        Args:
            audio_path: Path to audio file
            max_duration: Maximum duration to load (None for full file)
            
        Returns:
            Preprocessed audio tensor
        """
        try:
            print(f"🎵 Loading audio: {audio_path}")
            
            # Get audio info first
            info = torchaudio.info(audio_path)
            total_duration = info.num_frames / info.sample_rate
            
            if max_duration is not None:
                duration = min(total_duration, max_duration)
            else:
                duration = total_duration
            
            # Load audio using librosa for better format support
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False, duration=duration)
            
            # Ensure stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            elif audio.shape[0] > 2:
                audio = audio[:2]  # Take first 2 channels
            
            # Ensure exactly 2 channels
            if audio.shape[0] == 1:
                audio = np.repeat(audio, 2, axis=0)
            
            # Convert to tensor in FP32 first
            audio_tensor = torch.from_numpy(audio).float()
            
            # Add batch dimension
            audio_tensor = audio_tensor.unsqueeze(0)  # (1, 2, T)
            
            # Move to device
            audio_tensor = audio_tensor.to(self.device)
            
            # Convert to target dtype
            audio_tensor = self._ensure_tensor_dtype(audio_tensor, "audio")
            
            print(f"✅ Audio loaded: {audio_tensor.shape} ({duration:.2f}s) dtype={audio_tensor.dtype}")
            return audio_tensor
            
        except Exception as e:
            print(f"❌ Failed to load audio: {e}")
            raise
    
    def _create_fade_window(self, length: int, fade_type: str = "hann") -> torch.Tensor:
        """Create fade in/out window for smooth overlap with correct dtype"""
        if fade_type == "hann":
            window = torch.hann_window(length * 2, device=self.device)
            window = window[:length]  # Take first half for fade in
        elif fade_type == "linear":
            window = torch.linspace(0, 1, length, device=self.device)
        else:
            window = torch.ones(length, device=self.device)
        
        # Ensure correct dtype
        return self._ensure_tensor_dtype(window, "fade_window")
    
    def _split_into_chunks(self, audio: torch.Tensor) -> List[Tuple[torch.Tensor, int]]:
        """
        Split audio into overlapping chunks with proper dtype
        
        Args:
            audio: Input audio tensor (B, C, T)
            
        Returns:
            List of (chunk_tensor, start_idx) tuples
        """
        B, C, T = audio.shape
        chunks = []
        
        # Calculate number of chunks needed
        if T <= self.chunk_samples:
            # Audio is shorter than chunk size, pad it
            padded = F.pad(audio, (0, self.chunk_samples - T), mode='reflect')
            padded = self._ensure_tensor_dtype(padded, "padded_chunk")
            chunks.append((padded, 0))
        else:
            # Split into overlapping chunks
            start_idx = 0
            while start_idx < T:
                end_idx = min(start_idx + self.chunk_samples, T)
                chunk = audio[:, :, start_idx:end_idx]
                
                # Pad last chunk if needed
                if chunk.shape[-1] < self.chunk_samples:
                    pad_length = self.chunk_samples - chunk.shape[-1]
                    chunk = F.pad(chunk, (0, pad_length), mode='reflect')
                
                chunk = self._ensure_tensor_dtype(chunk, f"chunk_{start_idx}")
                chunks.append((chunk, start_idx))
                
                # Move to next chunk
                if end_idx >= T:
                    break
                start_idx += self.hop_samples
        
        return chunks
    
    def _merge_chunks(self, chunk_results: List[Tuple[torch.Tensor, int]], original_length: int) -> torch.Tensor:
        """
        Merge overlapping chunks back into full audio with proper dtype
        
        Args:
            chunk_results: List of (reconstructed_chunk, start_idx) tuples
            original_length: Original audio length
            
        Returns:
            Merged audio tensor
        """
        if not chunk_results:
            target_dtype = torch.float16 if self.use_fp16 else torch.float32
            return torch.zeros(1, 2, original_length, device=self.device, dtype=target_dtype)
        
        # Initialize output with correct dtype
        B, C = chunk_results[0][0].shape[:2]
        target_dtype = chunk_results[0][0].dtype
        merged = torch.zeros(B, C, original_length, device=self.device, dtype=target_dtype)
        weight_sum = torch.zeros(original_length, device=self.device, dtype=target_dtype)
        
        # Create fade windows for overlap
        if self.overlap_samples > 0:
            fade_in = self._create_fade_window(self.overlap_samples)
            fade_out = 1.0 - fade_in
        
        for chunk, start_idx in chunk_results:
            end_idx = min(start_idx + self.chunk_samples, original_length)
            chunk_length = end_idx - start_idx
            
            # Get the relevant part of the chunk
            chunk_data = chunk[:, :, :chunk_length]
            chunk_data = self._ensure_tensor_dtype(chunk_data, "chunk_data")
            
            # Create weight for this chunk
            weight = torch.ones(chunk_length, device=self.device, dtype=target_dtype)
            
            # Apply fade in/out for overlaps
            if start_idx > 0 and self.overlap_samples > 0:
                # Fade in at the beginning (overlap with previous chunk)
                overlap_length = min(self.overlap_samples, chunk_length)
                if overlap_length > 0:
                    weight[:overlap_length] = fade_in[:overlap_length]
                    chunk_data[:, :, :overlap_length] *= fade_in[:overlap_length].unsqueeze(0).unsqueeze(0)
            
            if end_idx < original_length and chunk_length >= self.overlap_samples and self.overlap_samples > 0:
                # Fade out at the end (overlap with next chunk)
                overlap_length = min(self.overlap_samples, chunk_length)
                if overlap_length > 0:
                    weight[-overlap_length:] = fade_out[:overlap_length]
                    chunk_data[:, :, -overlap_length:] *= fade_out[:overlap_length].unsqueeze(0).unsqueeze(0)
            
            # Add to merged result
            merged[:, :, start_idx:end_idx] += chunk_data
            weight_sum[start_idx:end_idx] += weight
        
        # Normalize by weights to handle overlaps
        weight_sum = torch.clamp(weight_sum, min=1e-8)
        merged = merged / weight_sum.unsqueeze(0).unsqueeze(0)
        
        return merged
    
    def encode_chunked(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """
        Encode audio using chunking with proper dtype handling
        
        Args:
            audio: Input audio tensor (B, C, T)
            
        Returns:
            List of latent tensors
        """
        chunks = self._split_into_chunks(audio)
        latents = []
        
        print(f"🔄 Encoding {len(chunks)} chunks...")
        
        with torch.no_grad():
            for i, (chunk, start_idx) in enumerate(tqdm(chunks, desc="Encoding")):
                try:
                    # Use safe forward pass
                    latent = self._safe_model_forward(self.model.encode, chunk, f"encode_chunk_{i}")
                    latents.append(latent)
                    
                except Exception as e:
                    print(f"❌ Error encoding chunk {i}: {e}")
                    # Create dummy latent to continue
                    target_dtype = torch.float16 if self.use_fp16 else torch.float32
                    dummy_latent = torch.zeros(chunk.shape[0], 16, chunk.shape[2]//512, chunk.shape[2]//512, 
                                             device=self.device, dtype=target_dtype)
                    latents.append(dummy_latent)
        
        return latents
    
    def decode_chunked(self, latents: List[torch.Tensor], original_length: int) -> torch.Tensor:
        """
        Decode latents using chunking with proper dtype handling
        
        Args:
            latents: List of latent tensors
            original_length: Original audio length
            
        Returns:
            Reconstructed audio tensor
        """
        print(f"🔄 Decoding {len(latents)} chunks...")
        
        chunk_results = []
        
        with torch.no_grad():
            for i, latent in enumerate(tqdm(latents, desc="Decoding")):
                try:
                    # Use safe forward pass
                    reconstructed_chunk = self._safe_model_forward(self.model.decode, latent, f"decode_chunk_{i}")
                    
                    start_idx = i * self.hop_samples
                    chunk_results.append((reconstructed_chunk, start_idx))
                    
                except Exception as e:
                    print(f"❌ Error decoding chunk {i}: {e}")
                    # Create dummy audio to continue
                    target_dtype = torch.float16 if self.use_fp16 else torch.float32
                    dummy_audio = torch.zeros(latent.shape[0], 2, self.chunk_samples, 
                                            device=self.device, dtype=target_dtype)
                    start_idx = i * self.hop_samples
                    chunk_results.append((dummy_audio, start_idx))
        
        # Merge chunks
        print("🔗 Merging chunks with overlap...")
        merged = self._merge_chunks(chunk_results, original_length)
        
        return merged
    
    def reconstruct_chunked(self, audio: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Full reconstruction pipeline with chunking and proper dtype handling
        
        Args:
            audio: Input audio tensor
            
        Returns:
            (latents, reconstructed_audio)
        """
        original_length = audio.shape[-1]
        
        # Ensure input has correct dtype
        audio = self._ensure_tensor_dtype(audio, "reconstruct_input")
        
        # Encode in chunks
        latents = self.encode_chunked(audio)
        
        # Decode in chunks
        reconstructed = self.decode_chunked(latents, original_length)
        
        return latents, reconstructed
    
    def calculate_compression_metrics(self, original: torch.Tensor, latents: List[torch.Tensor]) -> Dict[str, float]:
        """Calculate compression metrics for chunked processing"""
        # Original size
        original_elements = original.numel()
        original_bytes = original_elements * (2 if original.dtype == torch.float16 else 4)
        
        # Total latent size
        latent_elements = sum(latent.numel() for latent in latents)
        latent_bytes = latent_elements * (2 if latents[0].dtype == torch.float16 else 4)
        
        # Compression ratio
        compression_ratio = original_elements / latent_elements if latent_elements > 0 else 0
        size_reduction = (1 - latent_bytes / original_bytes) * 100 if original_bytes > 0 else 0
        
        # Per-chunk statistics
        avg_latent_elements = latent_elements / len(latents) if latents else 0
        chunk_compression_ratio = (original.shape[-1] / len(latents) * 2) / avg_latent_elements if avg_latent_elements > 0 else 0
        
        return {
            'original_elements': original_elements,
            'original_size_mb': original_bytes / (1024 * 1024),
            'latent_elements': latent_elements,
            'latent_size_mb': latent_bytes / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'size_reduction_percent': size_reduction,
            'num_chunks': len(latents),
            'avg_chunk_compression_ratio': chunk_compression_ratio,
            'chunk_duration': self.chunk_duration,
            'overlap_ratio': self.overlap_ratio
        }
    
    def calculate_audio_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
        """Calculate audio quality metrics with proper dtype handling"""
        # Move to CPU and convert to float32 for calculations
        orig_cpu = original.cpu().float()
        recon_cpu = reconstructed.cpu().float()
        
        # Ensure same length
        min_len = min(orig_cpu.shape[-1], recon_cpu.shape[-1])
        orig_cpu = orig_cpu[..., :min_len]
        recon_cpu = recon_cpu[..., :min_len]
        
        # MSE
        mse = F.mse_loss(recon_cpu, orig_cpu).item()
        
        # SNR (Signal-to-Noise Ratio)
        signal_power = torch.mean(orig_cpu ** 2).item()
        noise_power = torch.mean((orig_cpu - recon_cpu) ** 2).item()
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        # Spectral convergence (use smaller FFT for long audio)
        n_fft = min(1024, min_len // 4)
        hop_length = n_fft // 4
        
        try:
            orig_stft = torch.stft(orig_cpu.reshape(-1), n_fft=n_fft, hop_length=hop_length, return_complex=True)
            recon_stft = torch.stft(recon_cpu.reshape(-1), n_fft=n_fft, hop_length=hop_length, return_complex=True)
            
            orig_mag = torch.abs(orig_stft)
            recon_mag = torch.abs(recon_stft)
            
            spectral_conv = torch.norm(orig_mag - recon_mag) / torch.norm(orig_mag)
            spectral_conv = spectral_conv.item()
        except:
            spectral_conv = float('nan')
        
        # PESQ-like measure (simplified)
        correlation = F.cosine_similarity(orig_cpu.reshape(-1), recon_cpu.reshape(-1), dim=0).item()
        
        return {
            'mse': mse,
            'snr_db': snr,
            'spectral_convergence': spectral_conv,
            'correlation': correlation,
            'duration_seconds': min_len / self.sample_rate
        }
    
    def save_audio(self, audio: torch.Tensor, output_path: str, sample_rate: int = 44100):
        """Save audio tensor to file"""
        # Move to CPU and convert to float32
        audio_cpu = audio.cpu().float()
        
        # Remove batch dimension if present
        if audio_cpu.dim() == 3:
            audio_cpu = audio_cpu.squeeze(0)
        
        # Ensure shape is (C, T)
        if audio_cpu.dim() == 1:
            audio_cpu = audio_cpu.unsqueeze(0)
        
        # Normalize to [-1, 1]
        audio_cpu = torch.tanh(audio_cpu)
        
        # Save using torchaudio
        torchaudio.save(output_path, audio_cpu, sample_rate)
        print(f"💾 Audio saved: {output_path} ({audio_cpu.shape[-1]/sample_rate:.2f}s)")
    
    def save_latents(self, latents: List[torch.Tensor], output_path: str):
        """Save latent tensors"""
        # Convert to CPU for saving
        latents_cpu = [latent.cpu() for latent in latents]
        
        # Save as dictionary with metadata
        save_data = {
            'latents': latents_cpu,
            'chunk_duration': self.chunk_duration,
            'overlap_ratio': self.overlap_ratio,
            'num_chunks': len(latents_cpu),
            'chunk_samples': self.chunk_samples,
            'hop_samples': self.hop_samples,
            'sample_rate': self.sample_rate,
            'use_fp16': self.use_fp16
        }
        
        torch.save(save_data, output_path)
        print(f"💾 Latents saved: {output_path} ({len(latents)} chunks)")
    
    def load_latents(self, latent_path: str) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """Load latent tensors with proper dtype handling"""
        save_data = torch.load(latent_path, map_location='cpu')
        
        if isinstance(save_data, dict) and 'latents' in save_data:
            latents = save_data['latents']
            metadata = {k: v for k, v in save_data.items() if k != 'latents'}
        else:
            # Legacy format - just a list of tensors
            latents = save_data
            metadata = {}
        
        # Convert to target device and dtype
        processed_latents = []
        for latent in latents:
            latent = latent.to(self.device)
            latent = self._ensure_tensor_dtype(latent, "loaded_latent")
            processed_latents.append(latent)
        
        return processed_latents, metadata
    
    def process_audio_file(self, input_path: str, output_dir: str, max_duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Process a single audio file with chunking support and proper dtype handling
        
        Args:
            input_path: Input audio file path
            output_dir: Output directory
            max_duration: Maximum duration to process (None for full file)
            
        Returns:
            Processing results and metrics
        """
        start_time = time.time()
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get file stem for naming
        file_stem = Path(input_path).stem
        
        print(f"\n{'='*80}")
        print(f"🎵 Processing: {input_path}")
        print(f"🔧 Chunk Duration: {self.chunk_duration}s, Overlap: {self.overlap_ratio}")
        print(f"🔧 Precision: {'FP16' if self.use_fp16 else 'FP32'}")
        print(f"{'='*80}")
        
        try:
            # Load audio
            original_audio = self.load_audio(input_path, max_duration)
            original_duration = original_audio.shape[-1] / self.sample_rate
            
            # Calculate expected number of chunks
            if original_audio.shape[-1] <= self.chunk_samples:
                expected_chunks = 1
            else:
                expected_chunks = math.ceil((original_audio.shape[-1] - self.chunk_samples) / self.hop_samples) + 1
            
            print(f"📊 Audio Info: {original_duration:.2f}s, Expected chunks: {expected_chunks}")
            
            # Reconstruct with chunking
            print("🔄 Processing with chunking...")
            latents, reconstructed = self.reconstruct_chunked(original_audio)
            
            # Check if there were any errors during processing
            error_count = 0
            for i, latent in enumerate(latents):
                if torch.all(latent == 0):  # Check if it's a dummy latent
                    error_count += 1
            
            if error_count > 0:
                print(f"⚠️ Warning: {error_count}/{len(latents)} chunks failed during processing")
                results_has_errors = True
            else:
                print(f"✅ All {len(latents)} chunks processed successfully")
                results_has_errors = False
            
            # Calculate metrics
            print("📊 Calculating metrics...")
            compression_metrics = self.calculate_compression_metrics(original_audio, latents)
            audio_metrics = self.calculate_audio_metrics(original_audio, reconstructed)
            
            # Save results
            print("💾 Saving results...")
            
            # Save original (for comparison)
            original_path = output_dir / f"{file_stem}_original.wav"
            self.save_audio(original_audio, str(original_path))
            
            # Save reconstructed
            reconstructed_path = output_dir / f"{file_stem}_reconstructed.wav"
            self.save_audio(reconstructed, str(reconstructed_path))
            
            # Save latent representations
            latent_path = output_dir / f"{file_stem}_latents.pt"
            self.save_latents(latents, str(latent_path))
            
            # Processing time
            processing_time = time.time() - start_time
            
            # Compile results
            results = {
                'input_file': input_path,
                'processing_time': processing_time,
                'original_duration': original_duration,
                'has_errors': results_has_errors,
                'error_count': error_count,
                'chunking_info': {
                    'chunk_duration': self.chunk_duration,
                    'overlap_ratio': self.overlap_ratio,
                    'num_chunks': len(latents),
                    'expected_chunks': expected_chunks,
                    'use_fp16': self.use_fp16
                },
                'compression_metrics': compression_metrics,
                'audio_metrics': audio_metrics,
                'output_files': {
                    'original': str(original_path),
                    'reconstructed': str(reconstructed_path),
                    'latents': str(latent_path)
                }
            }
            
            # Print summary
            print(f"\n📈 RESULTS SUMMARY")
            print(f"{'='*50}")
            print(f"⏱️  Processing Time: {processing_time:.2f}s")
            print(f"🎵 Original Duration: {original_duration:.2f}s")
            print(f"🧩 Chunks Processed: {len(latents)}")
            print(f"🗜️  Compression Ratio: {compression_metrics['compression_ratio']:.1f}:1")
            print(f"📉 Size Reduction: {compression_metrics['size_reduction_percent']:.1f}%")
            print(f"📊 Original Size: {compression_metrics['original_size_mb']:.2f} MB")
            print(f"📊 Latent Size: {compression_metrics['latent_size_mb']:.4f} MB")
            print(f"🎯 SNR: {audio_metrics['snr_db']:.2f} dB")
            print(f"🔊 MSE: {audio_metrics['mse']:.6f}")
            print(f"🌊 Spectral Convergence: {audio_metrics['spectral_convergence']:.4f}")
            print(f"📈 Correlation: {audio_metrics['correlation']:.4f}")
            print(f"🔧 Precision Used: {'FP16' if self.use_fp16 else 'FP32'}")
            
            # Save detailed results
            results_path = output_dir / f"{file_stem}_results.json"
            with open(results_path, 'w') as f:
                # Convert numpy types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                # Clean results for JSON
                clean_results = json.loads(json.dumps(results, default=convert_numpy))
                json.dump(clean_results, f, indent=2)
            
            print(f"📄 Detailed results saved: {results_path}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error processing audio file: {e}")
            import traceback
            traceback.print_exc()
            raise


def find_audio_files(directory: str) -> List[str]:
    """Find audio files in directory"""
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.aac']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(directory, ext)))
        audio_files.extend(glob.glob(os.path.join(directory, ext.upper())))
    
    return sorted(audio_files)


def main():
    """Main inference function with debugging options"""
    parser = argparse.ArgumentParser(description='DCAE Model Inference with Chunking Support - FIXED VERSION')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint directory')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--fp32', action='store_true',
                       help='Use FP32 precision instead of FP16 (RECOMMENDED for debugging)')
    parser.add_argument('--skip_weights', action='store_true',
                       help='Skip loading weights (use random initialization for testing)')
    
    # Input/Output
    parser.add_argument('--input_dir', type=str, default='test_audio',
                       help='Directory containing test audio files')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--file_index', type=int, default=0,
                       help='Index of audio file to process (0 for first file)')
    
    # Processing
    parser.add_argument('--max_duration', type=float, default=None,
                       help='Maximum duration in seconds to process (None for full file)')
    parser.add_argument('--chunk_duration', type=float, default=2.0,
                       help='Duration of each chunk in seconds')
    parser.add_argument('--overlap_ratio', type=float, default=0.25,
                       help='Overlap ratio between chunks (0.0 to 0.5)')
    
    args = parser.parse_args()
    
    print("🚀 DCAE Inference with Chunking Starting... (FIXED VERSION)")
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📁 Input Directory: {args.input_dir}")
    print(f"📁 Output Directory: {args.output_dir}")
    print(f"🔧 Chunk Duration: {args.chunk_duration}s")
    print(f"🔄 Overlap Ratio: {args.overlap_ratio}")
    print(f"🔧 Precision: {'FP32' if args.fp32 else 'FP16'}")
    if args.skip_weights:
        print("⚠️ WARNING: Skipping weight loading - using random initialization!")
    
    # Recommend FP32 for this problematic model
    if not args.fp32:
        print("💡 RECOMMENDATION: This model has dtype issues. Consider using --fp32 flag")
    
    try:
        # Initialize inference engine with chunking
        inference_engine = DCAEInference(
            args.checkpoint, 
            args.device,
            chunk_duration=args.chunk_duration,
            overlap_ratio=args.overlap_ratio,
            use_fp16=not args.fp32,  # Use FP32 if --fp32 flag is set
            skip_weights=args.skip_weights
        )
        
        # Find audio files
        audio_files = find_audio_files(args.input_dir)
        
        if not audio_files:
            print(f"❌ No audio files found in {args.input_dir}")
            return
        
        print(f"🎵 Found {len(audio_files)} audio files:")
        for i, file in enumerate(audio_files):
            print(f"   {i}: {os.path.basename(file)}")
        
        # Select file to process
        if args.file_index >= len(audio_files):
            print(f"❌ File index {args.file_index} out of range (0-{len(audio_files)-1})")
            return
        
        selected_file = audio_files[args.file_index]
        print(f"\n🎯 Processing file {args.file_index}: {os.path.basename(selected_file)}")
        
        # Process the selected file
        results = inference_engine.process_audio_file(
            selected_file, 
            args.output_dir, 
            args.max_duration
        )
        
        print(f"\n🎉 Inference completed successfully!")
        print(f"📁 Results saved in: {args.output_dir}")
        print(f"🧩 Processed {results['chunking_info']['num_chunks']} chunks")
        print(f"⏱️ Total processing time: {results['processing_time']:.2f}s")
        print(f"🔧 Precision used: {'FP32' if args.fp32 else 'FP16'}")
        
        # Check for errors in results
        if results.get('has_errors', False):
            print("⚠️ Some chunks failed during processing - check output quality")
            print(f"   Failed chunks: {results.get('error_count', 0)}/{results['chunking_info']['num_chunks']}")
            print("💡 Consider using --fp32 flag for better stability")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        print("💡 Debugging suggestions:")
        print("   - Try running with --fp32 flag (STRONGLY RECOMMENDED)")
        print("   - Try running with --skip_weights flag to test pipeline")
        print("   - Check if checkpoint path is correct")
        print("   - The model appears to have mixed precision issues")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()