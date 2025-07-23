import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union

# Individual Components  
from .generator import LyroGenerator, GeneratorConfig
from .encoders import UnifiedTextEncoder
from .sampling import FlowMatchingSampler, ReferenceEncoder
from .losses import CombinedLoss, FlowMatchingLoss, AlignmentLoss
from .audio_encoder import LyroAudioEncoder, create_audio_encoder


class MultimodalLyroSystem(nn.Module):
    def __init__(self, generator_config: GeneratorConfig = None, use_audio_encoder: bool = True):
        super().__init__()
        
        if generator_config is None:
            generator_config = GeneratorConfig()
        
        self.generator = LyroGenerator(generator_config)
        self.text_encoder = UnifiedTextEncoder(
            embed_dim=generator_config.d_model,
            hidden_dim=generator_config.d_model
        )
        
        # 오디오 인코더 (선택적)
        self.use_audio_encoder = use_audio_encoder
        if use_audio_encoder:
            try:
                self.audio_encoder = create_audio_encoder()
                print("✅ Audio Encoder (DCAE) integrated into system")
            except Exception as e:
                print(f"⚠️  Audio Encoder initialization failed: {e}")
                self.use_audio_encoder = False
                self.audio_encoder = None
        else:
            self.audio_encoder = None
    
    def forward(
        self,
        lyrics: Optional[List[str]] = None,
        style: Optional[List[str]] = None,
        target_latents: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None  # REPA + DCAE용
    ) -> Dict[str, Any]:
        # Text encoding
        if lyrics is not None or style is not None:
            text_embed = self.text_encoder(lyrics=lyrics, style=style)
        else:
            device = next(self.parameters()).device
            batch_size = target_latents.shape[0] if target_latents is not None else 1
            text_embed = torch.zeros(batch_size, self.generator.config.d_model, device=device)
        
        results = {'text_embed': text_embed}
        
        # Training: compute unified loss (Flow Matching + REPA)
        if target_latents is not None:
            loss_dict = self.generator.training_loss(
                latents=target_latents,
                text_embed=text_embed,
                training_audio=audio  # REPA를 위해 전달
            )
            results.update(loss_dict)
        
        return results
    
    def encode_audio(self, audio: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        """오디오를 latent로 인코딩 (DCAE 사용)"""
        if not self.use_audio_encoder or self.audio_encoder is None:
            raise ValueError("Audio encoder not available. Set use_audio_encoder=True")
        
        latents, _ = self.audio_encoder.encode_audio(audio, sample_rate)
        return latents
    
    def decode_latents(self, latents: torch.Tensor, sample_rate: Optional[int] = None) -> Tuple[int, list]:
        """Latent를 오디오로 디코딩 (DCAE 사용)"""
        if not self.use_audio_encoder or self.audio_encoder is None:
            raise ValueError("Audio encoder not available. Set use_audio_encoder=True")
        
        return self.audio_encoder.decode_latents(latents, sample_rate)
    
    def generate_music(
        self,
        lyrics: Optional[List[str]] = None,
        style: Optional[List[str]] = None,
        num_steps: int = 50,
        cfg_scale: float = 7.5,
        return_audio: bool = False,
        sample_rate: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int, list]]:
        """
        음악 생성
        
        Args:
            lyrics: 가사
            style: 스타일
            num_steps: diffusion steps
            cfg_scale: classifier-free guidance scale
            return_audio: 오디오로 변환해서 반환 여부
            sample_rate: 출력 샘플링 레이트
            
        Returns:
            latents 또는 (latents, sample_rate, audio_list)
        """
        self.eval()
        with torch.no_grad():
            # Text encoding
            if lyrics is not None or style is not None:
                text_embed = self.text_encoder(lyrics=lyrics, style=style)
            else:
                device = next(self.parameters()).device
                text_embed = torch.zeros(1, self.generator.config.d_model, device=device)
            
            # Latent generation
            shape = (text_embed.shape[0], self.generator.config.latent_channels, 
                    self.generator.config.latent_time_steps)
            
            latents = self.generator.generate(shape, text_embed, None, num_steps, cfg_scale)
            
            if return_audio and self.use_audio_encoder:
                # Latent → Audio
                sr_out, audio_list = self.decode_latents(latents, sample_rate)
                return latents, sr_out, audio_list
            else:
                return latents
    
    def save_pretrained(self, save_directory: str):
        import os
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.generator.state_dict(), f"{save_directory}/generator.pth")
        torch.save(self.text_encoder.state_dict(), f"{save_directory}/text_encoder.pth")
        
        # 오디오 인코더는 DCAE 체크포인트를 별도로 관리하므로 저장하지 않음
        if self.use_audio_encoder:
            print("💾 Audio encoder (DCAE) checkpoints managed separately")
    
    def load_pretrained(self, load_directory: str):
        import os
        if os.path.exists(f"{load_directory}/generator.pth"):
            self.generator.load_state_dict(torch.load(f"{load_directory}/generator.pth"))
        if os.path.exists(f"{load_directory}/text_encoder.pth"):
            self.text_encoder.load_state_dict(torch.load(f"{load_directory}/text_encoder.pth"))
        
        print("📁 Audio encoder (DCAE) loads from dcae_vocoder/ directory")

__all__ = [
    'MultimodalLyroSystem',
    'LyroGenerator',
    'GeneratorConfig', 
    'UnifiedTextEncoder',
    'LyroAudioEncoder',
    'create_audio_encoder',
    'FlowMatchingSampler',
    'ReferenceEncoder',
    'CombinedLoss',
    'FlowMatchingLoss',
    'AlignmentLoss',
]

def create_multimodal_system(generator_config: GeneratorConfig = None) -> MultimodalLyroSystem:
    if generator_config is None:
        generator_config = GeneratorConfig()
    return MultimodalLyroSystem(generator_config)

def get_model_info():
    return {
        'version': '2.0 (Simplified)',
        'total_parameters': '~400M',
        'components': {
            'generator': {
                'parameters': '350M',
                'architecture': 'SSM + Flow Matching',
                'config': 'd_model=1024, n_layers=16, n_heads=16'
            },
            'text_encoder': {
                'parameters': '~50M',
                'architecture': 'Unified (Lyrics + Style)',
                'features': ['Cross-attention', 'Multi-tag support']
            }
        },
        'supported_tasks': ['SONG', 'INST'],
        'output': 'Latent vectors (16 channels, 128 timesteps)'
    }

def estimate_model_memory(batch_size: int = 4):
    memory_estimate = {
        'generator': 0.8,        # GB (350M params in fp16)
        'text_encoder': 0.1,     # GB (50M params)
        'training_overhead': 1.5, # GB (gradients, optimizer states)
        'batch_data': batch_size * 0.2,  # GB per batch
    }
    
    total_memory = sum(memory_estimate.values())
    
    return {
        'breakdown': memory_estimate,
        'total_estimated': f"{total_memory:.1f} GB",
        'recommended_gpu': "RTX 3080 (12GB)" if total_memory <= 10 else "RTX 4090 (24GB)",
        'batch_size': batch_size
    }

def get_system_capabilities():
    return {
        'text_processing': {
            'lyric_sections': ['verse', 'chorus', 'bridge', 'intro', 'outro'],
            'instruments': ['piano', 'guitar', 'synth', 'strings', 'drums'],
            'moods': ['emotional', 'energetic', 'calm', 'dark', 'bright'],
            'multi_tag_format': '[section:instrument:mood]',
            'example': '[verse 1:piano:emotional] lyrics content'
        },
        'generation_modes': {
            'SONG': 'Lyrics + style → music',
            'INST': 'Style only → instrumental'
        }
    }