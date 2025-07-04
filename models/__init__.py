"""
LYRO Models Package (수정됨 - Generator 전용)
"""

from .generator import LyroGenerator, GeneratorConfig, create_lyro_generator
from .ssm_flow import SSMFlowGenerator, create_ssm_flow_generator
from .encoders import CaptionEncoder, LyricsEncoder, ReferenceEncoder
from .losses import (
    FlowMatchingLoss, 
    REPALoss, 
    ReconstructionLoss, 
    PerceptualLoss,
    CombinedLoss
)
from .sampling import FlowMatchingSampler

__all__ = [
    # Generator
    'LyroGenerator',
    'GeneratorConfig',
    'create_lyro_generator',
    'SSMFlowGenerator',
    'create_ssm_flow_generator',
    
    # Encoders
    'CaptionEncoder',
    'LyricsEncoder', 
    'ReferenceEncoder',
    
    # Losses (Generator용)
    'FlowMatchingLoss',
    'REPALoss',
    'ReconstructionLoss',
    'PerceptualLoss',
    'CombinedLoss',
    
    # Sampling
    'FlowMatchingSampler',
    
    # Factory functions
    'create_lyro_models',
]


def create_lyro_models(config):
    """
    LYRO 모델 스위트 생성 (수정됨: Generator 전용)
    
    Args:
        config: 모델 설정
        
    Returns:
        dict: 모델 딕셔너리
    """
    models = {}
    
    # Generator (SSM + Flow Matching, 훈련 대상)
    models['generator'] = create_ssm_flow_generator(
        latent_channels=config.generator.latent_channels,
        latent_size=config.generator.latent_time_steps,
        d_model=config.generator.d_model,
        n_layers=config.generator.n_layers,
        d_state=config.generator.d_state
    )
    
    # 조건 인코더들
    models['caption_encoder'] = CaptionEncoder(
        model_name=config.encoder.caption_model,
        output_dim=config.encoder.caption_embed_dim
    )
    
    models['lyrics_encoder'] = LyricsEncoder(
        vocab_size=config.encoder.lyrics_vocab_size,
        embed_dim=config.encoder.lyrics_embed_dim,
        max_length=config.encoder.max_lyrics_length
    )
    
    models['reference_encoder'] = ReferenceEncoder(
        input_channels=config.generator.latent_channels,
        output_dim=config.encoder.reference_embed_dim
    )
    
    return models


def create_loss_functions(config):
    """
    Generator용 손실 함수 생성
    
    Args:
        config: 손실 설정
        
    Returns:
        CombinedLoss: 결합 손실 함수
    """
    
    # 개별 손실 컴포넌트들
    flow_loss = FlowMatchingLoss(
        beta_schedule="cosine",
        num_timesteps=1000
    )
    
    repa_loss = REPALoss(
        hubert_model=config.loss.hubert_model,
        layer_weights=config.loss.hubert_layers
    )
    
    recon_loss = ReconstructionLoss(
        l1_weight=1.0,
        l2_weight=0.1
    )
    
    perceptual_loss = PerceptualLoss(
        mel_weight=config.loss.mel_weight,
        stft_weight=config.loss.stft_weight
    )
    
    # 결합 손실
    combined_loss = CombinedLoss(
        flow_loss=flow_loss,
        repa_loss=repa_loss,
        recon_loss=recon_loss,
        perceptual_loss=perceptual_loss,
        weights={
            'flow': config.loss.flow_matching_weight,
            'repa': config.loss.repa_weight,
            'recon': config.loss.reconstruction_weight,
            'perceptual': config.loss.perceptual_weight
        }
    )
    
    return combined_loss


def get_model_info():
    """모델 아키텍처 정보 (수정됨 - Generator 전용)"""
    return {
        'ssm_generator': {
            'parameters': '~1.5B', 
            'architecture': 'S6 + Flow Matching',
            'conditions': ['lyrics', 'captions', 'reference'],
            'trainable': True,
            'output_format': 'latent_vectors'
        },
        'encoders': {
            'caption': 'Pretrained Text Encoder',
            'lyrics': 'Custom Tokenizer + Embedding',
            'reference': 'Latent-based'
        },
        'total_trainable_parameters': '~1.5B',
        'pipeline': 'Text → Generator → Latent Vectors',
        'audio_synthesis': 'External Vocoder Required'
    }


def estimate_model_memory(config):
    """모델 메모리 사용량 추정 (수정됨 - Generator 전용)"""
    memory_estimate = {
        'generator': 3.0,     # GB (1.5B params in fp16)
        'encoders': 0.3,      # GB
        'training_overhead': 2.0,  # GB (gradients, optimizer states)
        'batch_data': config.generator.batch_size * 0.2,  # GB per batch (latents only)
    }
    
    total_memory = sum(memory_estimate.values())
    
    return {
        'breakdown': memory_estimate,
        'total_estimated': f"{total_memory:.1f} GB",
        'recommended_gpu': "RTX 4090 (24GB)" if total_memory <= 24 else "H100 (80GB)",
        'note': 'Estimates for Generator-only training with latent vectors',
        'audio_synthesis': 'Requires external vocoder for audio generation'
    }