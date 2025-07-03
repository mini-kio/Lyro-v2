"""
LYRO Models Package (프리트레인된 DCAE + Vocoder + Generator 훈련)
"""

from .dcae import PretrainedDCAE, AdvancedVocoder, create_dcae_model, create_vocoder_model
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

__all__ = [
    # Pretrained DCAE + Vocoder
    'PretrainedDCAE',
    'AdvancedVocoder',
    'create_dcae_model',
    'create_vocoder_model',
    
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
    
    # Factory functions
    'create_lyro_models',
]


def create_lyro_models(config):
    """
    LYRO 모델 스위트 생성 (프리트레인된 DCAE + Vocoder + 훈련할 Generator)
    
    Args:
        config: 모델 설정
        
    Returns:
        dict: 모델 딕셔너리
    """
    models = {}
    
    # 프리트레인된 DCAE (훈련하지 않음)
    models['dcae'] = create_dcae_model(
        model_name=config.dcae.model_name,
        subfolder=config.dcae.subfolder,
        cache_dir=config.dcae.cache_dir,
        sample_rate=config.dcae.sample_rate
    )
    
    # 프리트레인된 Vocoder (훈련하지 않음)
    models['vocoder'] = create_vocoder_model(
        model_name=config.dcae.model_name,
        cache_dir=config.dcae.cache_dir,
        sample_rate=config.dcae.sample_rate
    )
    
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
        dcae_model=models['dcae']
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
    """모델 아키텍처 정보"""
    return {
        'dcae': {
            'type': 'Pretrained ACE-Step',
            'model_name': 'ACE-Step/ACE-Step-v1-3.5B',
            'latent_channels': 16,
            'compression_ratio': '~50:1',
            'trainable': False
        },
        'vocoder': {
            'type': 'Pretrained ACE-Step Vocoder',
            'model_name': 'ACE-Step/ACE-Step-v1-3.5B',
            'sample_rate': 44100,
            'trainable': False
        },
        'ssm_generator': {
            'parameters': '~1.5B', 
            'architecture': 'S6 + Flow Matching',
            'conditions': ['lyrics', 'captions', 'reference'],
            'trainable': True
        },
        'encoders': {
            'caption': 'Pretrained Text Encoder',
            'lyrics': 'Custom Tokenizer + Embedding',
            'reference': 'DCAE-based'
        },
        'pipeline': 'audio->mel->dcae_latent->generator->dcae_mel->vocoder->audio',
        'total_trainable_parameters': '~1.5B'
    }