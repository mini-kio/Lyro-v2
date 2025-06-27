# lyro/models/__init__.py
"""
LYRO Models Package
Unified model factory and imports
"""

from .dcae import DCAE, create_dcae
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
    # Models
    'DCAE',
    'SSMFlowGenerator',
    
    # Encoders
    'CaptionEncoder',
    'LyricsEncoder', 
    'ReferenceEncoder',
    
    # Losses
    'FlowMatchingLoss',
    'REPALoss',
    'ReconstructionLoss',
    'PerceptualLoss',
    'CombinedLoss',
    
    # Factory functions
    'create_dcae',
    'create_ssm_flow_generator',
    'create_lyro_models',
]


def create_lyro_models(config):
    """
    Create complete LYRO model suite
    
    Args:
        config: Model configuration
        
    Returns:
        dict: Dictionary containing all models
    """
    models = {}
    
    # DCAE for audio encoding/decoding
    models['dcae'] = create_dcae(
        latent_channels=config.dcae.latent_channels,
        sample_rate=config.dcae.sample_rate,
        **config.dcae.model_kwargs
    )
    
    # SSM + Flow matching generator
    models['generator'] = create_ssm_flow_generator(
        latent_channels=config.ssm.latent_channels,
        condition_dim=config.ssm.condition_dim,
        **config.ssm.model_kwargs
    )
    
    # Condition encoders
    models['caption_encoder'] = CaptionEncoder(
        model_name=config.encoders.caption_model,
        output_dim=config.encoders.caption_dim
    )
    
    models['lyrics_encoder'] = LyricsEncoder(
        vocab_size=config.encoders.vocab_size,
        embed_dim=config.encoders.lyrics_dim,
        max_length=config.encoders.max_lyrics_length
    )
    
    models['reference_encoder'] = ReferenceEncoder(
        dcae_model=models['dcae']
    )
    
    return models


def create_loss_functions(config, models):
    """
    Create all loss functions
    
    Args:
        config: Loss configuration
        models: Model dictionary
        
    Returns:
        CombinedLoss: Combined loss function
    """
    
    # Individual loss components
    flow_loss = FlowMatchingLoss(
        beta_schedule=config.loss.beta_schedule,
        num_timesteps=config.loss.num_timesteps
    )
    
    repa_loss = REPALoss(
        hubert_model=config.loss.hubert_model,
        layer_weights=config.loss.repa_layer_weights
    )
    
    recon_loss = ReconstructionLoss(
        l1_weight=config.loss.l1_weight,
        l2_weight=config.loss.l2_weight
    )
    
    perceptual_loss = PerceptualLoss(
        mel_weight=config.loss.mel_weight,
        stft_weight=config.loss.stft_weight
    )
    
    # Combined loss
    combined_loss = CombinedLoss(
        flow_loss=flow_loss,
        repa_loss=repa_loss,
        recon_loss=recon_loss,
        perceptual_loss=perceptual_loss,
        weights=config.loss.weights
    )
    
    return combined_loss


def get_model_info():
    """Get model architecture information"""
    return {
        'dcae': {
            'parameters': '~100M',
            'latent_channels': 16,
            'compression_ratio': '~50:1'
        },
        'ssm_generator': {
            'parameters': '~1.4B', 
            'architecture': 'S6 + Flow Matching',
            'conditions': ['lyrics', 'captions', 'reference']
        },
        'encoders': {
            'caption': 'Pretrained Text Encoder',
            'lyrics': 'Custom Tokenizer + Embedding',
            'reference': 'DCAE-based'
        },
        'total_parameters': '~1.5B'
    }