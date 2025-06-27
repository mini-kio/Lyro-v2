# lyro/data/__init__.py
"""
LYRO Data Package
Unified data loading and processing
"""

from .dataset import LyroDataset, create_lyro_datasets
from .collator import LyroCollator
from .tokenizer import LyroTokenizer

__all__ = [
    'LyroDataset',
    'LyroCollator', 
    'LyroTokenizer',
    'create_lyro_datasets'
]


def create_data_pipeline(config):
    """
    Create complete data pipeline
    
    Args:
        config: Data configuration
        
    Returns:
        dict: Data pipeline components
    """
    # Create tokenizer
    tokenizer = LyroTokenizer(
        vocab_size=config.tokenizer.vocab_size,
        max_length=config.tokenizer.max_length
    )
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_lyro_datasets(
        train_metadata=config.train_metadata,
        val_metadata=config.val_metadata,
        test_metadata=config.test_metadata,
        dataset_root=config.dataset_root,
        tokenizer=tokenizer,
        **config.dataset_kwargs
    )
    
    # Create collator
    collator = LyroCollator(
        tokenizer=tokenizer,
        max_audio_length=config.max_audio_length,
        max_text_length=config.max_text_length,
        **config.collator_kwargs
    )
    
    return {
        'tokenizer': tokenizer,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'collator': collator
    }