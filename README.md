# Lyro v2 - Simplified Music Generation Model

A streamlined version of the multimodal music generation system with significantly reduced complexity and improved maintainability.

## Features

- **Simplified Architecture**: Reduced from ~943M to ~400M parameters
- **Clean Codebase**: Removed redundant components and unnecessary complexity
- **Easy to Use**: Simple pipeline interface for music generation
- **Flow Matching**: Efficient latent space generation with CFG support

## Core Components

### 1. Generator (`models/generator.py`)
- **LyroGenerator**: Main generation model with SSM + Transformer architecture
- **S6Layer**: State space model implementation
- **MultiHeadAttention**: Attention mechanism with RoPE
- **ConditionProcessor**: Simplified text conditioning

### 2. Text Encoder (`models/encoders.py`)
- **UnifiedTextEncoder**: Combined lyrics and style encoding
- **ReferenceEncoder**: Optional reference audio encoding

### 3. System Integration (`models/multimodal_lyro.py`)
- **MultimodalLyroSystem**: Unified system interface
- Simple forward pass and generation methods

### 4. Pipeline (`inference/pipeline.py`)
- **LyroPipeline**: Easy-to-use inference interface
- **GenerationConfig**: Configurable generation parameters
- Helper functions for quick generation

## Usage

```python
from inference.pipeline import quick_generate

# Generate music from lyrics
latents = quick_generate(
    lyrics="Walking down the street tonight",
    style="pop ballad emotional",
    quality="high"
)

# Using the full pipeline
from inference.pipeline import LyroPipeline, GenerationInput, GenerationConfig

pipeline = LyroPipeline.from_pretrained("path/to/checkpoint")
input_data = GenerationInput(lyrics="Your lyrics here", style="genre and mood")
config = GenerationConfig(quality="high", num_steps=100)

result = pipeline.generate(input_data, config)
```

## Configuration

The system uses simplified configuration classes in `training/config.py`:

- **GeneratorConfig**: Model architecture settings
- **LossConfig**: Training loss configuration
- **DataConfig**: Dataset parameters
- **TrainingConfig**: Training setup

## Model Architecture

- **Model Size**: ~400M parameters (reduced from 943M)
- **Input**: Text embeddings (lyrics + style)
- **Output**: Latent vectors (16 channels, 128 timesteps)
- **Backbone**: SSM (State Space Model) + Transformer
- **Generation**: Flow Matching with CFG

## Key Improvements

1. **Code Reduction**: Removed ~60% of unnecessary code
2. **Simplified Components**: 
   - Merged redundant encoder implementations
   - Streamlined configuration system
   - Removed complex multimodal fusion layers
3. **Better Maintainability**: Clear separation of concerns
4. **Memory Efficiency**: Smaller model size and optimized operations

## Files Removed/Simplified

- Removed `models/ssm_flow.py` (redundant with generator.py)
- Simplified `models/encoders.py` (removed unused encoders)
- Streamlined `models/multimodal_lyro.py` 
- Cleaned up `inference/pipeline.py`
- Simplified `training/config.py`

## Development

The codebase is now much cleaner and easier to work with:
- Clear module boundaries
- Minimal dependencies between components
- Simplified interfaces
- Better code organization

This refactored version maintains the core functionality while being much more maintainable and efficient.