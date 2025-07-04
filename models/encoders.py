# lyro/models/encoders.py
"""
Condition Encoders for LYRO (수정됨 - Generator 전용)
Lyrics, Caption (MusicCaps), and Reference Latent encoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from transformers import AutoModel, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")


class LyricsEncoder(nn.Module):
    """
    Lyrics encoder with custom tokenizer and embedding
    Handles variable length lyrics with attention pooling
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_length: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, embed_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            token_ids: (B, L) token IDs
            attention_mask: (B, L) attention mask
            
        Returns:
            (B, embed_dim) lyrics embedding
        """
        B, L = token_ids.shape
        
        # Truncate if too long
        if L > self.max_length:
            token_ids = token_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
            L = self.max_length
            
        # Token embedding
        x = self.token_embedding(token_ids)  # (B, L, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embedding[:, :L, :]
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to boolean mask (True = ignore)
            mask = ~attention_mask.bool()
        else:
            mask = None
            
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)  # (B, L, embed_dim)
        
        # Output projection
        x = self.output_proj(x)
        
        # Attention pooling
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, embed_dim)
        pooled, _ = self.attention_pool(query, x, x, key_padding_mask=mask)
        
        return pooled.squeeze(1)  # (B, embed_dim)


class CaptionEncoder(nn.Module):
    """
    Caption encoder using pretrained text encoder
    Supports MusicCaps-style descriptions
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 768,
        freeze_encoder: bool = True,
        max_length: int = 256
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.max_length = max_length
        
        try:
            # Load pretrained model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)
            
            # Freeze encoder if specified
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                    
            # Get encoder output dimension
            encoder_dim = self.encoder.config.hidden_size
            
            # Output projection
            if encoder_dim != output_dim:
                self.proj = nn.Sequential(
                    nn.Linear(encoder_dim, output_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(output_dim * 2, output_dim),
                    nn.LayerNorm(output_dim)
                )
            else:
                self.proj = nn.Identity()
                
        except Exception as e:
            print(f"Warning: Failed to load pretrained encoder {model_name}: {e}")
            print("Using fallback embedding layer")
            
            # Fallback to simple embedding
            self.tokenizer = None
            self.encoder = None
            self.embedding = nn.Sequential(
                nn.Embedding(50000, 512),  # Large vocab size
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(512, 8, 2048, batch_first=True),
                    num_layers=6
                ),
                nn.AdaptiveAvgPool1d(1),
                nn.Linear(512, output_dim)
            )
            self.proj = nn.Identity()
            
    def encode_text(self, texts: list) -> torch.Tensor:
        """Encode text using tokenizer"""
        if self.tokenizer is None:
            # Fallback: simple character encoding
            max_len = min(self.max_length, 100)
            batch_size = len(texts)
            device = next(self.parameters()).device
            
            encoded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
            
            for i, text in enumerate(texts):
                for j, char in enumerate(text[:max_len]):
                    encoded[i, j] = ord(char) % 50000
                    
            return encoded
        else:
            # Use actual tokenizer
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            return encoded
    
    def forward(self, captions: list) -> torch.Tensor:
        """
        Args:
            captions: List of caption strings
            
        Returns:
            (B, output_dim) caption embeddings
        """
        if self.encoder is not None:
            # Use pretrained encoder
            encoded = self.encode_text(captions)
            
            # Move to correct device
            device = next(self.parameters()).device
            for key in encoded:
                if isinstance(encoded[key], torch.Tensor):
                    encoded[key] = encoded[key].to(device)
            
            # Encode
            with torch.set_grad_enabled(not getattr(self.encoder, '_frozen', False)):
                outputs = self.encoder(**encoded)
                
            # Mean pooling
            embeddings = outputs.last_hidden_state
            attention_mask = encoded.get('attention_mask', None)
            
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            else:
                embeddings = embeddings.mean(dim=1)
                
            # Project to output dimension
            embeddings = self.proj(embeddings)
            
        else:
            # Use fallback embedding
            encoded = self.encode_text(captions)
            embeddings = self.embedding(encoded)
            if embeddings.dim() > 2:
                embeddings = embeddings.mean(dim=1)
                
        return embeddings


class ReferenceEncoder(nn.Module):
    """
    Reference latent encoder (수정됨 - DCAE 제거, latent 직접 처리)
    Encodes reference latent vectors to condition generation
    """
    
    def __init__(self, input_channels: int = 16, output_dim: int = 512):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        # Reference latent processing network
        self.ref_processor = nn.Sequential(
            # Latent processing layers
            nn.Conv1d(input_channels, input_channels * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(input_channels * 2, input_channels * 4, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(input_channels * 4, input_channels * 8, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_channels * 8, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Style transfer layers
        self.style_layers = nn.ModuleList([
            nn.Linear(output_dim, output_dim) for _ in range(3)
        ])
        
    def forward(self, reference_latents: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            reference_latents: (B, C, T) reference latent vectors or None
            
        Returns:
            (B, output_dim) reference embedding
        """
        if reference_latents is None:
            # Return zero embedding
            batch_size = 1
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.output_dim, device=device)
            
        batch_size = reference_latents.shape[0]
        device = reference_latents.device
        
        try:
            # Process reference latents
            ref_embedding = self.ref_processor(reference_latents)
            
            # Apply style transformations
            for layer in self.style_layers:
                ref_embedding = ref_embedding + layer(F.gelu(ref_embedding))
                
            return ref_embedding
            
        except Exception as e:
            print(f"Warning: Reference encoding failed: {e}")
            # Return zero embedding on failure
            return torch.zeros(batch_size, self.output_dim, device=device)
            
    def extract_style_features(self, reference_latents: torch.Tensor, num_features: int = 8) -> torch.Tensor:
        """
        Extract multiple style features from reference latents
        
        Args:
            reference_latents: (B, C, T) reference latent vectors
            num_features: Number of style features to extract
            
        Returns:
            (B, num_features, output_dim) style features
        """
        batch_size = reference_latents.shape[0]
        
        # Segment latents into multiple parts
        segment_length = reference_latents.shape[-1] // num_features
        style_features = []
        
        for i in range(num_features):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length if i < num_features - 1 else reference_latents.shape[-1]
            
            segment = reference_latents[..., start_idx:end_idx]
            if segment.shape[-1] < 8:  # Too short
                segment = F.pad(segment, (0, 8 - segment.shape[-1]))
                
            segment_feature = self.forward(segment)
            style_features.append(segment_feature)
            
        return torch.stack(style_features, dim=1)  # (B, num_features, output_dim)


class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder that combines all condition types (수정됨 - Reference latent 사용)
    """
    
    def __init__(
        self,
        lyrics_encoder: LyricsEncoder,
        caption_encoder: CaptionEncoder,
        reference_encoder: ReferenceEncoder,
        fusion_dim: int = 1024
    ):
        super().__init__()
        
        self.lyrics_encoder = lyrics_encoder
        self.caption_encoder = caption_encoder
        self.reference_encoder = reference_encoder
        self.fusion_dim = fusion_dim
        
        # Get individual encoder output dimensions
        lyrics_dim = lyrics_encoder.embed_dim
        caption_dim = caption_encoder.output_dim
        reference_dim = reference_encoder.output_dim
        
        # Cross-attention fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=16,
            batch_first=True
        )
        
        # Individual projections to fusion dimension
        self.lyrics_proj = nn.Linear(lyrics_dim, fusion_dim)
        self.caption_proj = nn.Linear(caption_dim, fusion_dim)
        self.reference_proj = nn.Linear(reference_dim, fusion_dim)
        
        # Final fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(
        self,
        lyrics_tokens: Optional[torch.Tensor] = None,
        captions: Optional[list] = None,
        reference_latents: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode all modalities and return individual and fused embeddings
        
        Args:
            lyrics_tokens: (B, L) lyrics token IDs
            captions: List of caption strings
            reference_latents: (B, C, T) reference latent vectors
            attention_mask: (B, L) lyrics attention mask
            
        Returns:
            Dictionary containing individual and fused embeddings
        """
        batch_size = (
            lyrics_tokens.shape[0] if lyrics_tokens is not None else
            len(captions) if captions is not None else
            reference_latents.shape[0] if reference_latents is not None else 1
        )
        
        device = next(self.parameters()).device
        
        # Encode each modality
        embeddings = {}
        
        # Lyrics
        if lyrics_tokens is not None:
            lyrics_emb = self.lyrics_encoder(lyrics_tokens, attention_mask)
            embeddings['lyrics'] = self.lyrics_proj(lyrics_emb)
        else:
            embeddings['lyrics'] = torch.zeros(batch_size, self.fusion_dim, device=device)
            
        # Captions
        if captions is not None:
            caption_emb = self.caption_encoder(captions)
            embeddings['caption'] = self.caption_proj(caption_emb)
        else:
            embeddings['caption'] = torch.zeros(batch_size, self.fusion_dim, device=device)
            
        # Reference latents
        if reference_latents is not None:
            reference_emb = self.reference_encoder(reference_latents)
            embeddings['reference'] = self.reference_proj(reference_emb)
        else:
            embeddings['reference'] = torch.zeros(batch_size, self.fusion_dim, device=device)
            
        # Cross-modal attention fusion
        all_embeddings = torch.stack([
            embeddings['lyrics'],
            embeddings['caption'], 
            embeddings['reference']
        ], dim=1)  # (B, 3, fusion_dim)
        
        fused, _ = self.cross_attention(all_embeddings, all_embeddings, all_embeddings)
        fused = fused.mean(dim=1)  # (B, fusion_dim)
        
        # Final fusion
        concat_embeddings = torch.cat([
            embeddings['lyrics'],
            embeddings['caption'],
            embeddings['reference']
        ], dim=-1)  # (B, fusion_dim * 3)
        
        final_embedding = self.fusion_layers(concat_embeddings)
        
        # Add residual connection with cross-attention result
        final_embedding = final_embedding + fused
        
        return {
            'lyrics': embeddings['lyrics'],
            'caption': embeddings['caption'],
            'reference': embeddings['reference'],
            'fused': final_embedding
        }