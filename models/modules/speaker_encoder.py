import torch
import torch.nn as nn
from models.modules.utils import PositionalEncoding


class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        motion_dim=-1, 
        audio_dim=128,     # Changed default to 128
        feature_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.audio_local = nn.Sequential(
            nn.Conv1d(audio_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.pos_encoding = PositionalEncoding(feature_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final LayerNorm for stability
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, audio_feat, dummy_motion=None):
        """
        audio_feat : (B, T, D)
        """
        # 1. Local Feature Extraction (Conv1d expects (B, C, T))
        a = self.audio_local(audio_feat.transpose(1, 2)).transpose(1, 2)   # (B, T, D)

        # 2. Add Positional Encoding and Transformer
        fused = self.pos_encoding(a)
        out = self.transformer(fused)       # (B, T, D)

        return self.norm(out)