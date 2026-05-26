import torch
import torch.nn as nn
from models.modules.utils import PositionalEncoding

def get_shifted_alibi_mask(seq_len, num_heads, batch_size, delay_frames=8, device='cuda'):
    """
    Generates a biologically-shifted ALiBi mask for cross-attention.
    delay_frames: The optimal reaction delay (e.g., 8 frames = ~260ms at 30fps).
    """
    q_pos = torch.arange(seq_len, device=device).unsqueeze(1) 
    k_pos = torch.arange(seq_len, device=device).unsqueeze(0) 
    
    dist = q_pos - k_pos 
    causal_mask = dist < 0 
    
    shifted_dist = torch.abs(dist - delay_frames)
    
    slopes = [1.0 / (2 ** i) for i in range(1, num_heads + 1)]
    slopes = torch.tensor(slopes, device=device).view(num_heads, 1, 1)
    
    alibi_bias = -slopes * shifted_dist
    alibi_bias.masked_fill_(causal_mask, float('-inf'))
    
    alibi_bias = alibi_bias.repeat(batch_size, 1, 1)
    return alibi_bias


class SpeakerListenerFusion(nn.Module):
    """
    Fuses the historical context of the conversation.
    Listener's physical history queries the Speaker's audio history using ALiBi.
    """
    def __init__(self, motion_dim,feature_dim=128, nhead=4, num_layers=1, dropout=0.1, delay_frames=8):
        super().__init__()
        self.nhead = nhead
        self.delay_frames = delay_frames
        print(f'Speaker Listener Fusion Delay {self.delay_frames} Frames!!! \n =========')
        self.listener_proj = nn.Linear(motion_dim, feature_dim)
        # Positional Encoding is ONLY applied at the very end
        self.PE = PositionalEncoding(feature_dim, dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # 1 or 2 layers is usually plenty for historical fusion
        self.cross_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, speaker_feat, past_motion, project=True):
        """
        Inputs should already be projected to `feature_dim` (128).
        DO NOT apply Positional Encoding to these inputs before passing them here!
        """
        minT = past_motion.shape[1]
        speaker_feat = speaker_feat[:, -minT:]
        past_motion_emb = self.listener_proj(past_motion)
        B = past_motion_emb.size(0)
        K = past_motion_emb.size(1)

        # 1. Generate ALiBi Mask (Handles timeline and reaction delay natively)
        alibi_mask = get_shifted_alibi_mask(
            seq_len=K, 
            num_heads=self.nhead, 
            batch_size=B,
            delay_frames=self.delay_frames, 
            device=past_motion_emb.device
        )

        # 2. Cross-Modal Attention: Motion history looks back at Audio history
        fused = self.cross_decoder(
            tgt=past_motion_emb,       
            memory=speaker_feat,    
            memory_mask=alibi_mask     
        )
        
        fused = self.output_norm(fused)

        # 3. Add Positional Encoding to the FINAL fused output so the 
        # downstream ReactionDecoder knows the timeline of this unified memory.
        return self.PE(fused)