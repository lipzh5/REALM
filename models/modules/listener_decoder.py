import torch
import torch.nn as nn
from models.modules.utils import PositionalEncoding
from models.modules.reaction_gate import ReactionGate


class ReactionDecoder(nn.Module):
    def __init__(
        self,
        config, 
        out_motion_dim=58,
        feature_dim=128,
        window_size=8,
        nhead=4,
        num_layers=4,
        dropout=0.1,
        device='cuda'):
        super().__init__()
        self.reaction_gate = ReactionGate(dim=config.model.feature_dim)

        self.window_size = window_size
        self.feature_dim = feature_dim
        self.motion_dim = out_motion_dim
        self.device = device

        self.PE = PositionalEncoding(feature_dim, dropout)
        
        self.future_queries = nn.Parameter(torch.randn(1, window_size, feature_dim) * 0.02)
        
        mask = torch.triu(torch.ones(window_size, window_size) * float('-inf'), diagonal=1)
        self.register_buffer("tgt_mask", mask)

        self.exp_dim = getattr(config.model, 'exp_dim', 64)
        self.motion_proj = nn.Linear(out_motion_dim, feature_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.lstm = nn.LSTM(feature_dim, feature_dim, num_layers=1, batch_first=True)

        self.expr_out = nn.Linear(feature_dim, self.exp_dim)
        self.rot_out = nn.Linear(feature_dim, 3)
        self.tran_out = nn.Linear(feature_dim, 3)
        if out_motion_dim == 73:
            self.crop_out = nn.Linear(feature_dim, 3)
        

    def forward(self, fused_feat, past_motion=None):
        B = fused_feat.size(0)
        alpha = None 

        # 1. Prepare Queries
        queries = self.future_queries.repeat(B, 1, 1)
        queries = self.PE(queries)

        # 2. Prepare Memory 
        if past_motion is not None:
            past_motion_emb = self.PE(self.motion_proj(past_motion))
            alpha = self.reaction_gate(fused_feat)
            speaker_mem = alpha * fused_feat
            listener_mem = (1.0 - alpha) * past_motion_emb
            memory = torch.cat([listener_mem, speaker_mem], dim=1)
           
        else:
            memory = fused_feat

        # 3. Decoding 
        dec_out = self.decoder(
            tgt=queries,
            memory=memory,
            tgt_mask=self.tgt_mask[:self.window_size, :self.window_size]
        )

        lstm_out, _ = self.lstm(dec_out)
        
        pred_expr = self.expr_out(lstm_out)
        pred_rot = self.rot_out(lstm_out)
        pred_tran = self.tran_out(lstm_out)
        
        return pred_expr, pred_rot, pred_tran, alpha