import torch
import torch.nn as nn


class ReactionGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )
        nn.init.constant_(self.mlp[2].bias, 0.0)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, queries):
        # queries: (B, W, D)
        logits = self.mlp(queries)          # (B, W, 1)
        alpha = self.sigmoid(logits)        
        return alpha