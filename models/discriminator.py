import torch.nn as nn


class TemporalDiscriminator(nn.Module):
    def __init__(self, motion_dim=73, hidden_dim=128):
        super().__init__()
        # Conv1d expects (Batch, Channels, Time)
        self.net = nn.Sequential(
            nn.Conv1d(motion_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool1d(1), # Global average pooling over time
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, 1) # Outputs a single realism score
        )

    def forward(self, x):
        # x is (B, T, D) -> convert to (B, D, T) for Conv1d
        x = x.transpose(1, 2)
        return self.net(x)