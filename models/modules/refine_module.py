import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionRefineNet(nn.Module):
    def __init__(self, motion_dim=70, audio_dim=128, hidden_dim=128, num_conv_layers=3, use_attention=True):
        super().__init__()
        
        # 1. Local Temporal Feature Extraction (Dilated Convs)
        conv_layers = []
        conv_layers.append(nn.Conv1d(motion_dim, hidden_dim, kernel_size=3, padding=1))
        conv_layers.append(nn.GELU())
        
        for i in range(num_conv_layers - 2):
            dilation = 2 ** i
            padding = dilation
            conv_layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=padding, dilation=dilation))
            conv_layers.append(nn.GELU())
            
        self.local_temporal_net = nn.Sequential(*conv_layers)
        
        # =====================================================================
        # NEW: Audio-Conditioned Noise Modulator
        # Projects audio features into a scale (gamma) and bias (beta) 
        # to control the intensity of the injected high-frequency noise.
        # =====================================================================
        self.audio_modulator = nn.Sequential(
            nn.Conv1d(audio_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1) # * 2 because we need both gamma and beta
        )
        
        # 2. Feature-wise Refinement 
        # (Assuming ChannelAttention is defined elsewhere in your code)
        self.channel_attn = ChannelAttention(hidden_dim) 
        
        # 3. Global Temporal Context (Self-Attention)
        self.use_attention = use_attention
        if use_attention:
            # batch_first=True expects (B, T, D)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, 
                dropout=0.1, activation='gelu', batch_first=True
            )
            self.global_temporal_net = nn.TransformerEncoder(encoder_layer, num_layers=1)
            
        # 4. Output Projection back to motion dimensions
        self.output_proj = nn.Conv1d(hidden_dim, motion_dim, kernel_size=3, padding=1)
        
        # Zero-initialize the last layer for identity mapping at the start of training
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, coarse_motion, audio_features, inject_noise=True):
        """
        coarse_motion: (B, T, D_motion)
        audio_features: (B, T, D_audio) - Extracted from the Speaker Encoder
        returns: (B, T, D_motion)
        """
        # (B, T, D) -> (B, D, T) for Conv1d
        # print(f'coarse motion: {coarse_motion.shape}, audio feat: {audio_features.shape}')
        x = coarse_motion.transpose(1, 2)
        audio_x = audio_features.transpose(1, 2)
        
        # Local temporal smoothing
        features = self.local_temporal_net(x)
        
        # =====================================================================
        # NEW: Stochastic Micro-Dynamics Injection
        # =====================================================================
        # 1. Generate modulation parameters from the audio context
        # print(f'audio features: {audio_features.shape}, {audio_x.shape}')
        if inject_noise:
            modulation_params = self.audio_modulator(audio_x)
            # Split the output into Scale (gamma) and Shift (beta)
            gamma, beta = torch.chunk(modulation_params, 2, dim=1) 
            # 2. Sample raw stochastic noise matching the feature shape
            # This provides the raw "jitter"
            noise = torch.randn_like(features)
            
            # 3. Modulate the noise
            # If the audio signals an intense moment, gamma increases, amplifying the noise.
            # If the audio is silent, the network can learn to push gamma to 0.
            modulated_noise = (noise * gamma) + beta
            # 4. Inject the modulated noise into the motion features
            features = features + modulated_noise
            # =====================================================================

        # Channel attention to emphasize expression components vs pose components
        features = self.channel_attn(features)
        
        # Global temporal coherence via self-attention
        if self.use_attention:
            # Transformer expects (B, T, D) when batch_first=True
            features_t = features.transpose(1, 2)
            features_t = self.global_temporal_net(features_t)
            features = features_t.transpose(1, 2)
            
        # Project back to deltas
        deltas = self.output_proj(features)
        
        # Residual connection: Coarse + High-Frequency Delta
        refined_motion = x + deltas
        
        return refined_motion.transpose(1, 2)




class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation block to dynamically weight motion dimensions."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling along the temporal dimension
        pooled = x.mean(dim=2, keepdim=True) 
        attention = self.sigmoid(self.fc2(self.relu(self.fc1(pooled))))
        return x * attention