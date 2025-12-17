import torch
import torch.nn as nn

class QuantumTransformer(nn.Module):
    def __init__(self, input_dim=9, latent_dim=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.embedding(x)        # [B, T, F]
        x = x.permute(1,0,2)        # [T, B, F] for transformer
        x = self.transformer(x)
        x = x.permute(1,2,0)        # [B, latent, T]
        x = self.pool(x).squeeze(-1)
        return x
