import torch
import torch.nn as nn

class InterpretationNetwork(nn.Module):
    def __init__(self, latent_dim=128, output_dim=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)
