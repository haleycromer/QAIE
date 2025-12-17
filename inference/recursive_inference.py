import torch
import torch.nn as nn
from models.quantum_transformer import QuantumTransformer
from models.interpretation_network import InterpretationNetwork

class RecursiveInterference:
    """
    Recursive Interference Operator for QAIE.
    
    Continuously updates a latent state with incoming multi-modal tensors,
    producing evolving semantic/emotional/material interpretations.
    """
    
    def __init__(self, latent_dim=128, output_dim=15, alpha=0.3, device=None):
        """
        Parameters:
        - latent_dim: dimensionality of latent embeddings
        - output_dim: number of features to predict
        - alpha: blending factor for recursive update (0 < alpha <= 1)
        - device: 'cuda' or 'cpu'
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = QuantumTransformer(input_dim=9, latent_dim=latent_dim).to(self.device)
        self.interpreter = InterpretationNetwork(latent_dim=latent_dim, output_dim=output_dim).to(self.device)
        self.latent_state = torch.zeros(1, latent_dim).to(self.device)
        self.alpha = alpha  # weight of new observation

    def update(self, tensor):
        """
        Update latent state with a new tensor observation.
        
        Parameters:
        - tensor: [1, T, F] torch tensor of fused multi-modal data
        Returns:
        - output: torch tensor [1, output_dim] of semantic features
        """
        tensor = tensor.to(self.device)
        latent_new = self.encoder(tensor)
        # Recursive interference: blend previous latent with new
        self.latent_state = (1 - self.alpha) * self.latent_state + self.alpha * latent_new
        output = self.interpreter(self.latent_state)
        return output

    def reset(self):
        """Reset latent state to zero."""
        self.latent_state = torch.zeros_like(self.latent_state).to(self.device)
