import torch
import torch.nn as nn
import torch.optim as optim
from dataset.multimodal_synthetic import generate_multimodal_event
from preprocessing.tensor_fusion import multimodal_tensor
from models.quantum_transformer import QuantumTransformer
from models.interpretation_network import InterpretationNetwork
from blockchain.distributed_verification import record_observation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

encoder = QuantumTransformer().to(device)
interpreter = InterpretationNetwork().to(device)
optimizer = optim.Adam(list(encoder.parameters()) + list(interpreter.parameters()), lr=1e-3)
loss_fn = nn.MSELoss()

for step in range(20):
    audio, vib, env = generate_multimodal_event()
    tensor = multimodal_tensor(audio, vib, env)
    tensor = torch.tensor(tensor).unsqueeze(0).to(device)  # [1, T, F]
    
    target = torch.rand((1, 15)).to(device)  # synthetic target
    latent = encoder(tensor)
    output = interpreter(latent)
    
    loss = loss_fn(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record in blockchain
    record_observation(output.detach())
    
    print(f"Step {step+1} | Loss: {loss.item():.4f}")
