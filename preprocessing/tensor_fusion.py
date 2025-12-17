import numpy as np
from scipy.signal import stft

def multimodal_tensor(audio, vibration, env, sr=44100):
    """Convert multi-modal signals into a fused high-dimensional tensor"""
    def stft_tensor(sig):
        f, t, Zxx = stft(sig, fs=sr, nperseg=1024)
        amplitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        entropy = -np.sum(amplitude*np.log(amplitude + 1e-8), axis=0)
        return np.stack([amplitude, phase, entropy], axis=-1)
    
    audio_t = stft_tensor(audio)
    vib_t = stft_tensor(vibration)
    env_t = stft_tensor(env)
    fused = np.concatenate([audio_t, vib_t, env_t], axis=-1)
    return fused.astype(np.float32)
