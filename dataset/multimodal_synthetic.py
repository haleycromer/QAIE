import numpy as np

def generate_multimodal_event(duration=1.0, sr=44100):
    """Generate multi-modal audio + vibration + environmental signals"""
    t = np.linspace(0, duration, int(sr*duration))
    audio = 0.5*np.sin(2*np.pi*440*t) + 0.3*np.sin(2*np.pi*880*t) + 0.05*np.random.randn(len(t))
    vibration = np.sin(2*np.pi*5*t) + 0.02*np.random.randn(len(t))
    env = np.random.rand(len(t)) * 0.1
    return audio.astype(np.float32), vibration.astype(np.float32), env.astype(np.float32)
