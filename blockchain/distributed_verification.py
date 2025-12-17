import hashlib
from datetime import datetime

ledger = []

def record_observation(obs_vector):
    timestamp = datetime.utcnow().isoformat()
    obs_str = f"{timestamp}:{obs_vector.tolist()}"
    obs_hash = hashlib.sha256(obs_str.encode()).hexdigest()
    ledger.append({"hash": obs_hash, "vector": obs_vector.tolist(), "timestamp": timestamp})
    return obs_hash

def verify_consensus():
    hashes = [entry['hash'] for entry in ledger]
    return hashes[0] if all(h==hashes[0] for h in hashes) else None
