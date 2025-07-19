import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ActuatorDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), \
               torch.tensor(self.targets[idx], dtype=torch.float32)

def load_data(csv_path, window_size=10):
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Assume columns: error, velocity, Kp, Kd, tau_ff, torque
    features = df[['error', 'velocity', 'Kp', 'Kd', 'tau_ff']].values
    target = df['torque'].values

    sequences = []
    targets = []

    for i in range(len(df) - window_size):
        seq = features[i:i+window_size]
        tgt = target[i+window_size - 1]  # predict last time step
        sequences.append(seq)
        targets.append([tgt])

    split = int(len(sequences) * 0.8)
    train_dataset = ActuatorDataset(sequences[:split], targets[:split])
    val_dataset = ActuatorDataset(sequences[split:], targets[split:])

    return DataLoader(train_dataset, batch_size=32, shuffle=True), \
           DataLoader(val_dataset, batch_size=32)
