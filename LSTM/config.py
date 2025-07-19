import torch

config = {
    'input_size': 5,
    'hidden_size': 128,
    'output_size': 1,
    'epochs': 100000,  # Set high maximum epochs as safety limit
    'lr': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'target_mse': 1e-10,  # Target MSE threshold
    'patience': 200,  # Stop if no improvement for 100 epochs
    'min_improvement': 1e-12  # Minimum improvement to consider as progress
}