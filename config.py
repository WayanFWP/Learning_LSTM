config = {
    'input_size': 5,
    'hidden_size': 64,
    'output_size': 1,
    'epochs': 50,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
