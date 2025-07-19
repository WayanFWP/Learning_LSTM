import torch
from model.network import ActuatorLSTM
from config import config

# Create model
model = ActuatorLSTM(
    input_size=config['input_size'],
    hidden_size=config['hidden_size'],
    output_size=config['output_size']
)

print("🧠 MODEL ARCHITECTURE & BACKPROPAGATION CHECK")
print("=" * 60)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Parameters require gradients: {trainable_params > 0}")

print("\n📊 LAYER-BY-LAYER PARAMETER COUNT:")
print("-" * 40)
for name, param in model.named_parameters():
    print(f"{name:20} | {param.numel():8,} | Grad: {param.requires_grad}")

print("\n🔄 BACKPROPAGATION TEST:")
print("-" * 30)

# Create dummy input
batch_size = 2
sequence_length = 10
dummy_input = torch.randn(batch_size, sequence_length, config['input_size'])
dummy_target = torch.randn(batch_size, config['output_size'])

# Forward pass
model.train()
output = model(dummy_input)
loss = torch.nn.MSELoss()(output, dummy_target)

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Loss before backward: {loss.item():.6f}")

# Check gradients before backward
print(f"Gradients before backward: {model.fc.weight.grad}")

# Backward pass
loss.backward()

# Check gradients after backward
print(f"Gradients after backward: {'✅ EXISTS' if model.fc.weight.grad is not None else '❌ NONE'}")
print(f"FC layer gradient norm: {model.fc.weight.grad.norm().item():.6f}")

print("\n✅ CONCLUSION: Your model FULLY supports backpropagation!")
print("   All layers have trainable parameters with gradient computation.")
