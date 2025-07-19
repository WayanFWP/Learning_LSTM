import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        self.device = config['device']
        self.epochs = config['epochs']
        self.target_mse = config.get('target_mse', 1e-10)
        self.patience = config.get('patience', 100)
        self.min_improvement = config.get('min_improvement', 1e-12)
        
        # Track losses for visualization
        self.train_losses = []
        self.val_losses = []

    def validate(self):
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                total_val_loss += loss.item()
        return total_val_loss / len(self.val_loader)

    def plot_losses(self):
        plt.figure(figsize=(12, 5))
        
        # Plot training and validation losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training MSE Loss', color='blue')
        plt.plot(self.val_losses, label='Validation MSE Loss', color='red')
        plt.axhline(y=self.target_mse, color='green', linestyle='--', label=f'Target MSE ({self.target_mse})')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training and Validation MSE Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot log scale for better visibility
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, label='Training MSE Loss', color='blue')
        plt.plot(self.val_losses, label='Validation MSE Loss', color='red')
        plt.axhline(y=self.target_mse, color='green', linestyle='--', label=f'Target MSE ({self.target_mse})')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss (log scale)')
        plt.title('Training and Validation MSE Loss (Log Scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('mse_loss_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

    def train(self):
        self.model.to(self.device)
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        print(f"Training until MSE < {self.target_mse} (max {self.epochs} epochs)")
        print(f"Early stopping patience: {self.patience} epochs")
        print("-" * 60)
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            # Calculate average training loss
            avg_train_loss = total_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Calculate validation loss
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Check for improvement
            if val_loss < best_val_loss - self.min_improvement:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Print progress every 10 epochs or when target is reached
            if (epoch + 1) % 10 == 0 or val_loss < self.target_mse:
                print(f"Epoch {epoch+1:4d}/{self.epochs}, Train MSE: {avg_train_loss:.2e}, Val MSE: {val_loss:.2e}")
            
            # Check if target MSE is reached
            if val_loss < self.target_mse:
                print(f"\n🎉 TARGET MSE REACHED! 🎉")
                print(f"Validation MSE ({val_loss:.2e}) < Target MSE ({self.target_mse:.2e})")
                print(f"Training stopped at epoch {epoch+1}")
                break
            
            # Early stopping if no improvement
            if epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {self.patience} epochs without improvement")
                print(f"Best validation MSE: {best_val_loss:.2e}")
                break
        
        # Plot the losses after training
        self.plot_losses()
        
        # Print final statistics
        print(f"\nTraining completed!")
        print(f"Final Training MSE: {self.train_losses[-1]:.2e}")
        print(f"Final Validation MSE: {self.val_losses[-1]:.2e}")
        print(f"Target MSE: {self.target_mse:.2e}")
        print(f"Target reached: {'✅ YES' if self.val_losses[-1] < self.target_mse else '❌ NO'}")
        print("MSE loss plot saved as 'mse_loss_plot.png'")