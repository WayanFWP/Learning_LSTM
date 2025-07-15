from model.network import ActuatorLSTM
from train.trainer import Trainer
from data.preprocess import load_data
from config import config

def main():
    train_loader, val_loader = load_data('your_data.csv')
    
    model = ActuatorLSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        output_size=config['output_size']
    )

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

if __name__ == "__main__":
    main()
