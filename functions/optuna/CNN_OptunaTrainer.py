import optuna
import numpy as np
import torch.nn as nn
import torch.optim as optim
from functions.build_model.training import train_model
from functions.networks.simple_cnn import SimpleCNN

class CNN_OptunaTrainer:
    def __init__(self, train_loader, val_loader, epochs, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.trial_results = []
        self.best_history = {"val_loss": None}

    def objective(self, trial):
        # Define the search space
        learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.01, log = True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.2)
        num_neurons = trial.suggest_categorical("num_neurons", [16, 32, 64, 128])
        stride = trial.suggest_categorical("stride", [2, 4, 8])
        kernel_size = trial.suggest_categorical("kernel_size", [2, 8, 16, 32])
        
        print("Prueba actual:",
            "\nlearning_rate", learning_rate,
            "\ndropout_rate", dropout_rate,
            "\nnum_neurons", num_neurons,
            "\nstride", stride,
            "\nkernel_size", kernel_size)
        
        # Define your model
        model = SimpleCNN(
            dropout_rate=dropout_rate,
            kernel_size=kernel_size,
            stride=stride,
            num_neurons=num_neurons,
        ).to(self.device)
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train and validate the model
        train_losses, val_losses, val_accuracies, best_model = train_model(
                        model, optimizer, criterion, self.train_loader,
                        self.val_loader, self.epochs, self.device
                    )
        val_loss = min(val_losses)
        val_accuracy=val_accuracies[np.argmin(val_losses)]

        # Append trial details to the results list
        self.trial_results.append({
            "model": "SimpleCNN_Meta",
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "num_neurons": num_neurons,
            "stride": stride,
            "kernel_size": kernel_size,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        # Update the best history if this trial is better
        if self.best_history is None or val_loss < min([h["val_loss"] for h in self.trial_results]):
            self.best_history = {
                "model": "SimpleCNN_Meta",
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_accuracies": val_accuracies,
                "best_model": best_model.state_dict(),
                "hyperparameters": {
                "learning_rate": learning_rate,
                "dropout_rate": dropout_rate,
                "num_neurons": num_neurons,
                "stride": stride,
                "kernel_size": kernel_size,
                },
            }
        return val_loss
    
    def run_study(self, n_trials=3):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        return study