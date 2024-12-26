import copy
import optuna
import numpy as np
import torch.nn as nn
import torch.optim as optim
from functions.networks.linear import Linear
from functions.build_model.training import train_model

class Linear_OptunaTrainer:
    def __init__(self, train_loader, val_loader, epochs, device, type, dataset):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.type = type
        self.dataset = dataset
        self.trial_results = []
        self.best_history = {"val_loss": None}

    def objective(self, trial):
        # Define hyperparameter search space
        learning_rate = trial.suggest_categorical("learning_rate", [0.001, 0.01])
        dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2])
        num_neurons = trial.suggest_categorical("num_neurons", [32, 64, 128])

        print("Prueba actual:",
              "\nLearning rate:", learning_rate,
              "\nDropout rate:", dropout_rate,
              "\nNum neurons:", num_neurons)

        # Initialize model
        model = Linear(
            dropout_rate=dropout_rate,
            num_neurons=num_neurons,
        ).to(self.device)
        
        initial_weights = copy.deepcopy(model.state_dict())

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train and validate
        train_losses, val_losses, val_accuracies, best_model = train_model(
            model, optimizer, criterion, self.train_loader, self.val_loader, self.epochs, self.device
        )

        val_loss = min(val_losses)
        val_accuracy = val_accuracies[np.argmin(val_losses)]

        # Save trial results
        self.trial_results.append({
            "model": "Linear",
            "type": self.type,
            "dataset": self.dataset,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "num_neurons": num_neurons,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        # Update best model
        if self.best_history["val_loss"] is None or val_loss < self.best_history["val_loss"]:
            self.best_history.update({
                "model": "Linear",
                "type": self.type,
                "dataset": self.dataset,
                "val_loss": val_loss,
                "val_losses": val_losses,
                "val_accuracies": val_accuracies,
                "best_model": best_model.state_dict(),
                "initial_weights": initial_weights,
                "hyperparameters": {
                    "learning_rate": learning_rate,
                    "dropout_rate": dropout_rate,
                    "num_neurons": num_neurons,
                },
            })

        return val_loss
    
    def run_study(self, n_trials):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        return study


