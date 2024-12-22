import copy
import optuna
import numpy as np
import torch.nn as nn
import torch.optim as optim
from functions.build_model.training import train_model
from functions.networks.linear import Simple_Linear

class SimpleLinear_OptunaTrainer:
    def __init__(self, train_loader, val_loader, epochs, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.trial_results = []
        self.best_history = {"val_loss": None}

    def objective(self, trial):
        learning_rate = trial.suggest_categorical("learning_rate", [0.001, 0.01])
        dropout_rate = trial.suggest_categorical("dropout_rate", [0.1, 0.2])

        print("Trial:",
              "\nLearning rate:", learning_rate,
              "\nDropout rate:", dropout_rate)

        # Initialize model
        model = Simple_Linear(
            dropout_rate=dropout_rate,
        ).to(self.device)
        
        initial_weights = copy.deepcopy(model.state_dict())

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train and validate
        train_losses, val_losses, val_accuracies, best_model = train_model(
            model, optimizer, criterion, self.train_loader, self.val_loader, self.epochs, self.device, self.model_class
        )

        val_loss = min(val_losses)
        val_accuracy = val_accuracies[np.argmin(val_losses)]

        # Save trial results
        self.trial_results.append({
            "model": "Simple_Linear",
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "train_loss": train_losses,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        # Update best model
        if self.best_history["val_loss"] is None or val_loss < self.best_history["val_loss"]:
            self.best_history.update({
                "model": "Simple_Linear",
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "best_model": best_model.state_dict(),
                "initial_weights": initial_weights,
                "hyperparameters": {
                    "learning_rate": learning_rate,
                    "dropout_rate": dropout_rate,
                },
            })

        return val_loss
    
    def run_study(self):
        self.search_space = {
            "learning_rate": [0.01, 0.001],
            "dropout_rate": [0.1, 0.2]
        }
        sampler = optuna.samplers.GridSampler(self.search_space)
        study = optuna.create_study(sampler=sampler, direction="minimize")
        study.optimize(self.objective)
        return study
