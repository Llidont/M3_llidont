
import copy
import optuna
import numpy as np
import torch.nn as nn
import torch.optim as optim
from functions.build_model.training_with_metadata import train_model_with_metadata
from functions.build_model.training import train_model
from functions.networks.trans_Meta import trans_Meta

class Trans_Meta_OptunaTrainer:
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
        # Define the search space
        learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
        trans_dropout = trial.suggest_float("trans_dropout", 0.1, 0.3)
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        d_ff = trial.suggest_categorical("d_ff", [32, 64, 128])
        layer_filter = trial.suggest_categorical("layer_filter", [1, 4, 8])
        kernel_size = trial.suggest_categorical("kernel_size", [4, 8, 16])
        stride = trial.suggest_categorical("stride", [1, 2, 4])
        
        print("Prueba actual:",
            "\nlearning_rate", learning_rate,
            "\ntrans_dropout", trans_dropout,
            "\nd_model", d_model,
            "\nnum_heads", num_heads,
            "\nnum_layers", num_layers,
            "\nd_ff", d_ff,
            "\nlayer_filter", layer_filter,
            "\nkernel_size", kernel_size,
            "\nstride", stride)
        
        # Define your model
        model = trans_Meta(
            layer_filter=layer_filter,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            trans_dropout=trans_dropout,
            kernel_size=kernel_size,
            stride=stride
        ).to(self.device)
        
        initial_weights = copy.deepcopy(model.state_dict())
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train and validate the model
        train_losses, val_losses, val_accuracies, best_model = train_model(
            model, optimizer, criterion, self.train_loader,
            self.val_loader, self.epochs, self.device
        )
        val_loss = min(val_losses)
        val_accuracy = val_accuracies[np.argmin(val_losses)]

        # Append trial details to the results list
        self.trial_results.append({
            "model": "Trans_Meta",
            "type": self.type,
            "dataset": self.dataset,
            "learning_rate": learning_rate,
            "trans_dropout": trans_dropout,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "d_ff": d_ff,
            "layer_filter": layer_filter,
            "kernel_size": kernel_size,
            "stride": stride,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        # Update the best history if this trial is better
        if self.best_history["val_loss"] is None or val_loss < min([h["val_loss"] for h in self.trial_results]):
            self.best_history = {
                "model": "Trans_Meta",
                "type": self.type,
                "dataset": self.dataset,
                "train_losses": train_losses,
                "val_loss": val_loss,
                "val_losses": val_losses,
                "val_accuracies": val_accuracies,
                "best_model": best_model.state_dict(),
                "initial_weights": initial_weights,
                "hyperparameters": {
                    "learning_rate": learning_rate,
                    "trans_dropout": trans_dropout,
                    "d_model": d_model,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                    "d_ff": d_ff,
                    "layer_filter": layer_filter,
                    "kernel_size": kernel_size,
                    "stride": stride
                },
            }
        return val_loss
    
    def run_study(self, n_trials=3):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        return study
