import copy
import optuna
import numpy as np
import torch.nn as nn
import torch.optim as optim
from functions.build_model.training_with_metadata import train_model_with_metadata
from functions.networks.simple_cnn_Meta import SimpleCNN_Meta

class CNNMeta_OptunaTrainer:
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
        '''Método para generar cada una de las pruebas de Optuna'''
        # Definimos hiperparámetros
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
        
        # Inicializamos el modelo
        model = SimpleCNN_Meta(
            dropout_rate=dropout_rate,
            kernel_size=kernel_size,
            stride=stride,
            num_neurons=num_neurons,
        ).to(self.device)
        
        # Guardamos pesos iniciales por si fuese necesario
        initial_weights = copy.deepcopy(model.state_dict())
        
        # Definimos la función de pérdida y el optimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Entrenamiento del modelo
        train_losses, val_losses, val_accuracies, best_model = train_model_with_metadata(
                        model, optimizer, criterion, self.train_loader,
                        self.val_loader, self.epochs, self.device
                    )
        val_loss = min(val_losses)
        val_accuracy=val_accuracies[np.argmin(val_losses)]

        # Añadimos a la lista de pruebas
        self.trial_results.append({
            "model": "SimpleCNN_Meta",
            "type": self.type,
            "dataset": self.dataset,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "num_neurons": num_neurons,
            "stride": stride,
            "kernel_size": kernel_size,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        # Si pertoca, actualizamos el mejor modelo
        if self.best_history["val_loss"] is None or val_loss < min([h["val_loss"] for h in self.trial_results]):
            self.best_history = {
                "model": "SimpleCNN_Meta",
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
                    "dropout_rate": dropout_rate,
                    "num_neurons": num_neurons,
                    "stride": stride,
                    "kernel_size": kernel_size,
                },
            }
        return val_loss
    
    def run_study(self, n_trials=3):
        '''Método para iniciar búsqueda de hiperparámetros'''
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        return study