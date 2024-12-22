#Imports 
import os
import copy
import torch
import optuna
import numpy as np
import torch.nn as nn
import torch.optim as optim
from functions.optuna.get_val_accuracy import get_val_accuracy
from functions.networks.simple_cnn_with_metadata import SimpleCNN
from functions.load_data.load_data import load_data
from functions.optuna.CNN_OptunaTrainer import CNN_OptunaTrainer

# Consideraciones generales
BATCH_SIZE = 5
IMAGE_SIZE = (500, 500)
NUM_CLASSES = 2
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_built() else 
                      'cpu')
                      
print(DEVICE)
os.makedirs("models", exist_ok=True)
model_name='calc_cnnmeta'
trial_results = []       
best_history = None
train_loader, val_loader, test_loader = load_data(IMAGE_SIZE, BATCH_SIZE, 
                                            'datasets/calc_clean/calc_clean_info.csv')

# Corremos las búsquedas de hiperparámetros
CNN_trainer = CNN_OptunaTrainer(train_loader, val_loader, SimpleCNN, EPOCHS, DEVICE)
CNN_study = CNN_trainer.run_study(n_trials=2)

accuracy = get_val_accuracy(test_loader, CNN_trainer.best_history)

min_loss_index = np.argmin(best_history["val_losses"])
best_val_loss = best_history["val_losses"][min_loss_index]
best_val_acc = best_history["val_accuracies"][min_loss_index]
# Best hyperparameters
print("Best hyperparameters:", CNN_study.best_params)
print("Best validation loss:", best_val_loss)
print("Best validation accuracy:", best_val_acc)

print("Validation accuracy in test:")

print(f"Test Accuracy of the best model: {accuracy:.2f}%")

# Guardamos el modelo y su historia, así como los resultados del resto de los intentos
torch.save(best_history, "models/model_calc_cnnmeta.pth")
torch.save(trial_results, "models/history_calc_cnnmeta.pth")
