import os
import torch
import numpy as np
import pandas as pd
from functions.load_data.load_data import load_data
from functions.load_data.load_calc_with_metadata import load_calc_with_metadata
from functions.optuna.get_val_accuracy import get_val_accuracy
from functions.optuna.Linear_OptunaTrainer import Linear_OptunaTrainer
from functions.optuna.SimpleLinear_OptunaTrainer import SimpleLinear_OptunaTrainer

# Consideraciones generales
BATCH_SIZE = 5
IMAGE_SIZE = (500, 500)
NUM_CLASSES = 2
EPOCHS = 5 #20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_built() else 
                      'cpu')

print(DEVICE)
os.makedirs("models", exist_ok=True)
model_name='calc_linear'

trial_results = []
best_history = None
train_loader, val_loader, test_loader = load_data(IMAGE_SIZE, BATCH_SIZE, 
                                            'datasets/calc_clean/calc_clean_info.csv')

# Corremos las búsquedas de hiperparámetros
# Simple_Linear_trainer = SimpleLinear_OptunaTrainer(train_loader, val_loader, EPOCHS, DEVICE)
# Simple_Linear_study = Simple_Linear_trainer.run_study()

Linear_trainer = Linear_OptunaTrainer(train_loader, val_loader, EPOCHS, DEVICE)
Linear_study = Linear_trainer.run_study(n_trials=2)

accuracy = get_val_accuracy(test_loader, Linear_trainer.best_history)

print(f"Test Accuracy of the best model: {accuracy:.2f}%")

# Guardamos el modelo y su historia, así como los resultados del resto de los intentos
torch.save(best_history, "models/model_calc_linear.pth")
torch.save(trial_results, "models/history_calc_linear.pth")
