import os
import torch
import numpy as np
import pandas as pd
from functions.load_data.load_data import load_data
from functions.load_data.load_calc_with_metadata import load_calc_with_metadata
from functions.optuna.get_val_accuracy import get_val_accuracy
from functions.optuna.Linear_OptunaTrainer import Linear_OptunaTrainer
from functions.optuna.Linear_Meta_OptunaTrainer import Linear_Meta_OptunaTrainer
from functions.optuna.SimpleLinear_OptunaTrainer import SimpleLinear_OptunaTrainer
from functions.optuna.SimpleLinear_Meta_OptunaTrainer import SimpleLinear_Meta_OptunaTrainer

# Consideraciones generales
BATCH_SIZE = 5
IMAGE_SIZE = (500, 500)
NUM_CLASSES = 2
EPOCHS = 20
N_TRIALS = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_built() else 
                      'cpu')

print(DEVICE)
os.makedirs("models", exist_ok=True)
model_name='calc_linear'

# Inicialización de listas y diccionarios
trial_results_list = []
best_histories = {}
combined_trial_results = []

datasets = [
    ('datasets/calc_clean/calc_clean_info.csv', 'clean'),
    ('datasets/calc_prop/calc_prop_info.csv', 'prop'),
    ('datasets/calc_dist/calc_dist_info.csv', 'dist')
 ]

for dataset_path, dataset_name in datasets:

    # Load data with metadata
    train_loader, val_loader, test_loader = load_calc_with_metadata(IMAGE_SIZE, BATCH_SIZE, dataset_path)

    # Simple_Linear_Meta Model
    full_model_name = f"Simple_Linear_Meta_{dataset_name}"
    print(full_model_name)
    Simple_Linear_Meta_trainer = SimpleLinear_Meta_OptunaTrainer(train_loader, val_loader, EPOCHS, DEVICE, 'calc', dataset_name)
    Simple_Linear_Meta_study = Simple_Linear_Meta_trainer.run_study()
    accuracy = get_val_accuracy(test_loader, Simple_Linear_Meta_trainer.best_history)
    Simple_Linear_Meta_trainer.best_history['accuracy'] = accuracy
    print(f"Test Accuracy of {full_model_name} model: {accuracy:.2f}%")
    best_histories[full_model_name] = Simple_Linear_Meta_trainer.best_history
    trial_results_list.append(pd.DataFrame(Simple_Linear_Meta_trainer.trial_results))

    # Linear_Meta Model
    full_model_name = f"Linear_Meta_{dataset_name}"
    print(full_model_name)
    Linear_Meta_trainer = Linear_Meta_OptunaTrainer(train_loader, val_loader, EPOCHS, DEVICE, 'calc', dataset_name)
    Linear_Meta_study = Linear_Meta_trainer.run_study(n_trials=N_TRIALS)
    accuracy = get_val_accuracy(test_loader, Linear_Meta_trainer.best_history)
    Linear_Meta_trainer.best_history['accuracy'] = accuracy
    print(f"Test Accuracy of {full_model_name} model: {accuracy:.2f}%")
    best_histories[full_model_name] = Linear_Meta_trainer.best_history
    trial_results_list.append(pd.DataFrame(Linear_Meta_trainer.trial_results))

    # Load data without metadata
    train_loader, val_loader, test_loader = load_data(IMAGE_SIZE, BATCH_SIZE, dataset_path)
    
    # Simple_Linear Model
    full_model_name = f"Simple_Linear_{dataset_name}"
    print(full_model_name)
    Simple_Linear_trainer = SimpleLinear_OptunaTrainer(train_loader, val_loader, EPOCHS, DEVICE, 'calc', dataset_name)
    Simple_Linear_study = Simple_Linear_trainer.run_study()
    accuracy = get_val_accuracy(test_loader, Simple_Linear_trainer.best_history)
    Simple_Linear_trainer.best_history['accuracy'] = accuracy
    print(f"Test Accuracy of {full_model_name} model: {accuracy:.2f}%")
    best_histories[full_model_name] = Simple_Linear_trainer.best_history
    trial_results_list.append(pd.DataFrame(Simple_Linear_trainer.trial_results))

    # Linear Model
    full_model_name = f"Linear_{dataset_name}"
    print(full_model_name)
    Linear_trainer = Linear_OptunaTrainer(train_loader, val_loader, EPOCHS, DEVICE, 'calc', dataset_name)
    Linear_study = Linear_trainer.run_study(n_trials=N_TRIALS)
    accuracy = get_val_accuracy(test_loader, Linear_trainer.best_history)
    Linear_trainer.best_history['accuracy'] = accuracy
    print(f"Test Accuracy of {full_model_name} model: {accuracy:.2f}%")
    best_histories[full_model_name] = Linear_trainer.best_history
    trial_results_list.append(pd.DataFrame(Linear_trainer.trial_results))


# Obtenemos el modelo con mejor pérdida en validacion
best_model_name = min(best_histories, key=lambda x: best_histories[x]["val_loss"])
best_best_history = best_histories[best_model_name]

# Combinamos los resultados de los trials
combined_trial_results = pd.concat(trial_results_list, ignore_index=True)

# Mejor modelo con sus métricas
print(f"The best model is {best_model_name} with validation loss {best_best_history['val_loss']} and accuracy {best_best_history['accuracy']:.2f}%")

# Guardamos el modelo y su historia, así como los resultados del resto de los intentos
torch.save(best_histories, f"models/best_histories_{model_name}.pth")
torch.save(best_best_history, f"models/best_model_{model_name}.pth")
combined_trial_results.to_csv(f"models/history_{model_name}.csv", index=False)
