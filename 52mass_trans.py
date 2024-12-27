import os
import torch
import pandas as pd
from functions.load_data.load_data import load_data
from functions.load_data.load_mass_with_metadata import load_mass_with_metadata
from functions.optuna.get_val_accuracy import get_val_accuracy
#from functions.optuna.Trans_OptunaTrainer import Trans_OptunaTrainer
from functions.optuna.Trans_Meta_OptunaTrainer import Trans_Meta_OptunaTrainer

# Consideraciones generales
BATCH_SIZE = 5
IMAGE_SIZE = (500, 500)
NUM_CLASSES = 2
EPOCHS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_built() else 
                      'cpu')

print(DEVICE)
os.makedirs("models", exist_ok=True)

# Inicializamos listas y diccionarios
trial_results_list = []
best_histories = {}
combined_trial_results = []

datasets = [
    ('datasets/mass_prop/mass_prop_info.csv', 'prop'),
    ('datasets/mass_dist/mass_dist_info.csv', 'dist')
 ]

for dataset_path, dataset_name in datasets:
    # Transformer sin metadatos
    # train_loader, val_loader, test_loader = load_data(IMAGE_SIZE, BATCH_SIZE, dataset_path)
    # full_model_name = f"Trans_{dataset_name}"
    # print(full_model_name)
    # Trans_trainer = Trans_OptunaTrainer(train_loader, val_loader, EPOCHS, DEVICE, 'mass', dataset_name)
    # Trans_study = Trans_trainer.run_study(n_trials=16)
    # accuracy = get_val_accuracy(test_loader, Trans_trainer.best_history)
    # Trans_trainer.best_history['accuracy'] = accuracy
    # print(f"Test Accuracy of {full_model_name} model: {accuracy:.2f}%")
    # best_histories[full_model_name] = Trans_trainer.best_history
    # trial_results_list.append(pd.DataFrame(Trans_trainer.trial_results))

    # Transformer con metadatos
    train_loader, val_loader, test_loader = load_data(IMAGE_SIZE, BATCH_SIZE, dataset_path)
    full_model_name = f"Trans_Meta_{dataset_name}"
    print(full_model_name)
    Trans_Meta_trainer = Trans_Meta_OptunaTrainer(train_loader, val_loader, EPOCHS, DEVICE, 'mass', dataset_name)
    Trans_Meta_study = Trans_Meta_trainer.run_study(n_trials=16)
    accuracy = get_val_accuracy(test_loader, Trans_Meta_trainer.best_history)
    Trans_Meta_trainer.best_history['accuracy'] = accuracy
    print(f"Test Accuracy of {full_model_name} model: {accuracy:.2f}%")
    best_histories[full_model_name] = Trans_Meta_trainer.best_history
    trial_results_list.append(pd.DataFrame(Trans_Meta_trainer.trial_results))


# Obtenemos el modelo con mejor pérdida en validacion
best_model_name = min(best_histories, key=lambda x: best_histories[x]["val_loss"])
best_best_history = best_histories[best_model_name]

# Combinamos los resultados de los trials
combined_trial_results = pd.concat(trial_results_list, ignore_index=True)

# Mejor modelo con su historia
print(f"The best model is {best_model_name} with validation loss {best_best_history['val_loss']} and accuracy {best_best_history['accuracy']:.2f}%")

# Guardamos el modelo y su historia, así como los resultados del resto de los intentos
torch.save(best_histories, "models/best_histories_mass_trans.pth")
torch.save(best_best_history, "models/best_model_mass_trans.pth")
combined_trial_results.to_csv("models/history_mass_trans.csv", index=False)
