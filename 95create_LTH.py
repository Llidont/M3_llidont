import os
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from functions.build_model.pruning import prune_weights
from functions.load_data.load_data import load_data
from functions.load_data.load_calc_with_metadata import load_calc_with_metadata
from functions.load_data.load_mass_with_metadata import load_mass_with_metadata
from functions.build_model.training import train_model
from functions.build_model.training_with_metadata import train_model_with_metadata
from functions.networks.linear import Simple_Linear
from functions.networks.linear_Meta import Simple_Linear_Meta
from functions.networks.linear import Linear
from functions.networks.linear_Meta import Linear_Meta
from functions.networks.simple_cnn import SimpleCNN
from functions.networks.simple_cnn_Meta import SimpleCNN_Meta

# Define datasets
best_models = [
    ("best_model_calc_linear.pth", "calc_linear"),
    ("best_model_mass_linear.pth", "mass_linear"),
    ("best_model_calc_cnn.pth", "calc_cnn"),
    ("best_model_mass_cnn.pth", "mass_cnn"),
]

model_classes = {
    "Simple_Linear": Simple_Linear,
    "Simple_Linear_Meta": Simple_Linear_Meta,
    "Linear": Linear,
    "Linear_Meta": Linear_Meta,
    "SimpleCNN": SimpleCNN,
    "SimpleCNN_Meta": SimpleCNN_Meta,
    #"Transf_Meta": Transf_Meta,
}

BATCH_SIZE = 5
IMAGE_SIZE = (500, 500)
NUM_CLASSES = 2
EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                'mps' if torch.backends.mps.is_built() else 
                'cpu')

for models_path, models_name in best_models:
    model_data = torch.load(os.path.join('models', models_path), weights_only=True)
    model_name = model_data["model"]
    model_class = model_classes.get(model_name)
    metadata = True if model_name[-4:]=='Meta' else False
    if model_class is None:
        raise ValueError(f"Model '{model_name}' is not defined in model_classes.")
    print(model_data["model"])
    print(model_data["dataset"])
    print(model_data["type"])
    # Pasamos los hiperparámetros excepto los de entrenamiento
    filtered_hyperparameters = {
        k: v for k, v in model_data["hyperparameters"].items() 
        if k not in ["learning_rate"]
    }
    model = model_class(**filtered_hyperparameters).to(DEVICE)
    initial_model = copy.deepcopy(model)

    # Definimos función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_data["hyperparameters"]["learning_rate"])

    metadata = True if model_name[-4:]=='Meta' else False
    print(metadata)

    # Elegimos el dataset correcto
    if model_data["type"] == "calc":
        if model_data["dataset"] == "clean":
            dataset_path = "datasets/calc_clean/calc_clean_info.csv"
        elif model_data["dataset"] == "prop":
            dataset_path = "datasets/calc_prop/calc_prop_info.csv"
        elif model_data["dataset"] == "dist":
            dataset_path = "datasets/calc_dist/calc_dist_info.csv"
        else:
            raise ValueError(f"Unknown dataset: {model_data['dataset']}")
    elif model_data["type"] == "mass":
        if model_data["dataset"] == "clean":
            dataset_path = "datasets/mass_clean/mass_clean_info.csv"
        elif model_data["dataset"] == "prop":
            dataset_path = "datasets/mass_prop/mass_prop_info.csv"
        elif model_data["dataset"] == "dist":
            dataset_path = "datasets/mass_dist/mass_dist_info.csv"
        else:
            raise ValueError(f"Unknown dataset: {model_data['dataset']}")
    

    # Cargamos los datos
    if not metadata:
        train_loader, val_loader, test_loader = load_data(IMAGE_SIZE, BATCH_SIZE, dataset_path)
    elif model_data["type"]=="calc":
        train_loader, val_loader, test_loader = load_calc_with_metadata(IMAGE_SIZE, BATCH_SIZE, dataset_path)
    else:
        train_loader, val_loader, test_loader = load_mass_with_metadata(IMAGE_SIZE, BATCH_SIZE, dataset_path)
    

    # Entrenamos el modelo
    if not metadata:
        train_losses, val_losses, val_accuracies, best_model = train_model(
            model, optimizer, criterion, train_loader,
            val_loader, EPOCHS, DEVICE
        )
    else:
        train_losses, val_losses, val_accuracies, best_model = train_model_with_metadata(
            model, optimizer, criterion, train_loader,
            val_loader, EPOCHS, DEVICE
        )

    # Recortamos el modelo
    prunned_model = prune_weights(initial_model, best_model)

    # Volvemos a entrenar
    if not metadata:
        train_losses_lt, val_losses_lt, val_accuracies_lt, best_model_lt = train_model(
            prunned_model, optimizer, criterion, train_loader,
            val_loader, EPOCHS, DEVICE
        )
    else:
        train_losses_lt, val_losses_lt, val_accuracies_lt, best_model_lt = train_model_with_metadata(
            prunned_model, optimizer, criterion, train_loader,
            val_loader, EPOCHS, DEVICE
        )

    # Creamos el doble gráfico
    plt.figure(figsize=(14, 10))
    plt.suptitle(f'{models_name} - Original vs LTH', fontsize=16)

    # Plot de pérdida
    plt.subplot(2, 1, 1)
    plt.plot(val_losses, label='Original', color='green')
    plt.plot(val_losses_lt, label='LTH', color='red')
    plt.title(f'Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot de validación
    plt.subplot(2, 1, 2)
    plt.plot(val_accuracies, label='Original', color='green')
    plt.plot(val_accuracies_lt, label='LTH', color='red')
    plt.title(f'Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Guardamos el gráfico
    output_path = os.path.join('examples', f'LTH_{models_name}.png')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
