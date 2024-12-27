import os
import torch
import matplotlib.pyplot as plt

# Define datasets
best_models = [
    ("best_model_calc_linear.pth", "calc_linear"),
    ("best_model_mass_linear.pth", "mass_linear"),
    ("best_model_calc_cnn.pth", "calc_cnn"),
    ("best_model_mass_cnn.pth", "mass_cnn"),
]

val_losses = {}
val_accuracies = {}

# Iteramos en cada modelo
for models_path, models_name in best_models:
    model_data = torch.load(os.path.join('models', models_path), weights_only=True)
    # Extraemos las métricas
    val_losses[models_name] = model_data['val_losses']
    val_accuracies[models_name] = model_data['val_accuracies']

# Definimos colores para los modelos
colors = ['blue', 'green', 'red', 'orange']
models = list(val_losses.keys())

# Gráfica de pérdidas
plt.figure(figsize=(10, 6))
for i, model_name in enumerate(models):
    plt.plot(val_losses[model_name], label=model_name, color=colors[i % len(colors)])
plt.title('Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('examples/all_val_losses.png', bbox_inches='tight', dpi=200)

# Gráfica de accuracies
plt.figure(figsize=(10, 6))
for i, model_name in enumerate(models):
    plt.plot(val_accuracies[model_name], label=model_name, color=colors[i % len(colors)])
plt.title('Validation Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('examples/all_val_accuracies.png', bbox_inches='tight', dpi=200)
