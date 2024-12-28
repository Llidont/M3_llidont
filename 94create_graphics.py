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

# Gráfica de pérdidas y accuracies lado a lado
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# Gráfica de pérdidas
for i, model_name in enumerate(models):
    axes[0].plot(val_losses[model_name], label=model_name, color=colors[i % len(colors)])
axes[0].set_title('Validation Losses')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Gráfica de accuracies
for i, model_name in enumerate(models):
    axes[1].plot(val_accuracies[model_name], label=model_name, color=colors[i % len(colors)])
axes[1].set_title('Validation Accuracies')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True)

# Guardamos el gráfico
plt.tight_layout()
plt.savefig('examples/all_metrics.png', bbox_inches='tight', dpi=200)

