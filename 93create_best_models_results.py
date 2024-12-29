import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

column_rename_dict = {
    "dropout_rate": "dropout",
    "val_loss": "val loss",
    "accuracy": "test acc",
    "learning_rate": "learn rate",
    "num_neurons": "n neurons",
    "val_accuracy": "val acc",
    "num_layers": "n layers",
    "d_model": "model dim",
    "num_heads": "n heads",
    "d_ff": "ff dim",
    "layer_filter": "layer filter",
    "kernel_size": "kernel",
    "stride": "stride"
}

column_order = ['model', 'type', 'dataset',
                "learn rate", "dropout",
                 "n neurons", "stride", "kernel",
                'val loss', 'val acc', 'test acc']

best_models = [
    ("best_histories_calc_linear.pth", "calc_linear"),
    ("best_histories_mass_linear.pth", "mass_linear"),
    ("best_histories_calc_cnn.pth", "calc_cnn"),
    ("best_histories_mass_cnn.pth", "mass_cnn"),
]

# Iteramos en cada modelo
for models_path, models_name in best_models:
    model_data = torch.load(os.path.join('models', models_path), weights_only=True)
    models_info_list = []
    for model_name, model_info in model_data.items():
        hyperparameters = pd.DataFrame([model_info['hyperparameters']])
        hyperparameters['model'] = model_info['model']
        hyperparameters['type'] = model_info['type']
        hyperparameters['dataset'] = model_info['dataset']
        hyperparameters['val_loss'] = model_info['val_loss']
        val_losses = model_info['val_losses']
        best_model_position = np.argmin(val_losses)
        hyperparameters['val_accuracy'] = model_info['val_accuracies'][best_model_position]
        hyperparameters['accuracy'] = model_info['accuracy']
        models_info_list.append(hyperparameters)
    
    dataset = pd.concat(models_info_list, ignore_index=True)
    
    dataset = dataset.sort_values(by='val_accuracy', ascending=False).reset_index(drop=True)
    dataset['dropout_rate'] = dataset['dropout_rate'].round(6)
    dataset['learning_rate'] = dataset['learning_rate'].round(6)
    dataset['val_loss'] = dataset['val_loss'].round(3)
    dataset['val_accuracy'] = dataset['val_accuracy'].round(3)
    dataset['num_neurons'] = dataset['num_neurons'].fillna(0).astype('Int64').astype(str)
    dataset['num_neurons'] = dataset['num_neurons'].replace('0', '-')
    dataset['accuracy'] = dataset['accuracy'].round(1).astype(str) + '%'
    dataset = dataset.rename(columns=column_rename_dict)
    
    dataset = dataset[[col for col in column_order if col in dataset.columns]]

    # Dimensión dinámica
    rows, cols = dataset.shape
    cell_width = 1.1
    cell_height = 0.2
    fig_width = cols * cell_width
    fig_height = rows * cell_height

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    col_widths = [0.2] + [0.10] * (len(dataset.columns) - 1)
    tbl = table(ax, dataset, loc='center', colWidths=col_widths)
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.2)
    
    plt.savefig(f'examples/best_histories_{models_name}.png', bbox_inches='tight', dpi=200)
