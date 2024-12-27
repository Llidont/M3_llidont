import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

column_rename_dict = {
    "dropout_rate": "dropout",
    "val_loss": "val loss",
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
                'val loss', 'val acc']
datasets = [
    ("history_calc_linear.csv", "calc_linear"),
    ("history_mass_linear.csv", "mass_linear"),
    ("history_calc_cnn.csv", "calc_cnn"),
    ("history_mass_cnn.csv", "mass_cnn"),
]


# Para cada dataset
for models_path, models_name in datasets:
    print(f"Processing dataset: {models_name}")
    dataset = pd.read_csv(os.path.join('models', models_path))
    # Si se ha incluido la columna por error, se elimina
    if 'train_loss' in dataset.columns:
        dataset = dataset.drop(columns=['train_loss'])
    dataset = dataset.sort_values(by='val_loss', ascending=True).reset_index(drop=True).head(50)

    for col in dataset.select_dtypes(include=['float64']).columns:
        dataset[col] = dataset[col].round(3)
    
    dataset['num_neurons'] = dataset['num_neurons'].fillna('-')
    dataset['val_accuracy'] = dataset['val_accuracy'] * 100
    dataset['val_accuracy'] = dataset['val_accuracy'].round(1).astype(str) + '%'
    
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
    plt.savefig(f'examples/history_{models_name}.png', bbox_inches='tight', dpi=200)

    