# Cargamos las funciones
import os
import pandas as pd
import numpy as np
import re
from functions.clean_data.download_dataset_CBIS_DDSM import download_dataset_CBIS_DDSM
from functions.clean_data.display_images import display_images
from functions.clean_data.load_images_with_dimensions import load_images_with_dimensions
from functions.clean_data.get_shape_and_bounds import get_shape_and_bounds
from functions.clean_data.process_and_display_images import process_and_display_images
from functions.clean_data.crop_and_resize import crop_and_resize

# Carga y limpieza de datos -------------------------------------------

#df = pd.read_csv('datasets/calc_prop/calc_prop_info.csv')
#df = pd.read_csv('datasets/mass_prop/mass_prop_info.csv')
df = pd.read_csv('datasets/calc_clean/calc_clean_info.csv')
#df = pd.read_csv('datasets/mass_clean/mass_clean_info.csv')

# Ahora buscamos cuales son las dimensiones para poder cargarlas y procesarlas

dims = load_images_with_dimensions(df, 'ROI_path', 'image_path')
# Creamos summaries con la información para decidir dimensiones al procesar
width_height_summary = dims[['roi_width', 'roi_height', 'full_width', 'full_height']].describe()
print("Resumen de dimensiones:\n", width_height_summary)
print("\n")
data_minimums = dims[['roi_width', 'roi_height', 'full_width', 'full_height']].min()
print("Minimos de las imágenes:\n", data_minimums)
print("\n")
data_maxims = dims[['roi_width', 'roi_height', 'full_width', 'full_height']].max()
print("Máximos de las imágenes:\n", data_maxims)
print("Rows:", dims.shape[0])

target_columns = ['roi_width', 'roi_height', 'full_width', 'full_height']
diff_from_500 = df[target_columns].applymap(lambda x: x != 500)
df[diff_from_500.any(axis=1)][['roi_width', 'roi_height', 'full_width', 'full_height']]
