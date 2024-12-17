# Cargamos las funciones
import os
import pandas as pd
import numpy as np
import re
import zipfile
from functions.clean_data.download_dataset_CBIS_DDSM import download_dataset_CBIS_DDSM
from functions.clean_data.display_images import display_images
from functions.clean_data.load_images_with_dimensions import load_images_with_dimensions
from functions.clean_data.get_shape_and_bounds import get_shape_and_bounds
from functions.clean_data.process_and_display_images import process_and_display_images
from functions.clean_data.crop_and_resize import crop_and_resize

# Carga y limpieza de datos -------------------------------------------
try:
    di_data = pd.read_csv(os.path.join('datasets', 'archive', 'csv', 'dicom_info.csv'))
else:
    download_dataset_CBIS_DDSM()

full_mammo_images_data = di_data[di_data['SeriesDescription'] == 'full mammogram images'][['image_path', 'PatientBirthDate', 'PatientSex', 'StudyTime', 'Laterality', 'Modality']]
ROI_mask_images_data = di_data[di_data['SeriesDescription'] == 'ROI mask images'][['image_path', 'PatientBirthDate', 'PatientSex', 'StudyTime', 'Laterality', 'Modality']]

ROI_mask_images_data.isna().sum()
ROI_mask_images_data.shape
full_mammo_images_data.isna().sum()
full_mammo_images_data.shape

full_mammo_images = di_data[di_data['SeriesDescription'] == 'full mammogram images']['image_path']
ROI_mask_images = di_data[di_data['SeriesDescription'] == 'ROI mask images']['image_path']

# Amoldamos los paths a nuestra organización de directorios
# Aún no arreglamos el os.path.separator
full_mammo_images = full_mammo_images.str.replace('CBIS-DDSM/jpeg', 'datasets/archive/jpeg', regex=False)
ROI_mask_images = ROI_mask_images.str.replace('CBIS-DDSM/jpeg', 'datasets/archive/jpeg', regex=False)

# Cargamos los datasets
calc_test = pd.read_csv(os.path.join('datasets', 'archive', 'csv', 'calc_case_description_test_set.csv'))
calc_train = pd.read_csv(os.path.join('datasets', 'archive', 'csv', 'calc_case_description_train_set.csv'))


# Homogeneizamos los nombres de las columnas
calc_test = calc_test.rename(columns=lambda x: re.sub(r' ', '_', x))
calc_train = calc_train.rename(columns=lambda x: re.sub(r' ', '_', x))

calc_combined = pd.concat([calc_test, calc_train], ignore_index=True)


def get_id(path):
    ''' Para no depender del path inicial, cogemos el id después de jpeg'''
    start_index = path.find("jpeg/")
    after_jpeg = path[start_index + len("jpeg/"):]
    return after_jpeg.split('/')[0]

# Aplicamos las funciones
full_mammo_images_dict = dict(zip(map(get_id, full_mammo_images), full_mammo_images))
ROI_mask_images_dict = dict(zip(map(get_id, ROI_mask_images), ROI_mask_images))

columns_mapping = {
    "ROI_mask_file_path": ROI_mask_images_dict,
    "image_file_path": full_mammo_images_dict
}

for i in range(len(calc_combined)):
    for col_name, replacement_dict in columns_mapping.items():
        img_name = calc_combined.loc[i, col_name].split("/")[2]
        # Una vez ya tenemos el correcto, arreglamos el os.path.separator
        calc_combined.at[i, col_name] = replacement_dict.get(img_name, calc_combined.loc[i, col_name])

# Comprobamos que sea correcto, eliminamos las imágenes que no tenemos
calc_combined = calc_combined.loc[
    (calc_combined['ROI_mask_file_path'].str.startswith('datasets')) &
    (calc_combined['image_file_path'].str.startswith('datasets'))]
calc_combined=calc_combined.reset_index(drop=True)
print(calc_combined.loc[0, "ROI_mask_file_path"])


guardarimagenes = True
if(guardarimagenes):
    # Display images
    os.makedirs("examples", exist_ok=True)
    print('Mostramos imágenes:')
    display_images(calc_combined, 'image_file_path', 5,
                    os.path.join('examples', 'calc_clean_image.jpeg'))
    display_images(calc_combined, 'ROI_mask_file_path', 5,
                    os.path.join('examples', 'calc_clean_roi.jpeg'))


# Ahora buscamos cuales son las dimensiones para poder cargarlas y procesarlas
printdimensions = False
if(printdimensions):
    calc_combined_dims = load_images_with_dimensions(calc_combined)
    # Creamos summaries con la información para decidir dimensiones al procesar
    width_height_summary = calc_combined_dims[['roi_width', 'roi_height', 'full_width', 'full_height']].describe()
    print("Resumen de dimensiones:\n", width_height_summary)
    print("\n")
    data_minimums = calc_combined_dims[['roi_width', 'roi_height', 'full_width', 'full_height']].min()
    print("Minimos de las imágenes:\n", data_minimums)
    print("\n")
    data_maxims = calc_combined_dims[['roi_width', 'roi_height', 'full_width', 'full_height']].max()
    print("Máximos de las imágenes:\n", data_maxims)
    print("Rows:", calc_combined_dims.shape[0])


# Conseguimos la medida del contorno de los ROI
calc_combined[["shape_vertical", "shape_horizontal", 
               "top", "bottom", 
               "left", "right"]] = calc_combined["ROI_mask_file_path"].apply(
    lambda path: pd.Series(get_shape_and_bounds(path))
)

# Compute statistics
stats = {
    "max_vertical": max(calc_combined["shape_vertical"]),
    "min_vertical": min(calc_combined["shape_vertical"]),
    "max_horizontal": max(calc_combined["shape_horizontal"]),
    "min_horizontal": min(calc_combined["shape_horizontal"])
}

required_dim = (500,500)
max_bound = 3808

# Mostramos las imágenes procesadas
imagenesprocesadas=True
if(imagenesprocesadas):
    os.makedirs("examples", exist_ok=True)
    process_and_display_images(calc_combined, 5, required_dim, 'calc_clean', max_bounds=3808)


procesar_imagenes=True
if(procesar_imagenes):
    os.makedirs(os.path.join('datasets', 'calc_clean', 'ROI'), exist_ok=True)
    os.makedirs(os.path.join('datasets', 'calc_clean', 'image'), exist_ok=True)
    labels = []
    for i, row in calc_combined.iterrows():
        image_index = f"_{i:04d}"
        mask_path = row['ROI_mask_file_path']
        image_path = row['image_file_path']
        bounds = row[["top", "bottom", "left", "right"]]
        final_mask, final_image = crop_and_resize(mask_path, required_dim, bounds, image_path)
        
        # Save the cropped images
        final_mask.save(f"datasets/calc_clean/ROI/ROI" + image_index + ".png")
        final_image.save(f"datasets/calc_clean/image/image" + image_index + ".png")
        # Añadimos al diccionario
        labels.append({"index": image_index, "label": row["pathology"],
                       "ROI_path": "datasets/calc_clean/ROI/" + "ROI" + image_index + ".png",
                       "image_path": "datasets/calc_clean/image/" + "image" + image_index + ".png",
                       "calc_type": row["calc_type"],
                       "calc_distribution": row["calc_distribution"],
                       "subtlety": row["subtlety"],
                       "breast_density": row["breast_density"],
                       "shape_vertical": row["shape_vertical"],
                       "shape_horizontal": row["shape_horizontal"]                       
                       })
    # Guardamos el dataframe
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv("datasets/calc_clean/calc_clean_info.csv", index=False)
    print("Imagenes procesadas correctamente.")


