# Cargamos las funciones
import os
import pandas as pd
import re
from functions.clean_data.download_dataset_CBIS_DDSM import download_dataset_CBIS_DDSM
from functions.clean_data.display_images import display_images
from functions.clean_data.load_images_with_dimensions import load_images_with_dimensions
from functions.clean_data.get_shape_and_bounds import get_shape_and_bounds
from functions.clean_data.process_and_display_images import process_and_display_images
from functions.clean_data.crop_and_resize import crop_and_resize

# Carga y limpieza de datos -------------------------------------------
if not os.path.exists('datasets/archive/csv'):
    print("Descargando el dataset...")
    download_dataset_CBIS_DDSM()
    print("Dataset descargado")

di_data = pd.read_csv('datasets/archive/csv/dicom_info.csv')
mass_test = pd.read_csv('datasets/archive/csv/mass_case_description_test_set.csv')
mass_train = pd.read_csv('datasets/archive/csv/mass_case_description_train_set.csv')

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

# Homogeneizamos los nombres de las columnas
mass_test = mass_test.rename(columns=lambda x: re.sub(r' ', '_', x))
mass_train = mass_train.rename(columns=lambda x: re.sub(r' ', '_', x))

mass_combined = pd.concat([mass_test, mass_train], ignore_index=True)

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

for i in range(len(mass_combined)):
    for col_name, replacement_dict in columns_mapping.items():
        img_name = mass_combined.loc[i, col_name].split("/")[2]
        # Una vez ya tenemos el correcto, arreglamos el os.path.separator
        mass_combined.at[i, col_name] = replacement_dict.get(img_name, mass_combined.loc[i, col_name])

# Comprobamos que sea correcto
print(mass_combined.loc[0, "ROI_mask_file_path"])

guardarimagenes = False
if(guardarimagenes):
    # Comprobación de imágenes
    os.makedirs("examples", exist_ok=True)
    print('Mostramos imágenes:')
    display_images(mass_combined, 'image_file_path', 5,
                    os.path.join('examples', 'mass_example_image.jpeg'))
    display_images(mass_combined, 'ROI_mask_file_path', 5,
                    os.path.join('examples', 'mass_example_roi.jpeg'))

# Ahora buscamos cuales son las dimensiones para poder cargarlas y procesarlas
printdimensions = False
if(printdimensions):
    mass_combined_dims = load_images_with_dimensions(mass_combined)
    # Creamos summaries con la información para decidir dimensiones al procesar
    width_height_summary = mass_combined_dims[['roi_width', 'roi_height', 'full_width', 'full_height']].describe()
    print("Resumen de dimensiones:\n", width_height_summary)
    print("\n")
    data_minimums = mass_combined_dims[['roi_width', 'roi_height', 'full_width', 'full_height']].min()
    print("Minimos de las imágenes:\n", data_minimums)
    print("\n")
    data_maxims = mass_combined_dims[['roi_width', 'roi_height', 'full_width', 'full_height']].max()
    print("Máximos de las imágenes:\n", data_maxims)
    print("Rows:", mass_combined_dims.shape[0])


# Conseguimos la medida del contorno de los ROI
mass_combined[["shape_vertical", "shape_horizontal", 
               "top", "bottom", 
               "left", "right"]] = mass_combined["ROI_mask_file_path"].apply(
    lambda path: pd.Series(get_shape_and_bounds(path))
)

# Obtenemos las medidas máximas y mínimas
stats = {
    "max_vertical": max(mass_combined["shape_vertical"]),
    "min_vertical": min(mass_combined["shape_vertical"]),
    "max_horizontal": max(mass_combined["shape_horizontal"]),
    "min_horizontal": min(mass_combined["shape_horizontal"])
}

required_dim = (500,500)

# Comprobación de imágenes procesadas
imagenesprocesadas=False
if(imagenesprocesadas):
    process_and_display_images(mass_combined, 5, required_dim, 'mass_prop', all_prop=True)

procesar_imagenes=True
if(procesar_imagenes):
    os.makedirs("datasets/mass_prop/ROI", exist_ok=True)
    os.makedirs("datasets/mass_prop/image", exist_ok=True)
    labels = []
    for i, row in mass_combined.iterrows():
        image_index = f"_{i:04d}"
        mask_path = row['ROI_mask_file_path']
        image_path = row['image_file_path']
        bounds = row[["top", "bottom", "left", "right"]]
        final_mask, final_image = crop_and_resize(mask_path, required_dim, bounds, image_path, all_prop=True)
        
        # Guardamos las imágenes procesadas
        final_mask.save(f"datasets/mass_prop/ROI/ROI" + image_index + ".png")
        final_image.save(f"datasets/mass_prop/image/image" + image_index + ".png")
        # Añadimos al diccionario
        labels.append({"index": image_index, "label": row["pathology"],
                       "ROI_path": "datasets/mass_prop/ROI/" + "ROI" + image_index + ".png",
                       "image_path": "datasets/mass_prop/image/" + "image" + image_index + ".png",
                       "mass_shape": row["mass_shape"],
                       "mass_margins": row["mass_margins"],
                       "subtlety": row["subtlety"],
                       "image_view": row["image_view"],
                       "breast_density": row["breast_density"],
                       "shape_vertical": row["shape_vertical"],
                       "shape_horizontal": row["shape_horizontal"]                       
                       })
    # Guardamos el dataframe
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv("datasets/mass_prop/mass_prop_info.csv", index=False)
    print("Imagenes procesadas correctamente.")



