#: Carga de librerías --------------------------------------------------
import os

file_path = 'possible_paths.txt'

with open(file_path, 'r') as f:
    for p in f.readlines():
        p = p.strip()
        if os.path.isdir(p):
            print(f"Ruta actual: {p}")
            path = p
            os.chdir(path)
            break
    else:
        print("No valid paths found.")

# Cargamos el resto de funciones
from funciones_preprocesado_241123 import *

getmemory("Inicio")

# Carga y limpieza de datos -------------------------------------------
# Obtenemos las columnas que nos interesan
di_data = pd.read_csv(os.path.join(path, 'archive/csv/dicom_info.csv'))
cropped_images = di_data[di_data['SeriesDescription'] == 'cropped images']['image_path']
full_mammo_images = di_data[di_data['SeriesDescription'] == 'full mammogram images']['image_path']
ROI_mask_images = di_data[di_data['SeriesDescription'] == 'ROI mask images']['image_path']

# Amoldamos los paths a nuestra organización de directorios
cropped_images = cropped_images.str.replace('CBIS-DDSM/jpeg', path + '/archive/jpeg', regex=False)
full_mammo_images = full_mammo_images.str.replace('CBIS-DDSM/jpeg', path + '/archive/jpeg', regex=False)
ROI_mask_images = ROI_mask_images.str.replace('CBIS-DDSM/jpeg', path + '/archive/jpeg', regex=False)

# Cargamos los datasets
mass_test = pd.read_csv(os.path.join(path, 'archive/csv/mass_case_description_test_set.csv'))
mass_train = pd.read_csv(os.path.join(path, 'archive/csv/mass_case_description_train_set.csv'))

# Homogeneizamos los nombres de las columnas
mass_test = mass_test.rename(columns=lambda x: re.sub(r' ', '_', x))
mass_train = mass_train.rename(columns=lambda x: re.sub(r' ', '_', x))

mass_combined = pd.concat([mass_test, mass_train], ignore_index=True)

# Aplicamos las funciones
full_mammo_images_dict = dict(zip(map(get_id, full_mammo_images), full_mammo_images))
cropped_images_dict = dict(zip(map(get_id, cropped_images), cropped_images))
ROI_mask_images_dict = dict(zip(map(get_id, ROI_mask_images), ROI_mask_images))

columns_mapping = {
    "ROI_mask_file_path": ROI_mask_images_dict,
    "image_file_path": full_mammo_images_dict,
    "cropped_image_file_path": cropped_images_dict
}

for i in range(len(mass_combined)):
    for col_name, replacement_dict in columns_mapping.items():
        img_name = mass_combined.loc[i, col_name].split("/")[2]
        mass_combined.at[i, col_name] = replacement_dict.get(img_name, mass_combined.loc[i, col_name])


# Comprobamos que sea correcto
print(mass_combined.loc[0, "ROI_mask_file_path"])

getmemory("Antes borrar")

# Borramos variables que no se vayan a usar
del di_data, cropped_images, full_mammo_images, ROI_mask_images
    
getmemory("Después borrar")

guardarimagenes = False
if(guardarimagenes):
    # Display images
    print('Mostramos imágenes:')
    display_images(mass_combined, 'image_file_path', 5, 'image_file_path_example.jpeg')
    display_images(mass_combined, 'ROI_mask_file_path', 5, 'ROI_mask_file_path_example.jpeg')


# Ahora buscamos cuales son las dimensiones para poder cargarlas y procesarlas
printdimensions = False
if(printdimensions):
    mass_combined_dims = load_images_with_dimensions(mass_combined)
    # Creamos summaries con la información para decidir dimensiones al procesar
    width_height_summary = mass_combined_dims[['roi_width', 'roi_height', 'full_width', 'full_height', 'cropped_width', 'cropped_height']].describe()
    print("Resumen de dimensiones:\n", width_height_summary)
    print("\n")
    data_minimums = mass_combined_dims[['roi_width', 'roi_height', 'full_width', 'full_height', 'cropped_width', 'cropped_height']].min()
    print("Minimos de las imágenes:\n", data_minimums)
    print("\n")
    data_maxims = mass_combined_dims[['roi_width', 'roi_height', 'full_width', 'full_height', 'cropped_width', 'cropped_height']].max()
    print("Máximos de las imágenes:\n", data_maxims)
    print("Rows:", mass_combined_dims.shape[0])


# Mostramos las imágenes procesadas
imagenesprocesadas=True
if(imagenesprocesadas):
    print('ROI Masks:')
    process_and_display_images(mass_combined, 'ROI_mask_file_path', 5, 'roi_tensor_example.jpeg')
    process_and_display_images(mass_combined, 'image_file_path', 5, 'full_tensor_example.jpeg')
    process_and_display_images(mass_combined, 'cropped_image_file_path', 5, 'cropped_tensor_example.jpeg')

croppingerror = check_cropping_errors(mass_combined.copy())

getmemory("Después errores")

# Obtenemos los datasets de las imágenes correctas, negras o cortadas
correctas = [index for index, value in enumerate(croppingerror) if value == 0]
negras = [index for index, value in enumerate(croppingerror) if value == 1]
cortadas = [index for index, value in enumerate(croppingerror) if value == 2]

imagenes_correctas = mass_combined.iloc[correctas]
imagenes_negras = mass_combined.iloc[negras]
imagenes_cortadas = mass_combined.iloc[cortadas]

display_images(imagenes_negras, 'ROI_mask_file_path', 5, 'ejemplo_imagenes_negras.jpeg')
display_images(imagenes_cortadas, 'ROI_mask_file_path', 5, 'ejemplo_imagenes_cortadas.jpeg')

procesar_imagenes=True
if(procesar_imagenes):
    labels = []
    os.makedirs(os.path.join("clean", "ROI"), exist_ok=True)
    os.makedirs(os.path.join("clean", "image"), exist_ok=True)
    os.makedirs(os.path.join("clean", "cropped"), exist_ok=True)
    
    for index, row in imagenes_correctas.reset_index(drop=True).iterrows():
        # Creamos un identificador para cada imagen
        image_index = f"_{index:05d}"
        
        # Procesamos las imágenes y las guardamos
        roi_tensor = process_image_to_tensor(row["ROI_mask_file_path"], target_size=(2000, 3000))
        save_tensor_as_png(roi_tensor, os.path.join("clean", "ROI"), "ROI" + image_index)
        image_tensor = process_image_to_tensor(row["image_file_path"], target_size=(2000, 3000))
        save_tensor_as_png(image_tensor, os.path.join("clean", "image"), "image" + image_index)
        cropped_tensor = process_image_to_tensor(row["cropped_image_file_path"], target_size=(500, 500))
        save_tensor_as_png(cropped_tensor, os.path.join("clean", "cropped"), "cropped" + image_index)
        
        # Creamos diccionario
        labels.append({"index": image_index, "label": row["pathology"],
                       "ROI_path": os.path.join("clean", "ROI", "ROI" + image_index + ".png"),
                       "image_path": os.path.join("clean", "image", "image" + image_index + ".png"),
                       "cropped_path": os.path.join("clean", "cropped", "cropped" + image_index + ".png")})
    
    # Guardamos el dataframe
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv("clean_dataset_info.csv", index=False)
    print("Imagenes procesadas correctamente.")

getmemory("Fin")

