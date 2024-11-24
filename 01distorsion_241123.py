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
from funciones_distorsion_241123 import *

clean_dataset = pd.read_csv('clean_dataset_info.csv')
output_dir =  "distorted"
os.makedirs(os.path.join(output_dir, "ROI"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "image"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "cropped"), exist_ok=True)

augmented_dataset = []

new_index = 1
for i in range(0,10):
    random_row = clean_dataset.sample(1).iloc[0]
    new_row = paired_augmentations(random_row, f"_{new_index:05d}", output_dir)
    if new_row is None:
        continue
    augmented_dataset.append(new_row)
    new_index += 1


# Guardamos el dataframe
augmented_dataset_df = pd.DataFrame(augmented_dataset)
augmented_dataset_df.to_csv("distorted_dataset_info.csv", index=False)
print(f"{new_index} imagenes generadas correctamente.")


display_png_images(augmented_dataset_df, 'ROI_path', 5, 'ejemplo_distorted_ROI.jpeg')
display_png_images(augmented_dataset_df, 'image_path', 5, 'ejemplo_distorted_image.jpeg')
display_png_images(augmented_dataset_df, 'cropped_path', 5, 'ejemplo_distorted_cropped.jpeg')
