import os
import pandas as pd
from functions.clean_data.paired_augmentations import paired_augmentations
from functions.clean_data.display_png_images import display_png_images

clean_dataset = pd.read_csv(os.path.join('datasets', 'calc_prop', 'calc_prop_info.csv'))
output_dir =  os.path.join('datasets', 'calc_dist')
os.makedirs(os.path.join(output_dir), exist_ok=True)
os.makedirs(os.path.join(output_dir, "ROI"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "image"), exist_ok=True)

augmented_dataset = []
metadata = ["calc_type",
    "calc_distribution",
    "subtlety",
    "breast_density",
    "shape_vertical",
    "shape_horizontal"]

for i in range(0,len(clean_dataset)*2):
    random_row = clean_dataset.sample(1).iloc[0]
    new_row = paired_augmentations(random_row, i, output_dir, metadata)
    augmented_dataset.append(new_row)

# Guardamos el dataframe
print(f"{len(clean_dataset)*2} imagenes generadas correctamente.")
augmented_dataset_df = pd.DataFrame(augmented_dataset)
augmented_dataset_df.to_csv(os.path.join('datasets', 'calc_dist', 'calc_dist_info.csv'), index=False)
print(f"Dataset guardado correctamente.")

display_png_images(augmented_dataset_df, 'ROI_path', 5, os.path.join('examples', 'calc_dist_image.jpeg'))
display_png_images(augmented_dataset_df, 'image_path', 5, os.path.join('examples', 'calc_dist_roi.jpeg'))
