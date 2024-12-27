import os
import pandas as pd
from functions.clean_data.display_png_images import display_png_images

datasets = [
    ("mass_dist", "mass_dist_info"),
    ("mass_prop", "mass_prop_info"),
    ("mass_clean", "mass_clean_info"),
    ("calc_dist", "calc_dist_info"),
    ("calc_prop", "calc_prop_info"),
    ("calc_clean", "calc_clean_info"),
]

dataset_base_path = 'datasets'
examples_base_path = 'examples'

# Para cada dataset
for dataset_folder, dataset_file in datasets:
    print(f"Processing dataset: {dataset_folder}")
    
    dataset_path = os.path.join(dataset_base_path, dataset_folder, f"{dataset_file}.csv")
    roi_image_path = os.path.join(examples_base_path, f"{dataset_folder}_roi.jpeg")
    full_image_path = os.path.join(examples_base_path, f"{dataset_folder}_image.jpeg")
    
    dataset_df = pd.read_csv(dataset_path)
    display_png_images(dataset_df, 'ROI_path', 5, roi_image_path)
    display_png_images(dataset_df, 'image_path', 5, full_image_path)
