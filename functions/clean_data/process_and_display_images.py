import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from functions.clean_data.crop_and_resize import crop_and_resize

def process_and_display_images(data, number_of_images, target_size=(500, 500),
                                output_prefix='image', output_path='examples',
                                all_prop=False, max_bounds:int = 0) -> None:
    
    # Number of images to display in each batch (2x2 grid)
    rows_per_batch = 2
    cols_per_batch = 2
    
    for i in range(number_of_images):
        fig, axes = plt.subplots(rows_per_batch, cols_per_batch, figsize=(10, 10))
        axes = axes.flatten()
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        
        # Obtenemos imagen y etiquetado
        mask_path = data.iloc[i]["ROI_mask_file_path"]
        image_path = data.iloc[i]["image_file_path"]
        pathology = data.iloc[i]["pathology"]
        bounds = data.iloc[i][["top", "bottom", "left", "right"]]
        
        # Procesamos las imágenes
        final_mask, final_image = crop_and_resize(mask_path, target_size, bounds,
                                                  image_path, all_prop, max_bounds)
            
        # Convertimos las imágenes a arrays de numpy
        final_mask_array = np.array(final_mask)
        final_image_array = np.array(final_image)
        
        # Y las colocamos en la parrilla
        axes[0].imshow(final_mask_array, cmap="gray")
        axes[0].axis("off")
        
        axes[1].imshow(final_image_array)
        axes[1].axis("off")
        
        # Así como las originales
        axes[2].imshow(Image.open(mask_path), cmap="gray")
        axes[2].axis("off")
        
        axes[3].imshow(Image.open(image_path))
        axes[3].axis("off")
        
        # Colocamos el título en el centro de la figura
        fig.suptitle(f"{pathology}", fontsize=16, ha='center')
        
        # Y se guarda cada ejemplo por separado
        output_file = f"{output_path}/{output_prefix}_{i}.jpeg"
        fig.savefig(output_file, format='jpeg', bbox_inches="tight")
        plt.close(fig)


