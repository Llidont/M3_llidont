#: Carga de librerías --------------------------------------------------
# A falta de requirements
# pip install pandas torch torchvision Pillow matplotlib numpy psutil
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageChops
import random

def display_png_images(data, column, number, output_file='default.png'):
    '''Displays images listed in a column of a DataFrame and saves the combined output as a PNG.'''
    fig, axes = plt.subplots(1, number, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5)
    
    for i in range(number):
        image_path = data.iloc[i][column]
        
        print(f"Trying to read: {image_path}")
        
        if image_path.endswith(".png"):
            try:
                if os.path.exists(image_path):
                    # Open the PNG image
                    image = Image.open(image_path)
                    pathology = data.iloc[i]["label"]
                    axes[i].imshow(image)
                    axes[i].set_title(pathology)
                    axes[i].axis("off")
                else:
                    print(f"The PNG image does not exist: {image_path}")
                    
            except Exception as e:
                print(f"Error reading the PNG image: {image_path} - {e}")
        else:
            print(f"Unsupported file extension: {image_path}")
    
    # Save the figure as a PNG file
    fig.savefig(output_file, format='png', bbox_inches="tight")

def crop_or_pad_image(image, target_size=(2000, 3000)):
    width, height = image.size
    target_width, target_height = target_size
    
    # Crop si la imagen es demasiado grande
    left = max((width - target_width) // 2, 0)
    top = max((height - target_height) // 2, 0)
    right = min((width + target_width) // 2, width)
    bottom = min((height + target_height) // 2, height)
    image = image.crop((left, top, right, bottom))
    
    # Convertir a imagen gris
    if image.mode != "L":
        image = image.convert("L")
    
    # Padding si la imagen es demasidado pequeña
    delta_width = max(target_width - image.width, 0)
    delta_height = max(target_height - image.height, 0)
    left_padding = delta_width // 2
    top_padding = delta_height // 2
    
    padded_image = Image.new("L", target_size)
    padded_image.paste(image, (left_padding, top_padding))
    
    return padded_image


def is_all_black_or_margin_non_black(tensor, margin=10):
    """Check if the tensor is completely black or has non-black pixels in the margin."""
    if torch.all(torch.abs(tensor) < 1e-6):
        return 1
    
    top_margin = tensor[:margin, :]
    bottom_margin = tensor[-margin:, :]
    left_margin = tensor[:, :margin]
    right_margin = tensor[:, -margin:]
    
    if torch.any(top_margin > 0) or torch.any(bottom_margin > 0) or \
       torch.any(left_margin > 0) or torch.any(right_margin > 0):
        return 2
    
    return 0

def paired_augmentations(row, new_index, output_dir):
    """
    Genera distorsiones pareadas en las imágenes de la fila
    """
    import os
    # Open images
    with Image.open(row["ROI_path"]) as ROI, \
         Image.open(row["image_path"]) as image, \
         Image.open(row["cropped_path"]) as cropped:
        # Set random seed for reproducibility across augmentations
        seed = random.randint(0, 2**32 - 1)
        random.seed(seed)
        
        # Rotación (-30°, +30°)
        angle = random.uniform(-30, 30)
        ROI = ROI.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=0)
        image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=0)
        cropped = cropped.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=0)
        
        # Giro horizontal
        if random.random() > 0.5:
            ROI = ImageOps.mirror(ROI)
            image = ImageOps.mirror(image)
            cropped = ImageOps.mirror(cropped)
        
        # Traslación
        dx, dy = random.randint(-200, 200), random.randint(-300, 300)
        ROI = ROI.transform((2000, 3000), Image.AFFINE, (1, 0, dx, 0, 1, dy),
                        resample=Image.BICUBIC, fillcolor=0)
        image = image.transform((2000, 3000), Image.AFFINE, (1, 0, dx, 0, 1, dy),
                        resample=Image.BICUBIC, fillcolor=0)
        
        # Arreglamos tamaños?
        ROI = crop_or_pad_image(ROI)
        image = crop_or_pad_image(image)
        cropped= crop_or_pad_image(cropped, (500,500))
        
        ROI_array = np.array(ROI, dtype=np.float32) / 255.0
        ROI_tensor =  torch.tensor(ROI_array)
        
        if(is_all_black_or_margin_non_black(ROI_tensor, margin=10)==0):
            # Save augmented images
            ROI_path = os.path.join(output_dir, "ROI", "ROI" + str(new_index) + ".png")
            image_path = os.path.join(output_dir, "image", "image" + str(new_index) + ".png")
            cropped_path = os.path.join(output_dir, "cropped", "cropped" + str(new_index) + ".png")
            ROI.save(ROI_path, format="PNG")
            image.save(image_path, format="PNG")
            cropped.save(cropped_path, format="PNG")
            print(f"Distorsion de {new_index} completada")
            new_row = {"index": new_index,
                       "label": row["label"],
                       "ROI_path": ROI_path,
                       "image_path": image_path,
                       "cropped_path": cropped_path,
                       "original": row["index"]}
            return(new_row)
        else:
            print(f"Distorsion de {new_index} fallida.")
            return None


