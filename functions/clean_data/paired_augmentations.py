import os
import random
from PIL import Image, ImageOps
import pandas as pd
from pandas import Series

def paired_augmentations(row: Series, new_index: int, output_dir: str, metadata: list) -> Series:
    """
    Genera distorsiones pareadas en las imágenes indicadas en la fila y las guarda
    Inputs:
        row - Fila a procesar
        new_index - Número identificador de la nueva imagen
        output_dir - Ruta de salida
    Outputs:
        Fila con la información de las imágenes procesadas    
    """
    # Abrimos las imagenes para su procedimiento
    with Image.open(row["ROI_path"]) as ROI, \
         Image.open(row["image_path"]) as image:
        
        # Rotación (-30°, +30°)
        angle = random.uniform(-30, 30)
        ROI = ROI.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=0)
        image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=0)
		
        # Giro horizontal
        if random.random() > 0.5:
            ROI = ImageOps.mirror(ROI)
            image = ImageOps.mirror(image)
        
        new_index = f"{new_index:05d}"
        # Guardamos las imágenes y creamos la nueva fila
        ROI_path = os.path.join(output_dir, "ROI", "ROI_" + new_index + ".png")
        image_path = os.path.join(output_dir, "image", "image_" + new_index + ".png")
        ROI.save(ROI_path, format="PNG")
        image.save(image_path, format="PNG")
        new_row = {"index": new_index,
           "label": row["label"],
           "ROI_path": ROI_path,
           "image_path": image_path,
           "original": row["index"]}
		
        # Añadimos el resto de columnas a la nueva fila
        for column in metadata:
            new_row[column] = row[column]
        
        return(new_row)


