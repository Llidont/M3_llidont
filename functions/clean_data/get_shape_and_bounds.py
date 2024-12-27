import os
import numpy as np
from PIL import Image
from typing import Dict, Union

def get_shape_and_bounds(mask_path: str) -> Dict[str, Union[int, None]]:
    '''
    De un tensor, obtiene la región no negra
    Inputs:
        mask - path de la imagen a procesar
    Outputs:
        Dict - Devuelve los límites y el tamaño de la región de interés
    '''
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"La ruta '{mask_path}' no lleva a ningún archivo.")
    
    mask = np.array(Image.open(mask_path))
    binary_mask = mask > 0
    
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    # En caso de que la imagen sea trivialmente negra
    if not np.any(rows) or not np.any(cols):
        return {
            "shape_vertical": 0,
            "shape_horizontal": 0,
            "top": None,
            "bottom": None,
            "left": None,
            "right": None
        }
    
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    
    shape_vertical = bottom - top + 1
    shape_horizontal = right - left + 1
    
    return {"shape_vertical": shape_vertical, "shape_horizontal": shape_horizontal, 
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right}
