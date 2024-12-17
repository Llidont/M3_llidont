import numpy as np
from PIL import Image
from scipy.ndimage import zoom as scipyzoom
from typing import Tuple, TypedDict, Union
from functions.clean_data.get_shape_and_bounds import get_shape_and_bounds

class BoundsDict(TypedDict):
    top: int
    bottom: int
    left: int
    right: int

def crop_and_resize(mask_path: str, required_dim: Tuple[int, int],
                    bounds: BoundsDict, image_path: str = None,
                    all_prop: bool = False, max_bounds: int =0
                    ) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
    '''
    Dadas las rutas de la máscara y posiblemente de la imagen general, 
    utiliza los límites proporcionados para recortar, redimensionar y 
    rellenar según sea necesario.
    Inputs:
        mask_path - Ruta de la imagen a redimensionar
        required_dim - Tamaño deseado final
        bounds - Diccionario con los límites de la región de interés
        image_path - Si se incluye, ruta de la imagen que deberá
            transformarse como "mask_path"
        all_prop - Si todas las imágenes deben ampliar la región de
            interés. Falso por defecto.
        max_bounds - En caso de querer redimensionar proporcionalmente
            a un cuadrado de esas dimensiones.
    Outputs:
        Imagen(es) procesadas
    '''
    mask = np.array(Image.open(mask_path))
    if image_path is not None:
        image = np.array(Image.open(image_path))
    
    req_h, req_w = required_dim
    top, bottom = bounds[["top", "bottom"]]
    left, right = bounds[["left", "right"]]
    current_h = bottom - top
    current_w = right - left
    
    # En caso de tener que redimensionar
    if max_bounds!=0 or all_prop or current_h>req_h or current_w>req_w:
        if max_bounds!=0:
            aspect_ratio = min(req_h/(max_bounds-50), req_w/(max_bounds-50))
            #print("AR_maxbounds:", aspect_ratio, type(aspect_ratio))
        else:
            aspect_ratio = min((req_h-50)/current_h, (req_w-50)/current_w)
            #print("AR:", aspect_ratio, type(aspect_ratio))
        mask = scipyzoom(mask, aspect_ratio, order=3)
        top, bottom, left, right = (int(aspect_ratio*bound) for bound in (top, bottom, left, right))
        if image_path is not None:
            image = scipyzoom(image, aspect_ratio, order=3)
    
    current_h = bottom - top
    current_w = right - left
    #print("Current:", current_h, current_w)
    
    # Obtenemos las dimensiones de crop y padding
    missing_h = max(0,(req_h - current_h)//2)
    missing_w = max(0,(req_w - current_w)//2)
    c_top = max(0 , top - missing_h)
    pad_top = abs(min(0, top - missing_h))
    c_bot = min(bottom + missing_h, mask.shape[0]) 
    pad_bot = req_h - (c_bot - c_top) - pad_top
    c_left = max(0, left - missing_w)
    pad_left = abs(min(0, left - missing_w))
    c_right = min(right + missing_w, mask.shape[1]) 
    pad_right = req_w  - (c_right - c_left) - pad_left
    #print("Added:", missing_h, missing_w)
    #print("Crops:", c_top, c_bot, c_left, c_right)
    
    # Recortamos las imágenes
    cropped_mask = mask[c_top:c_bot, c_left:c_right]
    if image_path is not None:
        cropped_image = image[c_top:c_bot, c_left:c_right]
    
    # Aplicamos el padding
    #print("Padding:", pad_top, pad_bot, pad_left, pad_right)
    padded_mask = np.pad(
        cropped_mask,
        ((pad_top, pad_bot), (pad_left, pad_right)),
        mode='constant'
    )
    if image_path is not None:
        padded_image = np.pad(
            cropped_image,
            ((pad_top, pad_bot), (pad_left, pad_right)),
            mode='constant'
            )
        return Image.fromarray(padded_mask), Image.fromarray(padded_image)
    
    return Image.fromarray(padded_mask)
