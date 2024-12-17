import pandas as pd
from PIL import Image

def get_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return width, height
    except Exception as e:
        print(f"Error al abrir la imagen {image_path}: {e}")
        return None, None

def load_images_with_dimensions(data, roi_column: str ='ROI_mask_file_path',
                                    full_column: str ='image_file_path'):
    # Inicializamos las columnas
    data['roi_width'], data['roi_height'] = None, None
    data['full_width'], data['full_height'] = None, None
    
    for i in range(len(data)):
        # Dimensiones ROI
        roi_path = data[roi_column].iloc[i]
        roi_width, roi_height = get_image_dimensions(roi_path)
        data.at[i, 'roi_width'] = roi_width
        data.at[i, 'roi_height'] = roi_height
        
        # Dimensiones de la imagen completa
        full_path = data[full_column].iloc[i]
        full_width, full_height = get_image_dimensions(full_path)
        data.at[i, 'full_width'] = full_width
        data.at[i, 'full_height'] = full_height
    
    return data

