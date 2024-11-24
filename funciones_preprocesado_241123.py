#: Carga de librerías --------------------------------------------------
# A falta de requirements
# pip install pandas torch torchvision Pillow matplotlib pydicom numpy pympler psutil
import os
import re
import gc
import psutil
import sys
import torch
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pydicom
from pydicom.errors import InvalidDicomError
from pympler import asizeof

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def getmemory(point):
    ''' Devolver por pantalla la memoria ocupada por el proceso
        en un momento concreto '''
    pid = os.getpid()
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    print(f"{point} están ocupados {sizeof_fmt(memory_info.rss)}")

def get_id(path):
    ''' Para no depender del path inicial, cogemos el id después de jpeg'''
    start_index = path.find("jpeg/")
    after_jpeg = path[start_index + len("jpeg/"):]
    return after_jpeg.split('/')[0]

def display_images(data, column, number, output_file='default.jpeg'):
    ''' Carga las imágenes que se indiquen en una columna y las junta'''
    fig, axes = plt.subplots(1, number, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5)
    
    for i in range(number):
        image_path = data.iloc[i][column]
        
        print(f"Trying to read: {image_path}")
        if image_path.endswith(".dcm"):
            try:
                dicom_data = pydicom.dcmread(image_path)
                image = dicom_data.pixel_array
                pathology = data.iloc[i]["pathology"]
                axes[i].imshow(image, cmap="gray")
                axes[i].set_title(pathology)
                axes[i].axis("off")
                
            except (InvalidDicomError, FileNotFoundError) as e:
                print(f"Error en la lectura de la imagen DICOM: {image_path}")
        
        elif image_path.endswith(".jpg"):
            try:
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    pathology = data.iloc[i]["pathology"]
                    axes[i].imshow(image)
                    axes[i].set_title(pathology)
                    axes[i].axis("off")
                else:
                    print(f"La imagen JPEG no existe: {image_path}")
                    
            except Exception as e:
                print(f"Error en la lectura de la imagen JPEG: {image_path}")
                
        else:
            print(f"No se puede procesar esta extensión: {image_path}")
    fig.savefig(output_file, format='jpeg', bbox_inches="tight")

def get_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return width, height
    except Exception as e:
        print(f"Error al abrir la imagen {image_path}: {e}")
        return None, None

def load_images_with_dimensions(data):
    # Inicializamos las columnas
    data['roi_width'], data['roi_height'] = None, None
    data['full_width'], data['full_height'] = None, None
    data['cropped_width'], data['cropped_height'] = None, None
    
    for i in range(len(data)):
        # Dimensiones ROI
        roi_path = data.iloc[i, 13]
        roi_width, roi_height = get_image_dimensions(roi_path)
        data.at[i, 'roi_width'] = roi_width
        data.at[i, 'roi_height'] = roi_height
        
        # Dimensiones de la imagen completa
        full_path = data.iloc[i, 11]
        full_width, full_height = get_image_dimensions(full_path)
        data.at[i, 'full_width'] = full_width
        data.at[i, 'full_height'] = full_height
        
        # Dimensiones de la imagen recortada
        cropped_path = data.iloc[i, 12]
        cropped_width, cropped_height = get_image_dimensions(cropped_path)
        data.at[i, 'cropped_width'] = cropped_width
        data.at[i, 'cropped_height'] = cropped_height
    
    return data

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

def process_image_to_tensor(image_path, target_size):
    with Image.open(image_path) as image:
        processed_image = crop_or_pad_image(image, target_size)
        image_array = np.array(processed_image, dtype=np.float32) / 255.0
    return torch.tensor(image_array)

def process_and_store_images(data, tensor_column_name,
                             tensor_column='tensor_column',
                             target_size=(2000, 3000)):
    data[tensor_column] = None
    for i, row in data.iterrows():
        image_tensor = process_image_to_tensor(row[tensor_column_name], target_size=target_size)
        data.at[i, tensor_column] = image_tensor
        del image_tensor
    return data

def process_and_display_images(data, tensor_column_name, number, output_file='default.jpeg',
                         target_size=(2000, 3000), tensor_column='tensor_column'):
    
    fig, axes = plt.subplots(1, number, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5)
    
    for i in range(number):
        # Obtenemos la imagen y la etiqueta
        image_tensor = process_image_to_tensor(data.iloc[i][tensor_column_name], target_size)
        pathology = data.iloc[i]["pathology"]

        # Convertimos el tensor a un array NumPy (manejando GPU si es necesario)
        if image_tensor.device != torch.device('cpu'):
            image_tensor = image_tensor.cpu()

        image_array = (image_tensor * 255).numpy().astype(np.uint8)

        # Verificamos el canal de color
        if len(image_array.shape) == 3 and image_array.shape[-1] == 3: 
            axes[i].imshow(image_array)
        else:
            axes[i].imshow(image_array, cmap="gray")

        # Mostramos el título y quitamos ejes
        axes[i].set_title(pathology)
        axes[i].axis("off")
    
    # Guardamos el resultado
    fig.savefig(output_file, format='jpeg', bbox_inches="tight")

def is_all_black_or_margin_non_black(tensor, margin=10):
    """Comprobación si la imagen es completamente negra o está cortada:
        0 si es correcta
        1 si es negra
        2 si está cortada"""
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

def check_cropping_errors(data, batch_size=100, target_size=(2000, 3000)):
    croppingerror = []
    
    for i, row in data.iterrows():
        roi_tensor = process_image_to_tensor(row['ROI_mask_file_path'], target_size)
        croppingerror.append(is_all_black_or_margin_non_black(roi_tensor))
    
    return croppingerror
    
def save_tensor_as_png(tensor, path, name, mode="png"):
    
    tensor = tensor.unsqueeze(2)
    tensor = tensor.cpu().numpy()
    tensor = (tensor * 255).clip(0, 255).astype("uint8")
    image = Image.fromarray(tensor.squeeze())
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, f"{name}.{mode}")
    image.save(file_path, format=str.upper(mode))
    print(f"Image saved at: {file_path}")
