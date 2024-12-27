import os
import pydicom
from pydicom.errors import InvalidDicomError
import matplotlib.pyplot as plt
from PIL import Image

def display_images(data, column: str, number: int, output_file: str = 'default.jpeg'):
    ''' Carga las imágenes que se indiquen en una columna y las junta'''
    fig, axes = plt.subplots(1, number, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5)
    
    for i in range(number):
        image_path = data.iloc[i][column]
        
        print(f"Leyendo: {image_path}")
        if image_path.endswith(".dcm"):
            try:
                if os.path.exists(image_path):
                    dicom_data = pydicom.dcmread(image_path)
                    image = dicom_data.pixel_array
                    pathology = data.iloc[i]["pathology"]
                    axes[i].imshow(image, cmap="gray")
                    axes[i].set_title(pathology)
                    axes[i].axis("off")
                else: 
                    print(f"La imagen DICOM no existe: {image_path}")
            
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


