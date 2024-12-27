import os
import matplotlib.pyplot as plt
from PIL import Image

def display_png_images(data, column, number, output_file='default.png'):
    '''Muestra el número indicado de imágenes cuya ruta está en la columna
        indicada del dataset y lo guarda como PNG'''
    fig, axes = plt.subplots(1, number, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.5)
    
    for i in range(number):
        image_path = data.iloc[i][column]
        
        print(f"Leyendo: {image_path}")
        
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
            print(f"No se puede procesar esta extensión: {image_path}")
    
    # Save the figure as a PNG file
    fig.savefig(output_file, format='png', bbox_inches="tight")

