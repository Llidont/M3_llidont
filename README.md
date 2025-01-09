# Clasificación de tumores de mama mediante imágenes

# Repositorio: Clasificación de Tumores en Mamografías

Este repositorio contiene los scripts, funciones y modelos necesarios para la clasificación de tumores en mamografías, basados en el dataset público CBIS-DDSM. Incluye múltiples etapas de preprocesamiento, generación de datasets y entrenamiento de modelos, además de scripts adicionales para visualización y análisis de resultados.

## Ficha

Autora: Laura María Lidón Tárraga (llidont@uoc.edu)
Tutora: Ana Belén Nieto Librero
Repositorio: https://github.com/Llidont/M3_llidont
Respaldo: https://drive.google.com/drive/folders/1fbG6DSPdrKAo6__7JsctJLh4BKRHEwss (solo accesible con cuenta corporativa de la UOC)

## Requisitos
Para ejecutar el código, es necesario instalar las dependencias listadas en el archivo `requirements.txt` en unentorno de Python 3.12.3 para evitar problemas de compatibilidad. Además, el repositorio puede requerir si por cambios de versión no funciona de manera automática de la creación del directorio `datasets/archives` con el archivo comprimido `archives.zip` descargable [aquí](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset).

## Estructura del Repositorio

### Directorios Principales
- **`datasets/`**: Carpeta para almacenar el dataset original y los procesados.
- **`examples/`**: Contiene imágenes de ejemplo generadas durante el preprocesamiento y entrenamiento.
- **`functions/`**: Implementaciones de funciones para preprocesamiento y construcción de modelos.
- **`models/`**: Carpeta para almacenar los modelos entrenados.

### Scripts de Preprocesamiento
1. **`01preprocesado_calc.py` y `02preprocesado_mass.py`**: Creación de datasets iniciales (`clean`).
2. **`11preprocesado_calc_prop.py` y `12preprocesado_mass_prop.py`**: Creación de datasets escalados proporcionalmente (`prop`).
3. **`21distorsion_calc.py` y `22distorsion_mass.py`**: Aumento de datos aplicando distorsiones a los datasets `prop`.

### Scripts de Modelos
1. **`31calc_linear.py` y `32mass_linear.py`**: Modelos lineales para clasificación de calcificaciones y masas.
2. **`41calc_cnn.py` y `42mass_cnn.py`**: Redes neuronales convolucionales (CNN) básicas.
3. **`52mass_trans.py`**: Implementación de modelos avanzados usando transformadores.

### Scripts de Análisis y Resultados
1. **`91create_example_images.py`**: Generación de imágenes de ejemplo del dataset.
2. **`92create_trial_model_results.py`**: Procesa y organiza resultados de pruebas.
3. **`93create_best_models_results.py`**: Selección y análisis de los mejores modelos.
4. **`94create_graphics.py`**: Gráficos de evolución de métricas de los modelos.
5. **`95create_LTH.py`**: Implementación del `Lottery Ticket Hypothesis`.

## Notas
- Los datasets procesados y algunos elementos (.pth) solo están disponibles en [Google Drive](https://drive.google.com/drive/folders/1fbG6DSPdrKAo6__7JsctJLh4BKRHEwss) por su tamaño.
- Para consultas adicionales, contacta a la autora mediante correo: `llidont@uoc.edu`.

## Licencia
Este proyecto está disponible bajo la licencia de Reconocimiento-NoComercial-SinObraDerivada 3.0 de España de CreativeCommons.CreativeCommons3.


