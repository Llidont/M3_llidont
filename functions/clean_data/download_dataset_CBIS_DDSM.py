import os
import shutil
import kagglehub

def download_dataset_CBIS_DDSM():
    archives_path = 'datasets/archive'
    archives_zip = 'datasets/archive.zip'
    if os.path.exists(archives_zip):
        print(f"Se ha encontrado un archivo comprimido, descomprimiendo...")
        with zipfile.ZipFile(archives_zip, 'r') as zip_ref:
            zip_ref.extractall(archives_path)
    else:
        print(f"No se ha encontrado el archivo comprimido, descargando...")
        path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")
        print(path)
        os.makedirs(archives_path, exist_ok=True)
        shutil.move(path, archives_path)
        folders_to_move = ['csv', 'jpeg']
        for folder in folders_to_move:
            source_path = os.path.join(archives_path, '1', folder)
            destination_path = os.path.join(archives_path, folder)
            shutil.move(source_path, destination_path)
        shutil.rmtree(os.path.join(archives_path, '1'))