import kagglehub

def download_dataset_hist():
    # Download latest version
    custom_path = "datasets/"
    path = kagglehub.dataset_download("paultimothymooney/breast-histopathology-images", custom_path)
    
    print("Path to dataset files:", path)
