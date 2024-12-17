import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def load_mass_with_metadata(image_size, batch_size, file_name = "processed_dataset_info.csv"):

    processed_dataset = pd.read_csv(file_name).dropna()
    combined_dataset = processed_dataset.sample(frac=1).reset_index(drop=True)
    combined_dataset["label"] = combined_dataset["label"].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
    combined_dataset["label"] = combined_dataset["label"].map({'BENIGN': 0, 'MALIGNANT': 1})
    combined_dataset["mass_shape"] = indexify_column(combined_dataset, "mass_shape")
    combined_dataset["mass_margins"] = indexify_column(combined_dataset, "mass_margins")
    combined_dataset["image_view"] = indexify_column(combined_dataset, "image_view")

    train_split = 0.7
    val_split = 0.2
    test_split = 0.1
    train_size = int(len(combined_dataset) * train_split)
    val_size = int(len(combined_dataset) * val_split)

    train_data = combined_dataset[:train_size].reset_index(drop=True)
    val_data = combined_dataset[train_size:train_size + val_size].reset_index(drop=True)
    test_data = combined_dataset[train_size + val_size:].reset_index(drop=True)

    

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = ImageDataset(image_size, train_data, transform=transform)
    val_dataset = ImageDataset(image_size, val_data, transform=transform)
    test_dataset = ImageDataset(image_size, test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class ImageDataset(Dataset):
        def __init__(self, image_size, dataframe, transform=None):
            self.dataframe = dataframe
            self.transform = transform
            self.image_size = image_size
        
        def __len__(self):
            return len(self.dataframe)
        
        def __getitem__(self, idx):

            img_path = self.dataframe.loc[idx, "ROI_path"]
            label = self.dataframe.loc[idx, "label"]
            image = Image.open(img_path).convert("L").resize(self.image_size)
            if self.transform:
                image = self.transform(image)
            
            self.dataframe[["subtlety", "image_view", "breast_density", "shape_vertical", "shape_horizontal"]] = self.dataframe[
            ["subtlety", "breast_density", "shape_vertical", "shape_horizontal"]].apply(pd.to_numeric, errors="coerce")

            shape = torch.tensor(self.dataframe.loc[idx, "mass_shape"], dtype=torch.long)
            margins = torch.tensor(self.dataframe.loc[idx, "mass_margins"], dtype=torch.long)

            other_metadata = self.dataframe.loc[idx, ["subtlety", "image_view", "breast_density", "shape_vertical", "shape_horizontal"]].values
            other_metadata = np.array(other_metadata, dtype=np.float32)

            other_metadata = torch.tensor(other_metadata, dtype=torch.float32)

            label = torch.tensor(self.dataframe.loc[idx, "label"], dtype=torch.long)
            return image, shape, margins, other_metadata, label

def indexify_column(df, column_name):
    column_dict = {}
    i = 0
    all_items_in_column = set(df[column_name])
    for item in all_items_in_column:
        column_dict[item] = i
        i += 1
    df[column_name] = df[column_name].map(column_dict)
    return df[column_name]