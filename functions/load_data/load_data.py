import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def load_data(image_size, batch_size, file_name = "processed_dataset_info.csv"):

    processed_dataset = pd.read_csv(file_name)
    #processed_dataset['ROI_path'] = processed_dataset['ROI_path'].astype(str).apply(lambda x: x + '.png')
    combined_dataset = processed_dataset.sample(frac=1).reset_index(drop=True)
    combined_dataset["ROI_path"] = combined_dataset["ROI_path"].str.replace('\\', '/')
    combined_dataset["label"] = combined_dataset["label"].replace("BENIGN_WITHOUT_CALLBACK", "BENIGN")
    combined_dataset["label"] = combined_dataset["label"].map({'BENIGN': 0, 'MALIGNANT': 1})

    train_split = 0.7
    val_split = 0.2
    #test_split = 0.1
    train_size = int(len(combined_dataset) * train_split)
    val_size = int(len(combined_dataset) * val_split)

    train_data = combined_dataset[:train_size]
    val_data = combined_dataset[train_size:train_size + val_size]
    test_data = combined_dataset[train_size + val_size:]

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
            img_path = self.dataframe.iloc[idx]["ROI_path"]
            label = self.dataframe.iloc[idx]["label"]
            image = Image.open(img_path).convert("L").resize(self.image_size)
            if self.transform:
                image = self.transform(image)
            return image, label