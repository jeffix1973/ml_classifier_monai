import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

# Framework imports
import classifier.preprocessing as preprocessing


class DicomDataset(Dataset):
    def __init__(self, data_root_path, mode, df, resize_shape, num_classes):
        self.data_root_path = data_root_path
        self.mode = mode
        self.df = df
        self.resize_shape = resize_shape
        self.num_classes = num_classes
        self.df.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        dcm_path = self._get_dcm_path(row)
        label = int(row['label'])

        # Load and preprocess the image
        _, processed = preprocessing.preprocess_pipeline(dcm_path, self.resize_shape, self.mode)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return processed, label_tensor

    def _get_dcm_path(self, row):
        join_path = "/".join(row['OLEA_INSTANCE_PATH'].split("/")[-3:])
        dcm_path = os.path.join(self.data_root_path, join_path)
        if not dcm_path.endswith('.dcm'):
            dcm_path += '.dcm'
        return dcm_path

def create_dataloaders(data_root_path, csv_file, resize_shape, batch_size, num_classes):
    df = pd.read_csv(csv_file)
    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'eval']

    train_dataset = DicomDataset(data_root_path, 'train', df_train, resize_shape, num_classes)
    val_dataset = DicomDataset(data_root_path, 'eval', df_val, resize_shape, num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def create_test_dataloader(data_root_path, test_csv_file, resize_shape, batch_size, num_classes):
    df_test = pd.read_csv(test_csv_file)
    test_dataset = DicomDataset(data_root_path, 'test', df_test, resize_shape, num_classes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
