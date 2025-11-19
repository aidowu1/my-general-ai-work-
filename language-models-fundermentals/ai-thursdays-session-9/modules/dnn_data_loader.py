
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import modules.configs as cfg

class TabularDataset(Dataset):
    def __init__(self, X, y):
        # Ensure inputs are plain numpy arrays with correct dtype
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)  # for BCEWithLogitsLoss we want float labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Use torch.tensor to convert array-likes to tensors without relying
        # on PyTorch's NumPy C-extension (some builds may not have NumPy support).
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        return {
            "features": features,
            "label": label
        }

def create_dataloader(
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray, 
        y_val: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray,
        batch_size: int=cfg.DNN_BATCH_SIZE
        ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects for training, validation, and testing datasets.
    :param X_train: Training features
    :param y_train: Training labels
    :param X_val: Validation features
    :param y_val: Validation labels
    :param X_test: Test features
    :param y_test: Test labels
    :param batch_size: Batch size for DataLoaders
    :return: DataLoader objects for train, validation, and test datasets
    """
    train_ds = TabularDataset(X_train, y_train)
    val_ds = TabularDataset(X_val, y_val)
    test_ds = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader