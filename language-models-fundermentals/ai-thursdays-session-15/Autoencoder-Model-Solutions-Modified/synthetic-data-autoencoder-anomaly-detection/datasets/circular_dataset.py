# datasets/circular_dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset


class CircularGaussianDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


class CircularDatasetGenerator:

    def __init__(self, n_samples=10000):

        self.n_samples = n_samples

    def generate(self):

        n_normal = int(self.n_samples * 0.9)
        n_anom = self.n_samples - n_normal

        # -------- NORMAL DATA (RING) --------

        theta = np.random.uniform(0, 2*np.pi, n_normal)

        radius = np.random.normal(1.0, 0.05, n_normal)

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        normal = np.stack([x,y], axis=1)

        # -------- ANOMALY TYPE 1 (CENTER) --------

        n_inner = n_anom // 2

        inner = np.random.normal(
            loc=0,
            scale=0.2,
            size=(n_inner,2)
        )

        # -------- ANOMALY TYPE 2 (OUTSIDE) --------

        n_outer = n_anom - n_inner

        theta = np.random.uniform(0,2*np.pi,n_outer)

        radius = np.random.normal(2.5,0.2,n_outer)

        x = radius*np.cos(theta)
        y = radius*np.sin(theta)

        outer = np.stack([x,y],axis=1)

        anomalies = np.vstack([inner,outer])

        X = np.vstack([normal, anomalies])

        y = np.hstack([
            np.zeros(len(normal)),
            np.ones(len(anomalies))
        ])

        return X, y
    
def split_dataset(X, y):
    n = len(X)

    idx = np.random.permutation(n)

    train_end = int(n * 0.6)
    val_end = int(n * 0.9)

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx]
    )