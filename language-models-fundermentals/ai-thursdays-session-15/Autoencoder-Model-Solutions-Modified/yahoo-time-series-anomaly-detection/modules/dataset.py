# dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class YahooTimeSeriesDataset(Dataset):

    def __init__(self, data, labels, seq_len):

        self.seq_len = seq_len
        self.data = data
        self.labels = labels

        self.windows = []
        self.window_labels = []

        for i in range(len(data) - seq_len):

            window = data[i:i+seq_len]
            label = max(labels[i:i+seq_len])

            self.windows.append(window)
            self.window_labels.append(label)

        self.windows = np.array(self.windows)
        self.window_labels = np.array(self.window_labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):

        x = torch.FloatTensor(self.windows[idx])
        y = torch.FloatTensor([self.window_labels[idx]])

        return x, y