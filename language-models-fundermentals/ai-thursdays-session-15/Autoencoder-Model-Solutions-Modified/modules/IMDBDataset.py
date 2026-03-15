import torch
import pandas as pd
from torchtext.vocab import Vocab

import configs as cfg1
import configs_advanced as cfg2
from text_processing import text_pipeline

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(
      self, 
      dataframe: pd.DataFrame,
      config: cfg1.Config | cfg2.Config,
      vocab: Vocab 
      ):
        # Storing features and labels as tensors often improves performance
        # Ensure all columns are numeric before conversion
        self.review = dataframe["review"].values
        self.labels = torch.tensor(dataframe['sentiment'].values, dtype=torch.float32)

        self.config = config
        self.vocab = vocab

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return text_pipeline(self.review[idx], self.config, self.vocab), self.labels[idx]
