# config.py

from dataclasses import dataclass

@dataclass
class Config:

    data_path: str = "data/yahoo_s5.csv"

    sequence_length: int = 64
    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-3

    train_split: float = 0.6
    val_split: float = 0.3
    test_split: float = 0.1

    latent_dim: int = 16
    hidden_dim: int = 64

    device: str = "cuda" if torch.cuda.is_available() else "cpu"