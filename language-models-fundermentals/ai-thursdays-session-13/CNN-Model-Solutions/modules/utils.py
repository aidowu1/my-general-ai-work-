from torch.utils.data import DataLoader

from configs import Config

cfg = Config()

def sample_dataloader(
  data_loader: DataLoader, 
  n_batch_samples: int = 2, 
  n_samples: int = 5):
  """
  Samples dataloader (features/labels)
  :param data_loader: Data loader
  :param n_batches: Number of batches
  :param n_samples: Number of samples
  """
  counter = 0
  for inputs, labels in data_loader:
    counter += 1
    if counter > n_batch_samples:
      break
    inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
    print(f"inputs.shape: {inputs.shape}")
    print(f"labels.shape: {labels.shape}")
    print(f"labels[:n_samples]: {labels[:n_samples]}")
    print(f"inputs[:n_samples]: {inputs[:n_samples]}")
    print(f"{cfg.line_divider}{cfg.next_line}")
  