from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import Vocab

from simple_text_cnn import SimpleTextCNN
import configs as cfg

config = cfg.Config()

class SimpleTextClassifier:
  """
  Static class of the "simple" CNN classifier 
  """
  def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for inputs, labels in tqdm(loader):
        inputs, labels = inputs.to(config.device), labels.to(config.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


  def evaluate(model, loader, criterion):
      model.eval()
      total_loss = 0
      correct = 0
      total = 0

      with torch.no_grad():
          for inputs, labels in loader:
              inputs, labels = inputs.to(config.device), labels.to(config.device)
              outputs = model(inputs)
              loss = criterion(outputs, labels)
              total_loss += loss.item()

              preds = (outputs >= 0.5).float()
              correct += (preds == labels).sum().item()
              total += labels.size(0)

      return total_loss / len(loader), correct / total

  def run_train_validation_cycle(
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        vocab: Vocab,
        config_override: cfg.Config = config
    ) -> SimpleTextCNN:
    """
    Runs the training and validation cycle for the simple CNN model.
    :param train_loader: The DataLoader for the training data.
    :param val_loader: The DataLoader for the validation data.
    :param vocab: The vocabulary object.
    :param config_override: The configuration object to override the default config.
    :returns: The trained SimpleTextCNN model.
    """
    model = SimpleTextCNN(vocab).to(config_override.device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config_override.lr)
    for epoch in range(config_override.epochs):
        train_loss = SimpleTextClassifier.train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = SimpleTextClassifier.evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{config_override.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), config_override.model_simple_cnn_path)
    return model