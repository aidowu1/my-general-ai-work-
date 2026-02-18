from shutil import copy
import copy
from xml.parsers.expat import model
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from torch.utils.data import DataLoader

from advanced_text_cnn import AdvancedTextCNN
import configs_advanced as cfg

config = cfg.Config()

class AdvancedTextClassifier:
  """
  Class of the "advanced" CNN classifier 
  """
  def __init__(
        self,
        model: AdvancedTextCNN, 
        train_loader: DataLoader, 
        val_loader: DataLoader        
        ):
      """
      Consructor, initializes the AdvancedTextClassifier class. 
      This can be used to encapsulate any additional methods or attributes related to the advanced CNN model.
      params: 
        model: The advanced CNN model to be trained and evaluated.
        train_loader: The DataLoader for the training data.
        val_loader: The DataLoader for the validation data.
        
      """
      self.model = model
      self.train_loader = train_loader
      self.val_loader = val_loader
      self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
      self.criterion = nn.BCEWithLogitsLoss()
      self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
          self.optimizer, 
          mode=config.lr_schedule_mode, 
          factor=config.lr_schedule_factor, patience=config.patience, verbose=config.lr_schedule_is_verbose)

     
  def train_epoch(self) -> float:
    """
    Trains the model for one epoch and returns the average loss.
    """
    self.model.train()
    total_loss = 0

    for inputs, labels in tqdm(self.train_loader):
        inputs, labels = inputs.to(config.device), labels.to(config.device)

        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = self.criterion(logits, labels)
        
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()

    return total_loss / len(self.train_loader)


  def evaluate(self) -> tuple[float, float]:
      self.model.eval()
      total_loss = 0
      correct = 0
      total = 0

      with torch.no_grad():
          for inputs, labels in self.val_loader:
              inputs, labels = inputs.to(config.device), labels.to(config.device)
              
              logits = self.model(inputs)
              loss = self.criterion(logits, labels)
              
              total_loss += loss.item()

              preds = torch.sigmoid(logits)
              preds = (preds >= 0.5).float()

              correct += (preds == labels).sum().item()
              total += labels.size(0)

      return total_loss / len(self.val_loader), correct / total
  
  def run_train_validation_cycle(self) -> AdvancedTextCNN:
    """
    Runs the training and validation cycle for the advanced CNN model, including early stopping based on validation loss.
    """      
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.epochs):

        train_loss = self.train_epoch()
        val_loss, val_acc = self.evaluate()

        self.scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(self.model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print("Early stopping triggered.")
            break

    # Restore best model
    self.model.load_state_dict(best_model_state)
    torch.save(self.model.state_dict(), config.model_advanced_cnn_path)
    return self.model
