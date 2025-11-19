import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import time

import modules.configs as cfg
import modules.dnn_data_loader as ddl

class DNNClassifier(nn.Module):
    """
    A Deep Neural Network (DNN) classifier for binary classification tasks.
    """
    def __init__(
            self, 
            input_size, 
            hidden_size1: int = cfg.DNN_HIDDEN_SIZE1, 
            hidden_size2: int = cfg.DNN_HIDDEN_SIZE2, 
            output_size: int = cfg.DNN_OUTPUT_SIZE, 
            dropout_rate: float = cfg.DNN_DROPOUT_RATE):
        """
        Initialize the DNN classifier.
        :param input_size: Size of the input features   
        :param hidden_size1: Number of neurons in the first hidden layer
        :param hidden_size2: Number of neurons in the second hidden layer
        :param output_size: Size of the output layer
        :param dropout_rate: Dropout rate for regularization
        """
        super(DNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

class DNNMarketDirectionPredictor:
    """
    DNN Market Direction Predictor class
    """
    def __init__(self, input_size: int):
        """
        Initialize the DNN model.
        :param input_size: Size of the input features   
        """
        self.model = DNNClassifier(input_size).to(cfg.DEVICE)
        self.criterion = nn.BCELoss() # Binary Cross-Entropy Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.DNN_LEARNING_RATE)
        

    def evaluate_model(self, loader):
        """
        Evaluate the model on given DataLoader.
        :param loader: DataLoader for evaluation
        :return: Dictionary with AUC and accuracy
        """
        # Validation phase
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in loader:
                Xb = batch["features"].to(cfg.DEVICE)
                yb = batch["label"].to(cfg.DEVICE)
                outputs = self.model(Xb).cpu().detach().squeeze().tolist()
                labels = yb.cpu().detach().squeeze().tolist()

                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
    def train(
            self, 
            train_loader: DataLoader,
            val_loader: DataLoader):
        """
        Train the DNN model.
        :param train_loader: Training features and labels DataLoader
        :param val_loader: Validation features and labels DataLoader        
        """
        start_time = time.time()
        # initialize early stopping tracking
        best_val_auc = -np.inf
        best_epoch = 0
        patience_counter = 0
        best_state = None

        for epoch in range(1, cfg.DNN_NUM_EPOCHS + 1):
            self.model.train()
            epoch_losses = []
            for batch in train_loader:
                Xb = batch["features"].to(cfg.DEVICE)
                yb = batch["label"].to(cfg.DEVICE)
                # ensure labels have same shape as model outputs (batch, 1)
                if yb.dim() == 1:
                    yb = yb.unsqueeze(1)
                self.optimizer.zero_grad()
                logits = self.model(Xb)  # shape (batch, 1)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            # validation
            val_metrics = self.evaluate_model(val_loader)
            val_auc = val_metrics["auc"]
            val_acc = val_metrics["acc"]
            # self.scheduler.step(val_auc)
            avg_train_loss = float(np.mean(epoch_losses))
            print(f"Epoch {epoch:02d} | train_loss={avg_train_loss:.4f} | val_auc={val_auc:.4f} | val_acc={val_acc:.4f}")

            # early stopping
            # if val_auc > best_val_auc + cfg.EARLY_STOPPING_TOLERANCE:
            #     best_val_auc = val_auc
            #     best_epoch = epoch
            #     best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            #     patience_counter = 0
            # else:
            #     patience_counter = (epoch - best_epoch)
            # if (epoch - best_epoch) >= cfg.EARLY_STOPPING_PATIENCE:
            #     print(f"Early stopping after epoch {epoch}. Best epoch was {best_epoch} with val_auc {best_val_auc:.4f}")
            #     break

        total_time = time.time() - start_time
        print(f"Training finished in {total_time:.1f}s. Best epoch: {best_epoch}, Best val_auc: {best_val_auc:.4f}")

    def predict(self, test_loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict using the trained DNN model.
        :param test_loader: DataLoader for test data
        :return: Tuple of true labels, predicted probabilities, and predicted labels
        """
        self.model.eval()
        y_true_list = []
        y_pred_list = []
        y_prob_list = []
        with torch.no_grad():
            for batch in test_loader:
                Xb = batch["features"].to(cfg.DEVICE)
                logits = self.model(Xb)
                probs = torch.sigmoid(logits).cpu().detach().squeeze().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                preds = [1 if p >= 0.5 else 0 for p in probs]
                y_prob_list.extend(probs)
                y_pred_list.extend(preds)
                # labels may be tensors on CPU/GPU; normalize to Python list
                labels = batch["label"].cpu().detach().squeeze().tolist()
                if isinstance(labels, float) or isinstance(labels, (int,)):
                    labels = [labels]
                y_true_list.extend(labels)

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        y_prob = np.array(y_prob_list)
        return y_true, y_prob, y_pred

        
    def get_model(self) -> nn.Module:
        """
        Get the DNN model.
        :return: DNN model
        """
        return self.model