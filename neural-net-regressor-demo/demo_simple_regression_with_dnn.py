import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def plot_loss(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

def plot_predictions(y_true, y_pred):
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.show()

def plot_data_profile(y_raw):
    plt.plot(range(len(y_raw)), y_raw, 'r', alpha=0.2)
    plt.title("Raw Data Profile plot")  
    plt.xlabel("Data point index")
    plt.ylabel("Target")
    plt.show()

# 1. Generate and Split Raw Data (Numpy format)
np.random.seed(100)
TOTAL_SAMPLES = 1100
N_TEST_SAMPLES = 100
X_all = np.random.randn(TOTAL_SAMPLES, 5)
y_all = X_all.sum(axis=1, keepdims=True) + np.random.randn(TOTAL_SAMPLES, 1) * 0.1
X_raw = X_all[:TOTAL_SAMPLES - N_TEST_SAMPLES]
y_raw = y_all[:TOTAL_SAMPLES - N_TEST_SAMPLES]
X_test = X_all[TOTAL_SAMPLES - N_TEST_SAMPLES:]
y_test = y_all[TOTAL_SAMPLES - N_TEST_SAMPLES:]

plot_data_profile(y_raw)

# Use 25% for validation (0.25 fraction)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_raw, y_raw, test_size=0.25, random_state=42
)

# 2. Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)  # Fit and transform training
X_val_scaled = scaler.transform(X_val_raw)        # Transform validation using training stats

# 3. Convert to PyTorch Tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val = torch.tensor(X_val_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

train_set = TensorDataset(X_train, y_train)
val_set = TensorDataset(X_val, y_val)

# Create DataLoaders for batching
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)


class RegressionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegressionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),    # Input to Hidden 1
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   # Hidden 1 to Hidden 2
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)    # Hidden 2 to Output
        )
    
    def forward(self, x):
        return self.network(x)

model = RegressionNet(input_dim=X_raw.shape[1], hidden_dim=10, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1000

train_losses = []
val_losses = []
for epoch in range(100):
    model.train()
    sum_train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        train_loss = criterion(outputs, batch_y)
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
        sum_train_loss += train_loss.item()

    # Validation Phase
    model.eval()
    sum_val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_loss = criterion(outputs, batch_y)
            val_losses.append(val_loss.item())
            sum_val_loss += val_loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train Loss {sum_train_loss/len(train_loader):.4f} | Val Loss {sum_val_loss/len(val_loader):.4f}")

plot_loss(train_losses, val_losses)

model.eval()
with torch.no_grad():
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")
    plot_predictions(y_test_tensor.numpy(), y_pred.numpy())
    

