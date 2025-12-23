import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


def run():
    """ Train a neural network regressor on California housing dataset.
    This function reads the dataset, preprocesses it, defines a neural network model,
    trains the model, and evaluates its performance using Mean Squared Error (MSE).
    It uses PyTorch for model definition and training, and Matplotlib for plotting the training history.
    """

    # Configuration
    DATA_PATH = "housing.csv"
    FEATURE_COLUMNS = ["longitude", "latitude", "housing_median_age", "total_rooms",
                       "total_bedrooms", "population", "households", "median_income"]
    LABEL_COLUMN = "median_house_value"
    NUM_SAMPLES = 1000
    BATCH_SIZE = 10
    N_EPOCHS = 100
    LR = 1e-4

    # Load and prepare numpy arrays
    df = pd.read_csv(DATA_PATH, usecols=FEATURE_COLUMNS + [LABEL_COLUMN]).head(NUM_SAMPLES)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)

    # Split and build TensorDatasets + DataLoaders
    X_train_raw, X_valuation_raw, y_train, y_valuation = train_test_split(X, y, train_size=0.7, shuffle=True)
    
    # Standardizing data
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_valuation = scaler.transform(X_valuation_raw)
    
    # ensure tensors are float32 (matching model parameters) to avoid dtype-related issues
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_valuation_t = torch.from_numpy(X_valuation).float()
    y_valuation_t = torch.from_numpy(y_valuation).float()

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_valuation_t, y_valuation_t)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, loss and optimizer
    model = nn.Sequential(
        nn.Linear(len(FEATURE_COLUMNS), 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1),
    )
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_mse = np.inf
    best_weights = None
    history = []
        
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        # show progress bar for training batches; ensure it's enabled so it displays
        with tqdm.tqdm(train_loader, unit="batch", mininterval=0.1, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}")
            for xb, yb in bar:
                xb = xb.float()
                yb = yb.float()
                optimizer.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                batch_size = xb.size(0)
                train_loss_total += loss.item() * batch_size
                train_samples += batch_size
                # update postfix with a dict to ensure proper formatting
                bar.set_postfix({"train_mse": loss.item()})
    
        # Validation (evaluate on the entire val dataset)
        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.float()
                yb = yb.float()
                preds = model(xb)
                batch_loss = loss_fn(preds, yb)
                total_loss += batch_loss.item() * xb.size(0)
                total_samples += xb.size(0)
        mse = total_loss / total_samples if total_samples > 0 else float("nan")
        history.append(mse)
    
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())            

    # Report and plot
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    plt.plot(history)
    plt.show()



if __name__ == "__main__":
    run()