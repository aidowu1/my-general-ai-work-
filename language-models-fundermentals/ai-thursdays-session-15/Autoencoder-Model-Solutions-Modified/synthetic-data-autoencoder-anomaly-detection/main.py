from datasets.circular_dataset import (CircularDatasetGenerator, 
                                       CircularGaussianDataset,
                                       split_dataset
                                       )
import torch
import numpy as np
from torch.utils.data import DataLoader

from model import CircularAutoencoder  
from trainer import Trainer
from evaluator import reconstruction_errors, evaluate
from plots import plot_roc_curve, plot_losses, plot_predictions

def inspect_loaders(train_loader, val_loader, test_loader):
    """Inspect and print first 10 samples from each data loader."""
    for loader_name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
        for i, (X, y) in enumerate(loader):
            if i == 0:
                print(f"{loader_name} Loader - First 10 samples:")
                print(X[:10], y[:10])
            break
def run_training_and_validation_cycle(
        device: torch.device,
        model: torch.nn.Module,
        train_loader: DataLoader, 
        val_loader: DataLoader
):
    # This function can be called after inspecting the loaders to run the full training and evaluation pipeline.
    # # -------- MODEL --------
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    trainer = Trainer(model, optimizer, criterion, device)

    # -------- TRAINING --------
    n_epochs = 50
    for epoch in range(n_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate_epoch(val_loader)

        trainer.train_losses.append(train_loss)
        trainer.val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
  
    return model, trainer


def main():
    # -------- DATASET --------

    generator = CircularDatasetGenerator(n_samples=10000)
    X, y = generator.generate()
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    train_dataset = CircularGaussianDataset(X_train, y_train)
    val_dataset = CircularGaussianDataset(X_val, y_val)
    test_dataset = CircularGaussianDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)    

    inspect_loaders(train_loader, val_loader, test_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CircularAutoencoder()
    trained_model, trainer = run_training_and_validation_cycle(device, model, train_loader, val_loader)

    # -------- EVALUATION --------
    results = evaluate(trained_model, val_loader, device)

    print("Evaluation Results:")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    print("Classification Report:")
    print(results['classification_report'])

    #-------- PLOTS --------
    plot_losses(trainer) 
    test_errors, test_labels = reconstruction_errors(trained_model, test_loader, device)
    plot_roc_curve(test_labels, test_errors)

    # For visualization, we can use the test set and color by predicted anomaly scores
    test_preds = (test_errors > np.percentile(test_errors, 95)).astype(int)
    test_X = np.vstack([x.numpy() for x,_ in test_loader])
    plot_predictions(test_X, test_preds)

if __name__ == "__main__":
    main()