from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np  

from trainer import Trainer

def plot_roc_curve(test_labels: np.ndarray, test_errors: np.ndarray):
    fpr,tpr,_ = roc_curve(test_labels,test_errors)
    plt.plot(fpr,tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.show()

def plot_losses(trainer: Trainer):
    plt.plot(trainer.train_losses,label="train")
    plt.plot(trainer.val_losses,label="val")
    plt.legend()
    plt.title("Loss")
    plt.show()

def plot_predictions(X: np.ndarray, preds: np.ndarray):
    plt.scatter(
        X[:,0],
        X[:,1],
        c=preds,
        cmap="coolwarm",
        s=8
    )
    plt.title("Anomaly Predictions")
    plt.show()