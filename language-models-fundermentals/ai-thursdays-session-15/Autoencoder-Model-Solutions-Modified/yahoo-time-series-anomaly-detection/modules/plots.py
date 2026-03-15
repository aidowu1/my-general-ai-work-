# plots.py

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def plot_losses(train_losses, val_losses):

    plt.figure(figsize=(8,4))

    plt.plot(train_losses,label="train")
    plt.plot(val_losses,label="val")

    plt.title("Training / Validation Loss")

    plt.legend()

    plt.show()


def plot_roc(labels, scores):

    fpr,tpr,_ = roc_curve(labels,scores)

    plt.figure()

    plt.plot(fpr,tpr)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.show()