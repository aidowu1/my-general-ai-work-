import torch
import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def reconstruction_errors(model, loader, device):
    model.eval()
    errors = []
    labels = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            recon = model(x)
            err = torch.mean((recon-x)**2,dim=1)
            errors.extend(err.cpu().numpy())
            labels.extend(y.numpy())
    return np.array(errors), np.array(labels)

def evaluate(model, loader, device, threshold=None):

    errors, labels = reconstruction_errors(model, loader, device)

    if threshold is None:
        threshold = np.percentile(errors, 95)

    preds = (errors > threshold).astype(int)

    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, errors)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds)

    return {
        'f1_score': f1,
        'roc_auc': auc,
        'confusion_matrix': cm,
        'classification_report': report
    }