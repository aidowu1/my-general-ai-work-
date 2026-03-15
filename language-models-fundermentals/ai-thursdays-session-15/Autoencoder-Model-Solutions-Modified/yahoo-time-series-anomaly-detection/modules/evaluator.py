# evaluator.py

import numpy as np
import torch

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


class Evaluator:

    def __init__(self, model, device):

        self.model = model
        self.device = device


    def reconstruction_errors(self, loader):

        self.model.eval()

        errors = []
        labels = []

        with torch.no_grad():

            for x,y in loader:

                x = x.to(self.device)

                output = self.model(x)

                loss = torch.mean((output - x.view(x.size(0),-1))**2, dim=1)

                errors.extend(loss.cpu().numpy())
                labels.extend(y.numpy())

        return np.array(errors), np.array(labels)


    def find_threshold(self, errors):

        return np.percentile(errors, 95)


    def evaluate(self, errors, labels, threshold):

        preds = (errors > threshold).astype(int)

        f1 = f1_score(labels, preds)

        roc = roc_auc_score(labels, errors)

        cm = confusion_matrix(labels, preds)

        report = classification_report(labels, preds)

        return f1, roc, cm, report, preds