# trainer.py

import torch
import numpy as np


class Trainer:

    def __init__(self, model, optimizer, criterion, device):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, loader):

        self.model.train()

        losses = []

        for x,_ in loader:

            x = x.to(self.device)

            self.optimizer.zero_grad()

            recon = self.model(x)

            loss = self.criterion(recon, x)

            loss.backward()

            self.optimizer.step()

            losses.append(loss.item())

        return np.mean(losses)

    def validate_epoch(self, loader):

        self.model.eval()

        losses = []

        with torch.no_grad():

            for x,_ in loader:

                x = x.to(self.device)

                recon = self.model(x)

                loss = self.criterion(recon, x)

                losses.append(loss.item())

        return np.mean(losses)
    
