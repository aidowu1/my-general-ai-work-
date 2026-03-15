# trainer.py

import torch
import numpy as np
from tqdm import tqdm


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

        for x,_ in tqdm(loader):

            x = x.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(x)

            loss = self.criterion(output, x.view(x.size(0), -1))

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

                output = self.model(x)

                loss = self.criterion(output, x.view(x.size(0), -1))

                losses.append(loss.item())

        return np.mean(losses)

    def fit(self, train_loader, val_loader, epochs):

        for epoch in range(epochs):

            train_loss = self.train_epoch(train_loader)

            val_loss = self.validate_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(
                f"Epoch {epoch+1} | "
                f"Train Loss {train_loss:.5f} | "
                f"Val Loss {val_loss:.5f}"
            )