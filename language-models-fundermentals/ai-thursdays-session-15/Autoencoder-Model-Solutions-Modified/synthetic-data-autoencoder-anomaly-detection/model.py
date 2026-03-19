# model.py

import torch
import torch.nn as nn


class CircularAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(2,16),
            nn.ReLU(),

            nn.Linear(16,8),
            nn.ReLU(),

            nn.Linear(8,2)
        )

        self.decoder = nn.Sequential(

            nn.Linear(2,8),
            nn.ReLU(),

            nn.Linear(8,16),
            nn.ReLU(),

            nn.Linear(16,2)
        )

    def forward(self, x):

        latent = self.encoder(x)

        reconstruction = self.decoder(latent)

        return reconstruction