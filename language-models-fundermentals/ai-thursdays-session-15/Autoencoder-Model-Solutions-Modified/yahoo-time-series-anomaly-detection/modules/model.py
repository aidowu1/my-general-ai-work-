# model.py

import torch
import torch.nn as nn

class TimeSeriesAutoencoder(nn.Module):

    def __init__(self, seq_len, hidden_dim, latent_dim):

        super().__init__()

        input_dim = seq_len

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):

        x = x.view(x.size(0), -1)

        latent = self.encoder(x)

        reconstruction = self.decoder(latent)

        return reconstruction