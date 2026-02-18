import torch
import torch.nn as nn
from torchtext.vocab import Vocab

import configs_advanced as cfg

config = cfg.Config()

class AdvancedTextCNN(nn.Module):
    """
    Advanced text CNN network architecture based on Kim's paper, with multiple kernel sizes and global max pooling.
    It is derived torch.nn.Module and can be used as a standard PyTorch model.
    """
    def __init__(self, vocab: Vocab):
        """
        Constructor
        :param vocab: Vocabulary 
        """
        super(AdvancedTextCNN, self).__init__()
        self.vocab = vocab

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=self.vocab["<pad>"]
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=config.embedding_dim,
                out_channels=config.num_filters,
                kernel_size=k
            )
            for k in config.kernel_sizes
        ])

        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(
            config.num_filters * len(config.kernel_sizes),
            1
        )
        
    def forward_old(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))

        return x.squeeze()
    
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)               # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)              # (batch, embed_dim, seq_len)

        conv_outputs = []

        for conv in self.convs:
            c = torch.relu(conv(x))         # (batch, num_filters, L_out)
            c = torch.max_pool1d(c, c.shape[2])  # Global max pooling
            conv_outputs.append(c.squeeze(2))

        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)

        logits = self.fc(x)                 # NO sigmoid here
        return logits.squeeze(1)
