import torch
import torch.nn as nn
from torchtext.vocab import Vocab 

# import text_processing as tp
import configs as cfg

config = cfg.Config()

class SimpleTextCNN(nn.Module):
    def __init__(self, vocab: Vocab):
        """
        Constructor
        :param vocab: Vocabulary 
        """
        super(SimpleTextCNN, self).__init__()
        self.vocab = vocab

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=self.vocab["<pad>"]
        )

        self.conv1 = nn.Conv1d(config.embedding_dim, config.num_filters, config.kernel_size)
        self.conv2 = nn.Conv1d(config.num_filters, config.num_filters, config.kernel_size)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(self._get_conv_output(), config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, 1)

    def _get_conv_output(self):
        with torch.no_grad():
            x = torch.zeros(1, config.maxlen).long()
            x = self.embedding(x)
            x = x.permute(0, 2, 1)
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))

        return x.squeeze()
