import torch
import torch.nn as nn
from typing import Any, Dict, Tuple, List

class SimpleDnnClassifier(nn.Module):
    """
    Specification of Simple Deep Neural Network Classifier
    """
  def __init__(
      self, 
      input_dim: int, 
      hidden_dim: int, 
      output_dim: int
  ):
      """
      Constructor
      :param input_dim: Input dimension
      :param hidden_dim: Hidden dimension
      :param output_dim: Output dimension
      """
      super().__init__()
      self.fc1 = nn.Linear(input_dim, hidden_dim)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(hidden_dim, output_dim)

  def forward(self, x: Any):
      """
      Forward pass of the network
      :param: x: Input sequence
      """
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      return x

        
      
model = SimpleClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for step in range(3000):
  optimizer.zero_grad()
  loss = criterion(model(vectors), labels)
  loss.backward()
  optimizer.step()