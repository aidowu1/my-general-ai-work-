import torch
import torch.nn as nn

from modules.simple_dnn_classifier import SimpleClassifier

class ClassifierModel:
    """
    Classifier model
    """
    def __init__(
      self,
      vectors: torch.Tensor,
      labels: torch.Tensor,
      network: SimpleClassifier,
      batch_size: int = 32,
      learn_rate: int = 0.001,
      epochs: int = 10
    ):
        """
        Constructor
        :param vectors: Input vectors
        :param labels: Input labels
        :param network: Neural network
        :param batch_size: Batch size
        :param learn_rate: Learning rate
        :param epochs: Number of epochs
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vectors = vectors
        self._labels = labels
        self._batch_size = batch_size
        self._epochs = epochs
        self.train_dataset = torch.utils.data.TensorDataset(self._vectors, self._labels)    
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self._batch_size, shuffle=True)
        self._model = network.to(self._device)
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=learn_rate)

    def _accuracy(
        self,
        outputs, 
        labels
    ):
        """
        Calculates model accuracy
        :param outputs: Model outputs
        :param labels: True labels  
        :return: Accuracy
        """
        _, preds = torch.max(outputs, 1)
        return torch.sum(preds == labels).item() / len(labels)

    def train(self, epoch: int):
        """
        Trains the model
        """
        running_loss = 0.0
        running_acc = 0.0
        self._model.train()
        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self._device), labels.to(self._device)
            self._optimizer.zero_grad()
            outputs = self._model(inputs)
            loss = self._criterion(outputs, labels)
            loss.backward()
            self._optimizer.step()
            
            running_loss += loss.item()
            running_acc += self._accuracy(outputs, labels)
            if (i + 1) % 200 == 0:
                print(
                    f'Epoch {epoch}, Batch {i+1}, Loss: {running_loss / 200:.4f}, Accuracy: {running_acc / 200:.4f}')
                running_loss = 0.0
                running_acc = 0.0

    def test(
        self,
        test_vectors: torch.Tensor,
        test_labels: torch.Tensor
    ):
        """
        Tests the model
        :param test_vectors: Test input vectors
        :param test_labels: Test input labels
        :return: Test accuracy
        """
        test_dataset = torch.utils.data.TensorDataset(test_vectors, test_labels)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False)
        
        test_loss = 0.0
        test_acc = 0.0

        self._model.eval()
        total_acc = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                outputs = self._model(inputs)
                total_acc += self._accuracy(outputs, labels) * len(labels)
                loss = self._criterion(outputs, labels)
                test_loss += loss.item()
                test_acc += self._accuracy(outputs, labels)
                print(
                    f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_acc / len(test_loader):.4f}')        
        return total_acc / len(test_dataset)
    
    def run_train_and_test(
        self,
        test_vectors: torch.Tensor,
        test_labels: torch.Tensor
    ):
        """
        Runs training and testing
        :param test_vectors: Test input vectors
        :param test_labels: Test input labels
        :return: Test accuracy
        """
        for epoch in range(1, self._epochs + 1):
            self.train(epoch)
        test_accuracy = self.test(test_vectors, test_labels)
        return test_accuracy
          


   
      