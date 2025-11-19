from sklearn.linear_model import LogisticRegression
import numpy as np

import modules.configs as cfg



class LogisticRegressionClassifier:
    """
    Logistic Regression classifer
    """
    def __init__(self, random_seed: int = cfg.RANDOM_SEED):
        """
        Initialize the Logistic Regression model.
        :param random_seed: Random seed for reproducibility 
        """
        self.model = LogisticRegression(random_state=random_seed)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the Logistic Regression model.
        :param X_train: Training features
        :param y_train: Training labels        
        """
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained Logistic Regression model.
        :param X: Input features for prediction
        :return: Predicted labels
        """
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate the model on test data.
        :param X_test: Test features
        :param y_test: Test labels
        :return: Accuracy of the model on test data
        """
        accuracy = self.model.score(X_test, y_test)
        return accuracy