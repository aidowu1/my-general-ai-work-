# python/train_iris.py
import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "iris_logreg.onnx")

def train_and_export():
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto"))
    ])

    pipeline.fit(X_train, y_train)
    acc = pipeline.score(X_test, y_test)
    print(f"Trained logistic regression pipeline. Test accuracy: {acc:.4f}")

    initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
    onnx.save(onnx_model, MODEL_PATH)
    print(f"Saved ONNX model to {MODEL_PATH}")

if __name__ == "__main__":
    train_and_export()
