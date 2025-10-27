# train_xgb_diabetes.py
# Python 3.8+ recommended
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import xgboost as xgb

RANDOM_SEED = 42

def main():
    # 1) Load dataset
    data = load_diabetes()
    X = data['data']               # shape (442, 10)
    y = data['target']             # shape (442,)

    # 2) Train/test split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )

    # 3) Train XGBoost regressor (scikit-learn API wrapper)
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        objective='reg:squarederror',
        random_state=RANDOM_SEED,
        verbosity=1,
        early_stopping_rounds=20
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # 4) Evaluate in Python
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Test RMSE: {rmse:.4f}")

    # 5) Persist model in JSON format (recommended for cross-language)
    model_filename = "xgb_diabetes_model.json"
    # `save_model` saves JSON by extension and is compatible with XGBoost C API load.
    model.save_model(model_filename)
    print(f"Saved model to {model_filename}")

    # 6) Save test partition to CSV for C++ inference:
    # We'll save feature columns then target column named 'target'
    df_test = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(X_test.shape[1])])
    df_test['target'] = y_test
    csv_test_filename = "xgb_diabetes_test.csv"
    df_test.to_csv(csv_test_filename, index=False)
    print(f"Saved test CSV to {csv_test_filename} (n={len(df_test)})")

    # 7) Also save a small header file describing feature order if you want
    with open("features.txt", "w") as fh:
        fh.write("\n".join(df_test.columns[:-1]))  # feature names
    print("Saved features.txt (feature column order)")

if __name__ == "__main__":
    main()
