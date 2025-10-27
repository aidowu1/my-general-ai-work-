# train_xgb_diabetes.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

RANDOM_SEED = 42

def main():
    data = load_diabetes()
    X = data['data']
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED)
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        objective='reg:squarederror',
        random_state=RANDOM_SEED,
        verbosity=1
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=False)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Python training Test RMSE: {rmse:.4f}")
    model_filename = "xgb_diabetes_model.json"
    model.save_model(model_filename)
    print(f"Saved model to {model_filename}")
    df_test = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(X_test.shape[1])])
    df_test['target'] = y_test
    csv_test_filename = "xgb_diabetes_test.csv"
    df_test.to_csv(csv_test_filename, index=False)
    print(f"Saved test csv to {csv_test_filename}")
    with open("features.txt", "w") as fh:
        fh.write("\n".join(df_test.columns[:-1]))
    print("Saved features.txt")

if __name__ == "__main__":
    main()
