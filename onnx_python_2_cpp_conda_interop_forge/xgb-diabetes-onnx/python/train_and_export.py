# train_and_export.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
from onnxmltools.utils import save_model

# FloatTensorType may come from skl2onnx or onnxconverter_common depending on
# installed package versions. Try skl2onnx first for compatibility, fall back
# to onnxconverter_common.
try:
    from skl2onnx.common.data_types import FloatTensorType
except Exception:
    try:
        from onnxconverter_common.data_types import FloatTensorType
    except Exception:
        raise RuntimeError(
            "Could not import FloatTensorType from skl2onnx or onnxconverter_common.\n"
            "Install compatible packages, e.g. pip install skl2onnx onnxconverter-common onnxmltools onnx\n"
        )
import joblib


# FloatTensorType may be provided by either skl2onnx or onnxconverter_common
# Different versions of onnx/onnxconverter-common/onnxmltools can cause
# import errors like "No module named 'onnx.mapping'". Try skl2onnx first
# and fall back to onnxconverter_common. If both fail, raise a clear error
# with suggested pip/conda installation commands.
# try:
#     from skl2onnx.common.data_types import FloatTensorType
# except Exception:
#     try:
#         from onnxconverter_common.data_types import FloatTensorType
#     except Exception as e:
#         raise RuntimeError(
#             "Could not import FloatTensorType from skl2onnx or onnxconverter_common.\n"
#             "This commonly happens when installed versions of onnx, onnxmltools, or onnxconverter-common\n"
#             "are incompatible. Install compatible packages, for example (pip):\n\n"
#             "  pip install onnxmltools onnxconverter-common skl2onnx onnx==1.14.1\n\n"
#             "Or using conda (conda-forge):\n\n"
#             "  conda install -c conda-forge onnxmltools onnxconverter-common skl2onnx onnx\n\n"
#             "After installing, re-run this script.\n"
#         ) from e

def main():
    # 1. Load data
    X, y = load_diabetes(return_X_y=True)
    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # 2. Train XGBoost regressor using scikit-learn wrapper
    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='reg:squarederror',
        verbosity=0,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3. Save native xgboost model (optional)
    joblib.dump(model, "xgb_sklearn_diabetes.joblib")
    print("Saved native sklearn-wrapped XGBoost model -> xgb_sklearn_diabetes.joblib")

    # 4. Convert to ONNX
    # skl2onnx expects the scikit-learn like model and we must provide input type
    initial_type = [('input', FloatTensorType([None, X_train.shape[1]]))]
    # onnx_model = convert_sklearn(model, initial_types=initial_type)
    # onnx.save_model(onnx_model, "xgb_diabetes.onnx")
    # Use the native xgboost Booster when possible to avoid converter bugs
    try:
        booster = model.get_booster()
    except Exception:
        booster = model

    # Try converting the native Booster first (preferred). If it fails due to
    # converter limitations or incompatible xgboost versions, fall back to
    # converting the scikit-learn wrapper via skl2onnx (if available).
    conversion_errors = []
    try:
        onnx_model = convert_xgboost(booster, initial_types=initial_type)
        save_model(onnx_model, "xgb_diabetes.onnx")
        print("Saved ONNX model -> xgb_diabetes.onnx (opset 14) [using Booster]")
    except Exception as e1:
        conversion_errors.append(str(e1))
        # Attempt skl2onnx as a fallback
        try:
            from skl2onnx import convert_sklearn
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            import onnx
            onnx.save_model(onnx_model, "xgb_diabetes.onnx")
            print("Saved ONNX model -> xgb_diabetes.onnx (opset ?) [using skl2onnx fallback]")
        except Exception as e2:
            conversion_errors.append(str(e2))
            raise RuntimeError(
                "Failed to convert XGBoost model to ONNX. Errors:\n- Booster path: {}\n- skl2onnx fallback: {}\n\n".format(conversion_errors[0], conversion_errors[1])
                + "Try installing compatible versions of onnxmltools/onnxconverter-common/skl2onnx/onnx or downgrade xgboost.\n"
            )

    # 5. Evaluate predictions in Python (native xgboost)
    y_pred = model.predict(X_test)
    print("Python XGBoost results:")
    print("  MSE:", mean_squared_error(y_test, y_pred))
    print("  R2: ", r2_score(y_test, y_pred))

    # 6. Quick ONNX runtime validation (optional, needs onnxruntime)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession("xgb_diabetes.onnx", providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        # ensure float32
        X_test_f = X_test.astype(np.float32)
        onnx_pred = sess.run(None, {input_name: X_test_f})[0].ravel()
        print("ONNX runtime results:")
        print("  MSE:", mean_squared_error(y_test, onnx_pred))
        print("  R2: ", r2_score(y_test, onnx_pred))
    except Exception as e:
        print("ONNX runtime validation skipped or failed:", e)

    # Save test data (we will use it in C++) as CSV (float32)
    test_df = pd.DataFrame(X_test.astype(np.float32))
    test_df['target'] = y_test.astype(np.float32)
    test_df.to_csv("diabetes_test_set.csv", index=False)
    print("Saved test set to diabetes_test_set.csv (for C++ inference).")

if __name__ == "__main__":
    main()
