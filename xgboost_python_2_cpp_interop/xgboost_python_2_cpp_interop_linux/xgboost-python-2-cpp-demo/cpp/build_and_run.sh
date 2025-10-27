#!/usr/bin/env bash
set -e
echo "Detecting XGBoost include and library..."
PY_OUT=$(python3 get_xgboost_paths.py)
INC=$(echo "$PY_OUT" | sed -n '1p')
LIB=$(echo "$PY_OUT" | sed -n '2p')

echo "Include: $INC"
echo "Lib:     $LIB"

if [ -z "$INC" ] || [ -z "$LIB" ]; then
  echo "Could not detect XGBoost include or library. Please ensure python xgboost is installed."
  exit 1
fi

# Train model (creates xgb_diabetes_model.json and xgb_diabetes_test.csv)
echo "Training Python model..."
python3 train_xgb_diabetes.py

# Create build dir and run cmake with detected paths
mkdir -p build && cd build
cmake .. -DXGBOOST_INCLUDE_DIR="$INC" -DXGBOOST_LIBRARY="$LIB" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Copy model & csv into build if needed, run binary
cp ../xgb_diabetes_model.json .
cp ../xgb_diabetes_test.csv .

echo "Running C++ inference..."
./predict_xgb_cpp xgb_diabetes_test.csv xgb_diabetes_model.json
