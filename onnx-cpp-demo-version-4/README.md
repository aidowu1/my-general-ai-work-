# ONNX Iris Demo (Devcontainer)

Contents:
- .devcontainer/ : Dockerfile + devcontainer.json for VS Code Dev Containers
- python/ : training and python prediction clients
- models/ : generated ONNX model (created by train_iris.py)
- src/ : C++ ONNX Runtime client (accepts CSV)
- CMakeLists.txt : builds the C++ client

Quick start (inside devcontainer):
1. Train & export ONNX model:
   python3 python/train_iris.py
2. Test with Python:
   python3 python/predict_py.py
   or provide CSV: python3 python/predict_py.py samples.csv
3. Build C++:
   mkdir -p build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build .
4. Run C++:
   ./onnx_demo                 # default sample
   ./onnx_demo /workspace/samples.csv
