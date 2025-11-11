//#include <C:/Users/ADE/miniconda3/envs/xgb-ort/Library/include/onnxruntime/core\session/onnxruntime_cxx_api.h>

#include <onnxruntime_cxx_api.h>

#include <iostream>

using namespace std;

int main() {
    // Initialize ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    // Create session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Model loaded successfully
    std::cout << "ONNX Runtime environment initialized successfully." << std::endl;
    return 0;
}