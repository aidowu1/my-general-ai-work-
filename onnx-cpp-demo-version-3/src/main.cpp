#include <iostream>
#include "onnxruntime_cxx_api.h"

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-demo");
        std::cout << "ONNX Runtime environment created OK.\n";
        // show a tiny hint of API existence:
        std::cout << "Ort C++ API available.\n";
    } catch (const std::exception &ex) {
        std::cerr << "ONNX Runtime exception: " << ex.what() << "\n";
        return 2;
    }
    return 0;
}
