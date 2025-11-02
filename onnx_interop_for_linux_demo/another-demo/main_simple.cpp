#include <iostream>
#include <vector>
#include <numeric>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "Starting ONNX Runtime C++ Inference" << std::endl;

    // 1. Initialize the environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "XGBoostInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // 2. Define the model path
    const char* model_path = "model.onnx";

    // 3. Create the session
    // NOTE: On Windows, the path needs to be wchar_t* if using Ort::Session::Session(env, model_path, options)
    // Here we convert it for compatibility.
#ifdef _WIN32
    std::wstring wide_model_path(model_path, model_path + strlen(model_path));
    Ort::Session session(env, wide_model_path.c_str(), session_options);
#else
    Ort::Session session(env, model_path, session_options);
#endif

    // 4. Get input and output names (optional but good practice)
    Ort::AllocatorWithDefaultOptions allocator;
    const char* input_name = session.GetInputNameAllocated(0, allocator).get();
    const char* output_name = session.GetOutputNameAllocated(0, allocator).get();

    std::cout << "Input Name: " << input_name << ", Output Name: " << output_name << std::endl;

    // 5. Prepare input data
    // Assuming your XGBoost model expects a float input tensor
    // For demonstration, we use a single sample with 2 features.
    // Adjust dimensions based on your actual model's input shape.
    std::vector<float> input_data = {5.1f, 3.5f, 1.4f, 0.2f}; // Example features
    std::vector<int64_t> input_shape = {1, 4};   // Batch size 1, 2 features

    size_t input_tensor_size = input_data.size();
    
    // Define memory info
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Create input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        input_data.data(), 
        input_tensor_size, 
        input_shape.data(), 
        input_shape.size()
    );

    // 6. Run inference
    std::vector<Ort::Value> output_tensors;
    try {
        output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);
    } catch (const Ort::Exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return 1;
    }

    // 7. Get output data
    // XGBoost regressor typically outputs a single float value per sample
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_elements = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    std::cout << "Inference successful." << std::endl;
    std::cout << "Output value(s): ";
    for (size_t i = 0; i < output_elements; ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
