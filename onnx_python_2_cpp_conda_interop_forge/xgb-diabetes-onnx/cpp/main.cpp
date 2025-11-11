// main.cpp
#include <windows.h>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <memory>

using namespace std;

std::wstring convertUtf8ToUtf16(const std::string &utf8Str) {
    if (utf8Str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), (int)utf8Str.size(), NULL, 0);
    std::wstring utf16Str(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), (int)utf8Str.size(), &utf16Str[0], size_needed);
    return utf16Str;
}


std::vector<std::vector<float>> read_csv(const std::string &filename, std::vector<float> &targets) {
    std::vector<std::vector<float>> rows;
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Cannot open " + filename);
    }
    std::string header;
    std::getline(in, header); // skip header
    std::string line;
    while (std::getline(in, line)) {
        if(line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        std::vector<float> row;
        // There are 11 columns (10 features + target)
        for (int col = 0; col < 11; ++col) {
            if (!std::getline(ss, token, ',')) token = "";
            float v = token.empty() ? 0.0f : std::stof(token);
            if (col < 10) row.push_back(v);
            else targets.push_back(v);
        }
        rows.push_back(row);
    }
    return rows;
}

int main(int argc, char* argv[]) {
    try {
        std::string model_path = "xgb_diabetes.onnx";
        std::string test_csv = "diabetes_test_set.csv";

        // Read test CSV
        std::vector<float> targets;
        auto data = read_csv(test_csv, targets);
        size_t N = data.size();
        if (N == 0) {
            std::cerr << "No data found in " << test_csv << std::endl;
            return 1;
        }
        const size_t feature_count = data[0].size(); // should be 10

        // flatten inputs: row-major
        std::vector<float> input_tensor_values;
        input_tensor_values.reserve(N * feature_count);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < feature_count; ++j) {
                input_tensor_values.push_back(data[i][j]);
            }
        }

        // Create environment
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "xgb_infer");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Create session
        std::wstring wmodel_path = convertUtf8ToUtf16(model_path);
        Ort::Session session(env, wmodel_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        // Get input name & shape
        size_t num_input_nodes = session.GetInputCount();
        if (num_input_nodes != 1) {
            std::cerr << "Expected 1 input, got " << num_input_nodes << std::endl;
        }
        char* input_name = session.GetInputNameAllocated(0, allocator).release();
        Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_node_dims = tensor_info.GetShape();

        // We expect shape [N, feature_count] or [None, feature_count]
        std::vector<int64_t> input_shape = { static_cast<int64_t>(N), static_cast<int64_t>(feature_count) };

        // Create memory info
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Create tensor object from data
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        // Prepare output
        size_t num_output_nodes = session.GetOutputCount();
        char* output_name = session.GetOutputNameAllocated(0, allocator).release();

        // Run
        std::vector<const char*> input_names = { input_name };
        std::vector<const char*> output_names = { output_name };
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

        // Get pointer to output
        float* floatarr = output_tensors[0].GetTensorMutableData<float>();
        auto out_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        auto out_shape = out_info.GetShape();
        size_t out_len = 1;
        for (auto d : out_shape) out_len *= (size_t)d;

        std::cout << "Predictions (first 10):\n";
        for (size_t i = 0; i < std::min<size_t>(10, out_len); ++i) {
            std::cout << floatarr[i] << "\n";
        }

        // Optional: compute MSE and R2 like simple metrics
        double mse = 0.0;
        double sum_y = 0.0;
        double sum_y2 = 0.0;
        for (size_t i = 0; i < N; ++i) {
            double err = floatarr[i] - targets[i];
            mse += err * err;
            sum_y += targets[i];
            sum_y2 += targets[i] * targets[i];
        }
        mse /= static_cast<double>(N);
        double mean_y = sum_y / static_cast<double>(N);
        double ss_tot = 0.0;
        double ss_res = 0.0;
        for (size_t i = 0; i < N; ++i) {
            ss_tot += (targets[i] - mean_y) * (targets[i] - mean_y);
            double err = targets[i] - floatarr[i];
            ss_res += err * err;
        }
        double r2 = 1.0 - (ss_res / ss_tot);

        std::cout << "C++ ONNX Runtime MSE: " << mse << "\n";
        std::cout << "C++ ONNX Runtime R2:  " << r2 << "\n";

        // cleanup allocated names
        allocator.Free(input_name);
        allocator.Free(output_name);
    }
    catch (std::exception & e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
