// main.cpp
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

static std::vector<std::vector<float>> read_csv_features_and_targets(const std::string &path, std::vector<float> &targets_out) {
    std::vector<std::vector<float>> features;
    targets_out.clear();
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open CSV file: " + path);
    }

    std::string line;
    // Read header
    if (!std::getline(in, line)) {
        throw std::runtime_error("CSV is empty: " + path);
    }

    // Parse rows
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;
        while (std::getline(ss, cell, ',')) {
            // convert
            try {
                row.push_back(std::stof(cell));
            } catch (...) {
                row.push_back(0.0f);
            }
        }
        if (row.size() < 2) continue; // need at least one feature + target
        // last column is target
        float target = row.back();
        row.pop_back();
        targets_out.push_back(target);
        features.push_back(row);
    }
    return features;
}

int main(int argc, char** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "diabetes_reg.onnx";
    const std::string csv_path = argc > 2 ? argv[2] : "diabetes_test.csv";

    try {
        // Read CSV
        std::vector<float> targets;
        auto features = read_csv_features_and_targets(csv_path, targets);
        if (features.empty()) {
            std::cerr << "No test rows found in " << csv_path << std::endl;
            return 1;
        }
        size_t N = features.size();
        size_t num_features = features[0].size();

        std::cout << "Loaded " << N << " test rows with " << num_features << " features each." << std::endl;

        // Flatten features into single contiguous vector (row-major): [r0f0..r0fM-1, r1f0..]
        std::vector<float> input_tensor_values;
        input_tensor_values.reserve(N * num_features);
        for (size_t i = 0; i < N; ++i) {
            if (features[i].size() != num_features) {
                throw std::runtime_error("Row " + std::to_string(i) + " has inconsistent feature count.");
            }
            for (size_t j = 0; j < num_features; ++j) {
                input_tensor_values.push_back(features[i][j]);
            }
        }

        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "batch-infer");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        Ort::Session session(env, model_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        // Input names
        size_t num_input_nodes = session.GetInputCount();
        if (num_input_nodes != 1) {
            std::cout << "Warning: model has " << num_input_nodes << " inputs; this client expects a single input tensor." << std::endl;
        }
        char* input_name = session.GetInputName(0, allocator);
        std::cout << "Input name: " << (input_name ? input_name : "<null>") << std::endl;

        // Prepare shapes: [N, num_features]
        std::vector<int64_t> input_shape = { static_cast<int64_t>(N), static_cast<int64_t>(num_features) };

        // Create tensor
        size_t input_tensor_size = input_tensor_values.size();
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
                                                                  input_tensor_size, input_shape.data(), input_shape.size());

        // Output names
        size_t num_output_nodes = session.GetOutputCount();
        std::vector<const char*> output_node_names;
        std::vector<char*> output_names_owned;
        for (size_t i = 0; i < num_output_nodes; ++i) {
            char* out_name = session.GetOutputName(i, allocator);
            output_node_names.push_back(out_name);
            output_names_owned.push_back(out_name);
            std::cout << "Output[" << i << "] name: " << (out_name ? out_name : "<null>") << std::endl;
        }

        // Run inference (single call for full batch)
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                          &input_name, &input_tensor, 1,
                                          output_node_names.data(), output_node_names.size());

        // We assume single output
        if (output_tensors.size() < 1) {
            throw std::runtime_error("Model returned no outputs.");
        }

        // Extract output tensor data (could be shape [N] or [N,1])
        auto &out0 = output_tensors.front();
        auto out_typeinfo = out0.GetTensorTypeAndShapeInfo();
        auto out_shape = out_typeinfo.GetShape();
        size_t total_outputs = 1;
        for (auto d : out_shape) total_outputs *= (d < 0 ? 1 : static_cast<size_t>(d)); // treat negative dims as 1 (safety)
        float* out_data = out0.GetTensorMutableData<float>();

        std::cout << "Output tensor shape: [";
        for (size_t i = 0; i < out_shape.size(); ++i) {
            std::cout << out_shape[i] << (i+1 < out_shape.size() ? ", " : "");
        }
        std::cout << "] total elements: " << total_outputs << std::endl;

        // Our expectation: total_outputs == N or N*1
        if (total_outputs < N) {
            std::cerr << "Warning: fewer outputs than input rows (" << total_outputs << " < " << N << ")." << std::endl;
        }

        // Compute RMSE (use min(N, total_outputs) rows)
        size_t evaln = std::min(static_cast<size_t>(total_outputs), N);
        double sumsq = 0.0;
        for (size_t i = 0; i < evaln; ++i) {
            double pred = static_cast<double>(out_data[i]);
            double gt = static_cast<double>(targets[i]);
            double diff = pred - gt;
            sumsq += diff * diff;
        }
        double rmse = std::sqrt(sumsq / evaln);
        std::cout << "Evaluated " << evaln << " rows. RMSE = " << rmse << std::endl;

        // Print first min(10, evaln) comparisons
        size_t toshow = std::min(evaln, static_cast<size_t>(10));
        std::cout << "First " << toshow << " predictions (predicted | expected):" << std::endl;
        for (size_t i = 0; i < toshow; ++i) {
            std::cout << i << ": " << out_data[i] << " | " << targets[i] << std::endl;
        }

        // free names allocated by Ort
        allocator.Free(input_name);
        for (auto p : output_names_owned) allocator.Free(p);

    } catch (const std::exception &ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 2;
    }

    return 0;
}
