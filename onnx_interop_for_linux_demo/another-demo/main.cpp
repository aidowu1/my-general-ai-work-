\
        // src/main.cpp
        #include <iostream>
        #include <vector>
        #include <string>
        #include <memory>
        #include <fstream>
        #include <sstream>
        #include <numeric>
        #include <Windows.h>
        #include <onnxruntime_cxx_api.h>

        std::wstring convertUtf8ToUtf16(const std::string &utf8Str) {
			if (utf8Str.empty()) {
				return std::wstring();
			}
			int size_needed = MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), static_cast<int>(utf8Str.size()), NULL, 0);
			std::wstring utf16Str(size_needed, 0);
			MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), static_cast<int>(utf8Str.size()), &utf16Str[0], size_needed);
			return utf16Str;
		}


        template <typename T>
        T vectorProduct(const std::vector<T>& v)
        {
            return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
        }

        const char* copyString(std::string s)
        {
            const char* s2;

            // std::string::c_str() method
            s2 = s.c_str();

            return s2;
        }
        
        // Simple CSV loader: expects each row to contain 4 float features.
        std::vector<std::vector<float>> load_csv(const std::string& path) {
            std::vector<std::vector<float>> samples;
            std::ifstream ifs(path);
            if (!ifs) {
                throw std::runtime_error("Cannot open CSV file: " + path);
            }
            std::string line;
            while (std::getline(ifs, line)) {
                if(line.empty()) continue;
                std::stringstream ss(line);
                std::vector<float> row;
                std::string cell;
                while (std::getline(ss, cell, ',')) {
                    if(cell.size()==0) continue;
                    row.push_back(std::stof(cell));
                }
                if (!row.empty()) {
                    if (row.size() != 4) {
                        throw std::runtime_error("Each row must have 4 features. Found: " + std::to_string(row.size()));
                    }
                    samples.push_back(row);
                }
            }
            return samples;
        }

        int main(int argc, char** argv) {
            std::string model_path = "model.onnx";
            std::vector<std::string> labels = { "setosa", "versicolor", "virginica" };
            int num_classes = (int)(labels.size());
            std::string csv_path;
            if (argc > 1) csv_path = argv[1];

            // default sample if no CSV provided
            std::vector<std::vector<float>> samples;
            if (!csv_path.empty()) {
                try {
                    samples = load_csv(csv_path);
                } catch (const std::exception &ex) {
                    std::cerr << "Error loading CSV: " << ex.what() << std::endl;
                    return 2;
                }
            } else {
                samples = {{5.1f,3.5f,1.4f,0.2f}, {6.0f, 2.9f ,4.5f, 1.5f}};
                std::cout << "No CSV provided. Using default sample." << std::endl;
            }

            std::string instance_name{ "XGBoostInference" };

            try {

                //Create ORT Environment
                std::shared_ptr<Ort::Env> env;
                std::shared_ptr<Ort::Session> session;
                std::string input_name;
                std::vector<int64_t> input_dims;    // shape

                // Outputs
                std::string output_name;
                std::vector<int64_t> output_dims;    // shape

                //member function  label
                std::string m_label;

                // Set up options for session
                Ort::SessionOptions sessionOptions;
                sessionOptions.SetIntraOpNumThreads(1);

                // Set up environment
                env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, instance_name.c_str());

                // Create session
			   std::wstring wide_model_path = convertUtf8ToUtf16(model_path);
               session = std::make_shared<Ort::Session>(*env, wide_model_path.c_str(), sessionOptions);

                // Create session (old version)
                // session = std::make_shared<Ort::Session>(*env, model_path.c_str(), sessionOptions);

                // Specify the allocator
                Ort::AllocatorWithDefaultOptions allocator;

                // ********************************************************************************
                // Get input node details (i.e. names, shapes etc.)
                size_t num_input_nodes = session->GetInputCount();
                Ort::AllocatedStringPtr p_str_input_name = session->GetInputNameAllocated(0, allocator.operator OrtAllocator * ());
                input_name = p_str_input_name.get();

                // Input type
                Ort::TypeInfo input_type_info = session->GetInputTypeInfo(0);
                auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
                ONNXTensorElementDataType input_type = input_tensor_info.GetElementType();

                // Input shape
                input_dims = input_tensor_info.GetShape();

                // Report input details
                std::cout << "---------------------------------------" << std::endl;
                std::cout << "Number of Input Nodes: " << num_input_nodes << std::endl;
                std::cout << "Input Name: " << input_name << std::endl;
                std::cout << "Input Type: " << input_type << std::endl;
                std::cout << "Input Dimensions: " << input_dims.size() << std::endl;
                std::cout << "\n\n\n";

                // ********************************************************************************
                // Get output node details (i.e. names, shapes etc.)
                size_t num_output_nodes = session->GetOutputCount();
                Ort::AllocatedStringPtr p_str_output_name = session->GetOutputNameAllocated(0, allocator.operator OrtAllocator * ());
                output_name = p_str_output_name.get();

                // Output type
                Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(0);
                auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
                ONNXTensorElementDataType output_type = output_tensor_info.GetElementType();

                // Output shape
                output_dims = output_tensor_info.GetShape();

                // Report output details
                std::cout << "---------------------------------------" << std::endl;
                std::cout << "Number of Output Nodes: " << num_output_nodes << std::endl;
                std::cout << "Output Name: " << output_name << std::endl;
                std::cout << "Output Type: " << output_type << std::endl;
                std::cout << "Output Dimensions: " << output_dims.size() << std::endl;
                std::cout << "\n\n\n";

                // Setup input/output names vectors
                int num_datasets_for_prediction = 1;
                std::vector<const char*> inputNames{ input_name.c_str() };
                std::vector<const char*> outputNames{ output_name.c_str() };

                // Prepare input data tensor
                input_dims[0] = num_datasets_for_prediction;
                size_t input_tensor_size = vectorProduct(input_dims);
                std::vector<float> inputTensorValues(input_tensor_size);
                inputTensorValues.assign(samples[0].begin(), samples[0].end());
                Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
                auto input_tensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                    input_tensor_size, input_dims.data(), input_dims.size());

                // Invoke the model prediction run
                auto output_tensors = session->Run(
                    Ort::RunOptions{ nullptr }, 
                    inputNames.data(), 
                    &input_tensor, 
                    num_datasets_for_prediction,
                    outputNames.data(), 
                    num_datasets_for_prediction);

                // Get the inference result
                float* outArr = output_tensors.front().GetTensorMutableData<float>();
                
                // score the model, and print scores for first 5 classes
                for (int i = 0; i < num_classes; i++) 
                {
                    std::cout << "Score for class [" << i << "] =  " << outArr[i] << '\n';
                }
                
            } 
            catch (const Ort::Exception &ex) {
                std::cerr << "ONNX Runtime error: " << ex.what() << "\\n";
                return 2;
            } catch (const std::exception &ex) {
                std::cerr << "STD exception: " << ex.what() << "\\n";
                return 3;
            }
            return 0;
        }
