cmake_minimum_required(VERSION 3.12)
project(OnnxXGBoostInference CXX)

set(CMAKE_CXX_STANDARD 17)

# --- Configure Onnxruntime Path ---
set(ONNX_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-1.19.0")

# Include directories for Onnxruntime
include_directories(${ONNX_RUNTIME_DIR}/include)

message(STATUS "Found ONNX directory here: ${ONNX_RUNTIME_DIR}")

# Set library paths
if(WIN32)
    set(ONNX_RUNTIME_LIB "${ONNX_RUNTIME_DIR}/lib/onnxruntime.lib")
else()
    set(ONNX_RUNTIME_LIB "${ONNX_RUNTIME_DIR}/lib/libonnxruntime.so")
endif()

# Add an executable target
add_executable(XGBoostInference main.cpp)

# Link the onnxruntime libraries to your executable
target_link_libraries(XGBoostInference PRIVATE ${ONNX_RUNTIME_LIB})

# Add the runtime path for Windows
if(WIN32)
    add_custom_command(TARGET XGBoostInference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNX_RUNTIME_DIR}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:XGBoostInference>)
endif()