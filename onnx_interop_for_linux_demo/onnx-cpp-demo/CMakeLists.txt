cmake_minimum_required(VERSION 3.16)
project(onnx_demo VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Try to find headers and library in the common locations we copied in the Dockerfile
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h PATHS /usr/local/include /opt/onnxruntime/build/native/include /opt/onnxruntime/include)
find_library(ONNXRUNTIME_LIB NAMES onnxruntime libonnxruntime PATHS /usr/local/lib /opt/onnxruntime/runtimes/linux-x64/native /opt/onnxruntime/native)

if(NOT ONNXRUNTIME_INCLUDE_DIR)
  message(FATAL_ERROR "ONNX Runtime headers not found. Set ONNXRUNTIME_INCLUDE_DIR.")
endif()
if(NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "ONNX Runtime library not found. Set ONNXRUNTIME_LIB.")
endif()

message(STATUS "ONNX include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX lib: ${ONNXRUNTIME_LIB}")

add_executable(onnx_demo src/main.cpp)
target_include_directories(onnx_demo PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(onnx_demo PRIVATE ${ONNXRUNTIME_LIB})

# ensure runtime linker finds the .so in /usr/local/lib inside the container
set_target_properties(onnx_demo PROPERTIES INSTALL_RPATH "/usr/local/lib")
