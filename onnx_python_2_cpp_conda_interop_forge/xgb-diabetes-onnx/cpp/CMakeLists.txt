cmake_minimum_required(VERSION 3.15)
project(xgb_onnx_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# User must pass ONNXRUNTIME_DIR pointing to the conda env (or auto-detect)
if(NOT DEFINED ONNXRUNTIME_DIR)
    message(STATUS "ONNXRUNTIME_DIR not provided. Trying to use CONDA_PREFIX.")
    if(DEFINED ENV{CONDA_PREFIX})
        set(ONNXRUNTIME_DIR $ENV{CONDA_PREFIX})
    endif()
endif()

if(NOT ONNXRUNTIME_DIR)
    message(FATAL_ERROR "ONNXRUNTIME_DIR is not set. Set it to your conda env prefix that contains onnxruntime headers and libs.")
endif()

message(STATUS "Using ONNXRUNTIME_DIR = ${ONNXRUNTIME_DIR}")

# Typical Conda layout on Windows: 
# headers in <env>/Library/include or <env>/include
# libs in <env>/Library/lib or <env>/Libs
# Try a few common locations
set(ORT_INCLUDE_DIRS
    "${ONNXRUNTIME_DIR}/include"
    "${ONNXRUNTIME_DIR}/Library/include"
    "${ONNXRUNTIME_DIR}/Library/include/onnxruntime"
)
set(ORT_LIB_DIRS
    "${ONNXRUNTIME_DIR}/Library/lib"
    "${ONNXRUNTIME_DIR}/Libs"
    "${ONNXRUNTIME_DIR}/lib"
)

# find include dir
set(ORT_FOUND_INCLUDE "")
foreach(p ${ORT_INCLUDE_DIRS})
    if(EXISTS "${p}")
        set(ORT_FOUND_INCLUDE ${p})
        break()
    endif()
endforeach()

# find lib dir that contains onnxruntime.lib (windows) or libonnxruntime.so (linux)
set(ORT_FOUND_LIB "")
foreach(p ${ORT_LIB_DIRS})
    if(EXISTS "${p}")
        # crude check: look for lib in dir
        if (EXISTS "${p}/onnxruntime_conda.lib" OR EXISTS "${p}/libonnxruntime.so" OR EXISTS "${p}/libonnxruntime.dylib")
            set(ORT_FOUND_LIB ${p})
            break()
        endif()
    endif()
endforeach()

if(NOT ORT_FOUND_INCLUDE OR NOT ORT_FOUND_LIB)
    message("include: ${ORT_FOUND_INCLUDE}")
    message("lib: ${ORT_FOUND_LIB}")
    message(FATAL_ERROR "Could not locate ONNX Runtime includes/libs under ${ONNXRUNTIME_DIR}. Check your Conda env and pass ONNXRUNTIME_DIR pointing to the conda env folder.")
endif()

message(STATUS "ONNX Runtime include: ${ORT_FOUND_INCLUDE}")
message(STATUS "ONNX Runtime lib: ${ORT_FOUND_LIB}")

# Set the onnxruntime_cxx_api.h header located ${ONNXRUNTIME_DIR}\\Library\\include\\onnxruntime\\core\\session\\onnxruntime_cxx_api.h 
# set(HEADERS, ${ONNXRUNTIME_DIR}\\Library\\include\\onnxruntime\\core\\session\\onnxruntime_cxx_api.h)
set(HEADERS_DIR ${ONNXRUNTIME_DIR}\\Library\\include\\onnxruntime\\core\\session)

# Setup target
add_executable(xgb_onnx_infer main.cpp)

# include and link
target_include_directories(xgb_onnx_infer PRIVATE "${ORT_FOUND_INCLUDE}" "${HEADERS_DIR}")
target_link_directories(xgb_onnx_infer PRIVATE "${ORT_FOUND_LIB}")

# On Windows link with onnxruntime.lib; on Unix libonnxruntime.so
if(WIN32)
    target_link_libraries(xgb_onnx_infer PRIVATE onnxruntime_conda)
else()
    target_link_libraries(xgb_onnx_infer PRIVATE onnxruntime)
endif()

# Ensure runtime search path for shared libraries (for local run)
if(UNIX)
    set_target_properties(xgb_onnx_infer PROPERTIES
        INSTALL_RPATH "${ORT_FOUND_LIB}"
    )
endif()
