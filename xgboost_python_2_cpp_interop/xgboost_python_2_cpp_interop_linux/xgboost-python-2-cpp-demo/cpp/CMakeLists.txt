cmake_minimum_required(VERSION 3.15)
project(xgb_predict_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED XGBOOST_INCLUDE_DIR)
  message(FATAL_ERROR "Please set -DXGBOOST_INCLUDE_DIR=<path> (path to xgboost include directory)")
endif()

if(NOT DEFINED XGBOOST_LIBRARY)
  message(FATAL_ERROR "Please set -DXGBOOST_LIBRARY=<full-path-to-libxgboost.so>")
endif()

include_directories(${XGBOOST_INCLUDE_DIR})
add_executable(predict_xgb_cpp predict_xgb_cpp.cpp)
target_link_libraries(predict_xgb_cpp PRIVATE ${XGBOOST_LIBRARY})
