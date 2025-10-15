cmake_minimum_required(VERSION 3.20)
project(xgb_predict_cpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Adjust these paths to where xgboost installed
set(XGBOOST_INCLUDE_DIR "xgboost/include")
set(XGBOOST_LIBRARY_DIR "xgboost/lib")

include_directories(${XGBOOST_INCLUDE_DIR})
link_directories(${XGBOOST_LIBRARY_DIR})

add_executable(predict_xgb_cpp predict_xgb_cpp.cpp)
target_link_libraries(predict_xgb_cpp PRIVATE xgboost) # or xgboost.lib
