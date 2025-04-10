cmake_minimum_required(VERSION 3.17)
project(FAISS_LITE LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(faiss REQUIRED)

add_library(faiss_lite SHARED faiss_lite.cu)
target_link_libraries(faiss_lite faiss)

add_executable(test test.cu)
target_link_libraries(test faiss_lite)