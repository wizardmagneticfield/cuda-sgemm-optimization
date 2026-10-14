cmake_minimum_required(VERSION 3.18)
project(cuda_sgemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# RTX 4090 = Ada Lovelace, sm_89. Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(sgemm src/main.cu)
target_include_directories(sgemm PRIVATE src)
target_compile_options(sgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math>)
target_link_libraries(sgemm PRIVATE CUDA::cublas)
