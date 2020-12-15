cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cfd-cuda LANGUAGES CXX CUDA)

add_executable(cfd-cuda
        main.cu
        boundary.cpp
        jacobi.cu
        cfdio.cpp)

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to
# particles will also build with -std=c++11
target_compile_features(cfd-cuda PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the
# particle library to be built with -dc as the member functions
# could be called by other libraries and executables
set_target_properties(cfd-cuda
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)