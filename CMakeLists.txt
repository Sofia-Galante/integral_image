cmake_minimum_required(VERSION 3.21)
project(integral_image CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(integral_image main.cu Image.cu Image.cuh)

set_target_properties(integral_image PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
