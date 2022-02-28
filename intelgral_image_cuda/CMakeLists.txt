cmake_minimum_required(VERSION 3.21)
project(integral_image_cuda CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -dlcm=cg")

add_executable(integral_image main.cu Image.cuh IntegralGeneratorSeq.cu IntegralGeneratorSeq.cuh IntegralGeneratorPar.cu IntegralGeneratorPar.cuh ImageController.cu ImageController.cuh)

set_target_properties(integral_image PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
