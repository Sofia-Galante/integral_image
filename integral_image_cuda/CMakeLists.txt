cmake_minimum_required(VERSION 3.21)
project(integral_image_cuda CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -dlcm=cg")
set(CMAKE_EXE_LINKER_FLAGS "-static")

add_executable(integral_image_cuda main.cu IntegralGeneratorSeq.cu IntegralGeneratorSeq.cuh IntegralGeneratorPar.cu IntegralGeneratorPar.cuh ImageControllerGPU.cu ImageControllerGPU.cuh ../Image.h)

set_target_properties(integral_image_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
