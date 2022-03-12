cmake_minimum_required(VERSION 3.21)
project(integral_image_omp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "-static")

add_executable(integral_image_omp main.cpp IntegralGeneratorOMP.cpp IntegralGeneratorOMP.h ../Image.h ImageController.cpp ImageController.h)
