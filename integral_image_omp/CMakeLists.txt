cmake_minimum_required(VERSION 3.21)
project(integral_image_omp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

add_executable(integral_imageOMP main.cpp IntegralGeneratorOMP.cpp IntegralGeneratorOMP.h ImageController.cpp ImageController.h Image.h)
