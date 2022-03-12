//
// Created by Sofy on 25/02/2022.
//

#ifndef INTEGRAL_IMAGE_INTEGRALGENERATORPAR_CUH
#define INTEGRAL_IMAGE_INTEGRALGENERATORPAR_CUH

#include "ImageControllerGPU.cuh"


__global__ void generateIntegralGPUglobalMem(int width, int height, int const * original, int * result);
__global__ void generateIntegralGPUsharedMem(int width, int height, int const * original, int * result);

__host__ void setUp(Image const &original, Image const &result, Image &dev_original, Image &dev_result);
__host__ void finish(Image &dev_original, Image &dev_result, Image &result);
#endif //INTEGRAL_IMAGE_INTEGRALGENERATORPAR_CUH
