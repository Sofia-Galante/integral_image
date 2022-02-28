//
// Created by Sofy on 25/02/2022.
//

#ifndef INTEGRAL_IMAGE_INTEGRALGENERATORPAR_CUH
#define INTEGRAL_IMAGE_INTEGRALGENERATORPAR_CUH

#include "ImageController.cuh"


__global__ void generateIntegralGPUglobalMem(int width, int height, int const * original, int * result);
__global__ void generateIntegralGPUsharedMem(int width, int height, int const * original, int * result);

__host__ void setUp(Image const &original, Image const &result, Image &dev_original, Image &dev_result);
__host__ void finish(Image &dev_original, Image &dev_result, Image &result);

__host__ void findGridAndBlockDim(int width, int height, int &gx, int &gy, int &bx, int &by);

#endif //INTEGRAL_IMAGE_INTEGRALGENERATORPAR_CUH
