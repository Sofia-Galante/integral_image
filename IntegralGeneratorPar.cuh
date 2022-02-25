//
// Created by Sofy on 25/02/2022.
//

#ifndef INTEGRAL_IMAGE_INTEGRALGENERATORPAR_CUH
#define INTEGRAL_IMAGE_INTEGRALGENERATORPAR_CUH


#include "IntegralGenerator.cuh"

__global__ void generateIntegralParV1(int const width, int const height, int const * original, int * result);
__global__ void generateIntegralParV2(int const width, int const height, int const * original, int * result);

/*
class IntegralGeneratorPar : public IntegralGenerator {
public:
    explicit IntegralGeneratorPar(Image &i) : IntegralGenerator(i) {};

    __host__ Image generateIntegral() override;
private:

    __device__ int calculate(int x, int y);
};
*/

#endif //INTEGRAL_IMAGE_INTEGRALGENERATORPAR_CUH
