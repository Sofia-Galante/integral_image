//
// Created by Sofy on 25/02/2022.
//

#ifndef INTEGRAL_IMAGE_INTEGRALGENERATORSEQ_CUH
#define INTEGRAL_IMAGE_INTEGRALGENERATORSEQ_CUH

#include "IntegralGenerator.cuh"

__host__ void generateIntegralSeq(int const width, int const height, int const * original, int * result);

/*
class IntegralGeneratorSeq : public IntegralGenerator{
public:
    explicit IntegralGeneratorSeq(Image& i) : IntegralGenerator(i) {};

    __host__ Image generateIntegral() override;

private:
    __host__ int calculate(int x, int y);
};
*/

#endif //INTEGRAL_IMAGE_INTEGRALGENERATORSEQ_CUH
