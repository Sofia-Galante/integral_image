//
// Created by Sofy on 28/02/2022.
//

#ifndef INTEGRAL_IMAGEOMP_INTEGRALGENERATOROMP_H
#define INTEGRAL_IMAGEOMP_INTEGRALGENERATOROMP_H

#include <omp.h>
#include <vector>

void generateIntegralCPUomp(int width, int height, int const * original, int * result, int threads1, int threads2);

#endif //INTEGRAL_IMAGEOMP_INTEGRALGENERATOROMP_H
