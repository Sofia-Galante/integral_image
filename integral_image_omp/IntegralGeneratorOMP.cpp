//
// Created by Sofy on 28/02/2022.
//

#include "IntegralGeneratorOMP.h"

void generateIntegralCPUomp(int width, int height, int const * original, int * result){
#pragma omp parallel for collapse(2) default(none) shared(original, result) firstprivate(width, height)
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            int value = 0;
#pragma omp parallel for collapse(2) default(none) shared(original) firstprivate(row, col, width) reduction(+: value)
            for(int y = row; y >= 0; y--){
                for(int x = col; x >= 0; x--){
                    value += original[y * width + x];
                }
            }
            result[row * width + col] = value;
        }
    }
}