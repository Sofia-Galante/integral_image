//
// Created by Sofy on 28/02/2022.
//

#include "IntegralGeneratorOMP.h"

#include "iostream"

void generateIntegralCPUompFirstLevel(int width, int height, int const * original, int * result, int threads){
#pragma omp parallel for collapse(2) default(none) shared(original, result) firstprivate(width, height) \
num_threads(threads)
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            int value = 0;
            for(int y = row; y >= 0; y--){
                for(int x = col; x >= 0; x--){
                    value += original[y * width + x];
                }
            }
            result[row * width + col] = value;
        }
    }
}

void generateIntegralCPUompSecondLevel(int width, int height, int const * original, int * result, int threads){
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            int value = 0;
#pragma omp parallel for collapse(2) default(none) shared(original) firstprivate(row, col, width) \
num_threads(threads) reduction(+: value)
            for(int y = row; y >= 0; y--){
                for(int x = col; x >= 0; x--){
                    value += original[y * width + x];
                }
            }
            result[row * width + col] = value;
        }
    }
}

void generateIntegralCPUompBothLevels(int width, int height, int const * original, int * result, int threads1, int threads2){
#pragma omp parallel for collapse(2) default(none) shared(original, result) firstprivate(width, height, threads2) \
num_threads(threads1)
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            int value = 0;
#pragma omp parallel for collapse(2) default(none) shared(original) firstprivate(row, col, width) \
num_threads(threads2) reduction(+: value)
            for(int y = row; y >= 0; y--){
                for(int x = col; x >= 0; x--){
                    value += original[y * width + x];
                }
            }
            result[row * width + col] = value;
        }
    }
}