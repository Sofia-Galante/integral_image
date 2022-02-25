//
// Created by Sofy on 25/02/2022.
//

#include "IntegralGeneratorPar.cuh"

__global__ void generateIntegralParV1(int const width, int const height, int const * original, int * result){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int value = 0;
    for(int y = row; y >= 0; y--){
        for(int x = col; x >= 0; x--){
            value += original[y * width + x];
        }
    }
    result[row * width + col] = value;
}