//
// Created by Sofy on 25/02/2022.
//

#include "IntegralGeneratorPar.cuh"
#include <cstdio>

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

__global__ void generateIntegralParV2(int const width, int const height, int const * original, int * result){
    //Utilizzo della shared memory


    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int w = blockDim.x;
    int h = blockDim.y;

    int row = by * h + ty;
    int col = bx * w + tx;

    extern __shared__ int sharedOriginal[];

    int value = 0;

    for(int _by = 0; _by <= by; _by++){
        for(int _bx = 0; _bx <= bx; _bx++){ //itera tra i blocchi
            int _row = _by * h + ty;
            int _col = _bx * w + tx;
            sharedOriginal[ty * w + tx] = original[_row * width + _col];
            __syncthreads();//ogni thread scrive nella shared memory e poi aspetta

            int _y;
            int _x;

            if(_bx < bx){
                _x = w-1;
            }
            else{
                _x = tx;
            }

            if(_by < by){
                _y = h-1;
            }
            else{
                _y = ty;
            }

            for(int y = _y; y >= 0; y--){
                for(int x = _x; x >= 0; x--){
                    value += sharedOriginal[y * w + x];
                }
            }
            __syncthreads();
        }
    }
    result[row * width + col] = value;
}