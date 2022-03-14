//
// Created by Sofy on 25/02/2022.
//

#include "IntegralGeneratorPar.cuh"
#include "cstdio"

__host__ void setUp(Image const &original, Image const &result, Image &dev_original, Image &dev_result){
    dev_original = allocateOnDevice(original);
    dev_result = allocateOnDevice(result);
}

__host__ void finish(Image &dev_original, Image &dev_result, Image &result){
    freeImageDev(dev_original);
    copyFromDeviceToHost(dev_result, result);
    freeImageDev(dev_result);
}

__global__ void generateIntegralGPUglobalMem(int width, int height,
                                             int const * original, int * result){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int value = 0;
    if(col < width && row < height){
        for(int y = row; y >= 0; y--){
            for(int x = col; x >= 0; x--){
                value += original[y * width + x];
            }
        }
        result[row * width + col] = value;
    }
}

__global__ void generateIntegralGPUsharedMem(int width, int height, int const * original, int * result){
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
    int _row, _col, _x, _y;
    for(int _by = 0; _by <= by; _by++){
        for(int _bx = 0; _bx <= bx; _bx++){ //itera tra i blocchi
            _row = _by * h + ty;
            _col = _bx * w + tx;
            if(_col < width && _row < height)
                sharedOriginal[ty * w + tx] = original[_row * width + _col];
            else
                sharedOriginal[ty * w + tx] = 0;
            __syncthreads();//ogni thread scrive nella shared memory e poi aspetta

            if(_bx < bx){ _x = w-1;}
            else{ _x = tx;}

            if(_by < by){_y = h-1;}
            else{_y = ty;}
            if(col < width && row < height){
                for(int y = _y; y >= 0; y--){
                    for(int x = _x; x >= 0; x--){
                        value += sharedOriginal[y * w + x];
                    }
                }
            }
            __syncthreads();
        }
    }
    if(col < width && row < height)
        result[row * width + col] = value;
}