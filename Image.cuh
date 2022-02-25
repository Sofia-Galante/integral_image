//
// Created by Sofy on 25/02/2022.
//

#ifndef INTEGRAL_IMAGE_IMAGE_CUH
#define INTEGRAL_IMAGE_IMAGE_CUH

struct Image {
    int * pixels = nullptr;
    int width = 0;
    int height = 0;
};

/*
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <vector>

#include <cuda_runtime_api.h>

class Image {
public:
    explicit __host__ Image(int width, int heigth);
    __host__ Image(Image const & original);
    __host__ ~Image() {
        free(hostImage);
        if(devImage)
            cudaFree(devImage);
    }

    __host__ int getWidth() const {
        return width;
    }
    __host__ int getHeight() const{
        return height;
    }
    __host__  int getValue(int x, int y){
        assert(x >= 0 && y >= 0 && x < width && y < height);
        return image[y * width + x];
    }
    __host__ void setValue(int x, int y, int value){
        assert(x >= 0 && y >= 0 && x < width && y < height);
        image[y * width + x] = value;
    }
    __host__ void print();

private:
    __host__ void allocateOnDevice();

    int width;
    int height;
    int * hostImage;
    int * devImage;
};

*/
#endif //INTEGRAL_IMAGE_IMAGE_CUH
