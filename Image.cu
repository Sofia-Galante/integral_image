//
// Created by Sofy on 25/02/2022.
//
/*
#include "Image.cuh"

__host__ Image::Image(int width, int height) {
    this->width = width;
    this->height = height;

    hostImage = (int *) malloc(width * height * sizeof(int));

    for(int i = 0; i < width * height; i++)
        hostImage[i] = rand() % 10;

    devImage = nullptr;
}

__host__ Image::Image(const Image &original) {
    this->width = original.width;
    this->height = original.height;
    hostImage = (int *) malloc(width * height * sizeof(int));
    for(int i = 0; i < width*height; i++)
        this->hostImage[i] = original.hostImage[i];

    if(devImage)
        allocateOnDevice();
    else
        devImage = nullptr;
}


__host__ void Image::allocateOnDevice() {
    cudaMalloc((void**) &devImage, sizeof(int)*width*height);
}

__host__ void Image::print() {
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++)
            printf("%d ", hostImage[i * width + j]);
        printf("\n");
    }
}

*/
