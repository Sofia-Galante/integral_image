//
// Created by Sofy on 25/02/2022.
//

#include "Image.cuh"

__host__ Image::Image(int width, int height) {
    this->width = width;
    this->height = height;

    host_image = (int*) malloc(width*height*sizeof(int));
    dev_image = nullptr;

    generateImage();
}

__host__ Image::~Image() {
    delete(host_image);
    if(dev_image != nullptr)
        cudaFree(dev_image);
}

__host__ void Image::generateImage() {
    for(int i = 0; i < width * height; i++)
        host_image[i] = rand() % 100;
}

__host__ void Image::copyHostToDeviceImage() {
    cudaMalloc((void**) &dev_image, width*height*sizeof(int));
    cudaMemcpy(dev_image, host_image, width*height*sizeof(int), cudaMemcpyHostToDevice);
}

__host__ void Image::copyDeviceToHostImage() {
    cudaMemcpy(host_image, dev_image, width*height*sizeof(int), cudaMemcpyDeviceToHost);
}

__host__ int * Image::getImage() {
    return host_image;
}

__host__ int * Image::getDevImage() {
    if (dev_image == nullptr)
        copyHostToDeviceImage();
    return dev_image;
}
