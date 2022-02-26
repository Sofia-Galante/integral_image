//
// Created by Sofy on 25/02/2022.
//

#include "ImageController.cuh"

__host__ Image allocateOnDevice(Image const &hostImage) {
    Image devImage;
    devImage.width = hostImage.width;
    devImage.height = hostImage.height;
    cudaMalloc((void**)&devImage.pixels, devImage.width*devImage.height* sizeof(int));
    copyFromHostToDevice(hostImage, devImage);
    return devImage;
}

__host__ void copyFromHostToDevice(const Image &hostImage, Image &devImage) {
    assert(hostImage.height == devImage.height && hostImage.width == hostImage.width);
    cudaMemcpy(devImage.pixels, hostImage.pixels, devImage.width*devImage.height* sizeof(int), cudaMemcpyHostToDevice);
}

__host__ void copyFromDeviceToHost(Image const &devImage, Image &hostImage) {
    assert(hostImage.height == devImage.height && hostImage.width == hostImage.width);
    cudaMemcpy(hostImage.pixels, devImage.pixels, devImage.width*devImage.height* sizeof(int), cudaMemcpyDeviceToHost);
}

__host__ Image copyImage(const Image &image) {
    Image copied;
    copied.width = image.width;
    copied.height = image.height;
    copied.pixels = (int*) malloc(image.width * image.height * sizeof(int));

    for(int i = 0; i < copied.width * copied.height; i++)
        copied.pixels[i] = image.pixels[i];

    return copied;
}

__host__ Image generateImage(int width, int heigth) {
    Image image;
    image.width = width;
    image.height = heigth;
    image.pixels = (int *) (int*) malloc(image.width * image.height * sizeof(int));
    for(int i = 0; i < image.width * image.height; i++)
        image.pixels[i] = rand() % 10;
    return image;
}

__host__ void freeImageHost(Image &hostImage) {
    assert(hostImage.pixels != nullptr);
    hostImage.width = 0;
    hostImage.height = 0;
    free(hostImage.pixels);
}

__host__ void freeImageDev(Image &devImage) {
    assert(devImage.pixels != nullptr);
    devImage.width = 0;
    devImage.height = 0;
    cudaFree(devImage.pixels);
}

__host__ void printImage(Image const &image) {
    for(int i = 0; i < image.height; i++){
        for(int j = 0; j < image.width; j++)
            printf("%d ", image.pixels[i * image.width + j]);
        printf("\n");
    }
    printf("\n");
}