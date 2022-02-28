//
// Created by Sofy on 25/02/2022.
//

#ifndef INTEGRAL_IMAGE_IMAGECONTROLLER_CUH
#define INTEGRAL_IMAGE_IMAGECONTROLLER_CUH

#include "Image.cuh"
#include <cstdlib>
#include <cstdio>
#include <cassert>


__host__ Image generateImage(int width, int height);
__host__ Image copyImage(Image const &image);
__host__ Image allocateOnDevice(Image const &hostImage);

__host__ void freeImageHost(Image &hostImage);
__host__ void freeImageDev(Image &devImage);

__host__ void copyFromHostToDevice(Image const &hostImage, Image &devImage);
__host__ void copyFromDeviceToHost(Image const &devImage, Image &hostImage);

__host__ void printImage(Image const &image);

__host__ bool areTheSame(Image const &image1, Image const &image2);


#endif //INTEGRAL_IMAGE_IMAGECONTROLLER_CUH
