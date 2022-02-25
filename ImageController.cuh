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

__host__ void fromHostToDevice(Image const &hostImage, Image &devImage);
__host__ void fromDeviceToHost(Image &hostImage, Image const &devImage);

__host__ void printImage(Image const &image);


#endif //INTEGRAL_IMAGE_IMAGECONTROLLER_CUH