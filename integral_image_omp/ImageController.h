//
// Created by Sofy on 28/02/2022.
//

#ifndef INTEGRAL_IMAGEOMP_IMAGECONTROLLER_H
#define INTEGRAL_IMAGEOMP_IMAGECONTROLLER_H

#include "Image.h"
#include <cstdlib>
#include <cstdio>
#include <cassert>


Image generateImage(int width, int height);
Image copyImage(Image const &image);

void freeImage(Image &hostImage);

void printImage(Image const &image);

bool areTheSame(Image const &image1, Image const &image2);

#endif //INTEGRAL_IMAGEOMP_IMAGECONTROLLER_H
