//
// Created by Sofy on 28/02/2022.
//

#include "ImageController.h"



Image copyImage(const Image &image) {
    Image copied;
    copied.width = image.width;
    copied.height = image.height;
    copied.pixels = (int*) malloc(image.width * image.height * sizeof(int));

    for(int i = 0; i < copied.width * copied.height; i++)
        copied.pixels[i] = image.pixels[i];

    return copied;
}

Image generateImage(int width, int heigth) {
    Image image;
    image.width = width;
    image.height = heigth;
    image.pixels = (int *) (int*) malloc(image.width * image.height * sizeof(int));
    for(int i = 0; i < image.width * image.height; i++)
        image.pixels[i] = rand() % 10;
    return image;
}

void freeImage(Image &hostImage) {
    assert(hostImage.pixels != nullptr);
    hostImage.width = 0;
    hostImage.height = 0;
    free(hostImage.pixels);
}


void printImage(Image const &image) {
    for(int i = 0; i < image.height; i++){
        for(int j = 0; j < image.width; j++)
            printf("%d ", image.pixels[i * image.width + j]);
        printf("\n");
    }
    printf("\n");
}