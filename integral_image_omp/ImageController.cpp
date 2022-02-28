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

bool areTheSame(Image const &image1, Image const &image2){
    if(image1.width != image2.width)
        return false;
    if(image1.height != image2.height)
        return false;
    for(int i = 0; i < image1.width*image1.height; i++)
        if(image1.pixels[i] != image2.pixels[i])
            return false;
    return true;
}