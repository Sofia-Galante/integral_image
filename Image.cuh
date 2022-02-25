//
// Created by Sofy on 25/02/2022.
//

#ifndef INTEGRAL_IMAGE_IMAGE_CUH
#define INTEGRAL_IMAGE_IMAGE_CUH


#include <cstdlib>

class Image {
public:
    explicit __host__ Image(int width, int heigth);
    __host__ ~Image();

    __host__ int * getImage();
    __host__ int * getDevImage();

private:
    __host__ void generateImage();
    __host__ void copyHostToDeviceImage();
    __host__ void copyDeviceToHostImage();

    int width;
    int height;
    int * host_image;
    int * dev_image;
};


#endif //INTEGRAL_IMAGE_IMAGE_CUH
