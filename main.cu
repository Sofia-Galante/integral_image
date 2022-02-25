#include "Image.cuh"
#include "IntegralGeneratorSeq.cuh"
#include "IntegralGeneratorPar.cuh"
#include "ImageController.cuh"
#include <cstdio>
#include <ctime>


int main() {
    srand(time(NULL));
    Image original;
    Image original_dev;
    Image result;
    Image resultP1;

    int width = 6;
    int height = 4;


    original = generateImage(width, height);
    result = copyImage(original);
    resultP1 = copyImage(original);

    generateIntegralSeq(original.width, original.height, original.pixels, result.pixels);
    printImage(original);
    printImage(result);

    original_dev = allocateOnDevice(original);
    resultP1 = allocateOnDevice(resultP1);

    dim3 gridDim(2, 2);
    dim3 blockDim(3, 2);

    generateIntegralParV1<<<gridDim, blockDim>>>(original_dev.width, original_dev.height, original_dev.pixels, resultP1.pixels);
    cudaDeviceSynchronize();

    Image P1_host;
    P1_host = copyImage(original);
    fromDeviceToHost(P1_host, resultP1);

    printImage(P1_host);

    Image resultP2;
    resultP2 = copyImage(original);
    resultP2 = allocateOnDevice(resultP2);

    generateIntegralParV2<<<gridDim, blockDim, (width/blockDim.x)*(height/blockDim.y)*sizeof(int)>>>(original_dev.width, original_dev.height, original_dev.pixels, resultP2.pixels);
    cudaDeviceSynchronize();

    Image P2_host;
    P2_host = copyImage(original);
    fromDeviceToHost(P2_host, resultP2);

    printImage(P2_host);

    freeImageHost(original);
    freeImageHost(result);
    freeImageDev(resultP1);
    freeImageDev(original_dev);
    freeImageHost(P1_host);
    freeImageDev(resultP2);
    freeImageHost(P2_host);


    return 0;
}
