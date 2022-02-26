#include "IntegralGeneratorSeq.cuh"
#include "IntegralGeneratorPar.cuh"
#include "ImageController.cuh"

#include <cstdio>
#include <ctime>

__host__ Image CPU_sequential(Image const &original){
    Image result = copyImage(original);
    generateIntegralCPUseq(original.width, original.height, original.pixels, result.pixels);

    return result;
}
__host__ Image GPU_globalMem(Image const &original, dim3 grid){
    Image result = copyImage(original);
    Image dev_original;
    Image dev_result;
    setUp(original, result, dev_original, dev_result);

    int x = original.width/grid.x;
    int y = original.height/grid.y;
    if(original.width%grid.x != 0)
        x++;
    if(original.height%grid.y != 0)
        y++;
    dim3 block(x, y);

    generateIntegralGPUglobalMem<<<grid, block>>>(dev_original.width, dev_original.height, dev_original.pixels, dev_result.pixels);
    cudaDeviceSynchronize();

    finish(dev_original, dev_result, result);
    return result;
}
__host__ Image GPU_sharedMem(Image const &original, dim3 grid){
    Image result = copyImage(original);
    Image dev_original;
    Image dev_result;
    setUp(original, result, dev_original, dev_result);

    int x = original.width/grid.x;
    int y = original.height/grid.y;
    if(original.width%grid.x != 0)
        x++;
    if(original.height%grid.y != 0)
        y++;
    dim3 block(x, y);

    int sharedSize = x*y*sizeof(int);

    generateIntegralGPUsharedMem<<<grid, block, sharedSize>>>(dev_original.width, dev_original.height, dev_original.pixels, dev_result.pixels);
    cudaDeviceSynchronize();

    finish(dev_original, dev_result, result);
    return result;
}

int main() {
    srand(time(NULL));
    Image original = generateImage(20, 40);

    Image seq = CPU_sequential(original);

    dim3 grid(10, 10);
    Image par1 = GPU_globalMem(original, grid);
    Image par2 = GPU_sharedMem(original, grid);

    printImage(original);
    printf("\n");
    printImage(seq);
    printf("\n");
    printImage(par1);
    printf("\n");
    printImage(par2);

    freeImageHost(original);
    freeImageHost(seq);
    freeImageHost(par1);
    freeImageHost(par2);

    return 0;

}
