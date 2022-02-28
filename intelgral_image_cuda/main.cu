#include "IntegralGeneratorSeq.cuh"
#include "IntegralGeneratorPar.cuh"
#include "ImageController.cuh"

#include <cstdio>
#include <ctime>
#include <chrono>
#include <cassert>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

__host__ double CPU_sequential(Image const &original, Image &result){
    result = copyImage(original);

    auto t1 = high_resolution_clock::now();
    generateIntegralCPUseq(original.width, original.height, original.pixels, result.pixels);
    auto t2 = high_resolution_clock::now();

    duration<double, std::milli> time = t2 - t1;

    return time.count();
}
__host__ double GPU_globalMem(Image const &original, Image &result, dim3 grid){
    result = copyImage(original);
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

    assert(x*y <= 1024);


    auto t1 = high_resolution_clock::now();
    generateIntegralGPUglobalMem<<<grid, block>>>(dev_original.width, dev_original.height, dev_original.pixels, dev_result.pixels);
    cudaDeviceSynchronize();
    auto t2 = high_resolution_clock::now();

    finish(dev_original, dev_result, result);

    duration<double, std::milli> time = t2 - t1;

    return time.count();
}
__host__ double GPU_sharedMem(Image const &original, Image &result, dim3 grid){
    result = copyImage(original);
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

    assert(x*y <= 1024);

    auto t1 = high_resolution_clock::now();
    generateIntegralGPUsharedMem<<<grid, block, sharedSize>>>(dev_original.width, dev_original.height, dev_original.pixels, dev_result.pixels);
    cudaDeviceSynchronize();
    auto t2 = high_resolution_clock::now();

    finish(dev_original, dev_result, result);
    duration<double, std::milli> time = t2 - t1;

    return time.count();
}

int main() {
    srand(time(NULL));
    Image original = generateImage(100, 200);

    Image seq, par1, par2;
    double s, p1, p2;
    s = CPU_sequential(original, seq);

    dim3 grid(10, 10);
    p1 = GPU_globalMem(original, par1, grid);
    p2 = GPU_sharedMem(original, par2, grid);

    /*
    printImage(original);
    printf("\n");
    printImage(seq);
    printf("\n");
    printImage(par1);
    printf("\n");
    printImage(par2);
    printf("\n");
    */
    if(!areTheSame(seq, par1))
        printf("ERRORE 1\n");
    if(!areTheSame(seq, par2))
        printf("ERRORE 2\n");
    if(!areTheSame(par2, par1))
        printf("ERRORE 3\n");
    printf("Tempo impiegato: %f ms\n", s);
    printf("Tempo impiegato: %f ms\n", p1);
    printf("Tempo impiegato: %f ms\n", p2);

    freeImageHost(original);
    freeImageHost(seq);
    freeImageHost(par1);
    freeImageHost(par2);

    return 0;

}
