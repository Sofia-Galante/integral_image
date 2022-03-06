#include "IntegralGeneratorSeq.cuh"
#include "IntegralGeneratorPar.cuh"
#include "ImageControllerGPU.cuh"

#include <cstdio>
#include <ctime>
#include <chrono>
#include <cassert>
#include <fstream>

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
__host__ double GPU_globalMem(Image const &original, Image &result, dim3 grid, dim3 block){
    result = copyImage(original);
    Image dev_original;
    Image dev_result;
    setUp(original, result, dev_original, dev_result);

    /*
    int x = original.width/grid.x;
    int y = original.height/grid.y;
    if(original.width%grid.x != 0)
        x++;
    if(original.height%grid.y != 0)
        y++;
    dim3 block(x, y);

    assert(x*y <= 1024);
    */

    auto t1 = high_resolution_clock::now();
    generateIntegralGPUglobalMem<<<grid, block>>>(dev_original.width, dev_original.height, dev_original.pixels, dev_result.pixels);
    cudaDeviceSynchronize();
    auto t2 = high_resolution_clock::now();

    finish(dev_original, dev_result, result);

    duration<double, std::milli> time = t2 - t1;

    return time.count();
}
__host__ double GPU_sharedMem(Image const &original, Image &result, dim3 grid, dim3 block){
    result = copyImage(original);
    Image dev_original;
    Image dev_result;
    setUp(original, result, dev_original, dev_result);

    /*
    int x = original.width/grid.x;
    int y = original.height/grid.y;
    if(original.width%grid.x != 0)
        x++;
    if(original.height%grid.y != 0)
        y++;
    dim3 block(x, y);
    assert(x*y <= 1024);
    */

    int sharedSize = block.x*block.y*sizeof(int);




    auto t1 = high_resolution_clock::now();
    generateIntegralGPUsharedMem<<<grid, block, sharedSize>>>(dev_original.width, dev_original.height, dev_original.pixels, dev_result.pixels);
    cudaDeviceSynchronize();
    auto t2 = high_resolution_clock::now();

    finish(dev_original, dev_result, result);
    duration<double, std::milli> time = t2 - t1;

    return time.count();
}

__host__ void dimTest(){
    printf("\n\nTEST 1: CAMBIO DIMENSIONE DELL'IMMAGINE\n");

    Image seq, par1, par2;
    double s, p1, p2;

    std::ofstream csvFile;
    csvFile.open ("../../report/test/test1_cuda.csv");
    for(int i = 10; i <= 500; i+=10){
        printf("\nDimensioni = %d x %d\n", i, i);
        srand(SEED*i);
        Image original = generateImage(i, i);
        s = CPU_sequential(original, seq);
        int gx, gy, bx, by;
        findGridAndBlockDim(i, i, gx, gy, bx, by);
        dim3 grid(gx, gy);
        dim3 block(bx, by);

        printf("%d, %d - %d, %d\n", gx, gy, bx, by);

        p1 = GPU_globalMem(original, par1, grid, block);
        p2 = GPU_sharedMem(original, par2, grid, block);

        assert(areTheSame(seq, par2));
        assert(areTheSame(seq, par1));

        printf("Tempo sequenziale: %f ms\n", s);
        printf("Tempo GPU1: %f ms\n", p1);
        printf("Tempo GPU2: %f ms\n", p2);

        freeImageHost(original);
        freeImageHost(seq);
        freeImageHost(par1);
        freeImageHost(par2);

        csvFile << i*i << ";" << s << ";" << p1 << ";" << p2 << "\n";
    }
    csvFile.close();

}

int main() {
    dimTest();
    return 0;
}
