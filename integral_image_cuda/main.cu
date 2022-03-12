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
    for(int size = 10; size <= 500; size+=10){
        printf("\nDimensioni = %d x %d\n", size, size);
        srand(SEED*size);
        Image original = generateImage(size, size);
        s = CPU_sequential(original, seq);

        int n = (size-1)/16 + 1;
        dim3 grid(n, n);
        dim3 block(16, 16);

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

        csvFile << size*size << ";" << s << ";" << p1 << ";" << p2 << "\n";
    }
    csvFile.close();

}

__host__ void _gridTest(std::string fileName, int d, int GB[4][4]){
    Image original = generateImage(d, d);

    Image par1, par2;
    double p1, p2;

    std::ofstream csvFile;
    csvFile.open (fileName);
    for(int i = 0; i < 4; i++) {
        int gx = GB[i][0];
        int gy = GB[i][1];
        int bx = GB[i][2];
        int by = GB[i][3];

        dim3 grid(gx, gy);
        dim3 block(bx, by);

        printf("%d, %d - %d, %d\n", gx, gy, bx, by);

        p1 = GPU_globalMem(original, par1, grid, block);
        p2 = GPU_sharedMem(original, par2, grid, block);

        printf("Tempo GPU1: %f ms\n", p1);
        printf("Tempo GPU2: %f ms\n", p2);

        freeImageHost(par1);
        freeImageHost(par2);

        csvFile << gx << " x " << gy << " - " << bx << " x " << by << ";" << p1 << ";" << p2 << "\n";
    }
    csvFile.close();

    printf("\n");

    freeImageHost(original);
}

__host__ void gridTest1(){
    printf("\n\nTEST 2: CAMBIO DIMENSIONE DELLA GRID - CASI LIMITE\n");
    srand(SEED);

    int GB10[4][4] = {
            {1, 10, 10, 1}, //righe
            {10, 1, 1, 10}, //colonne
            {10, 10, 1, 1}, //massimizzare i blocchi
            {1, 1, 10, 10}, //massimizzare i thread
    };

    int GB100[4][4] = {
            {1, 100, 100, 1}, //righe
            {100, 1, 1, 100}, //colonne
            {100, 100, 1, 1}, //massimizzare i blocchi
            {4, 4, 32, 32}, //massimizzare i thread
    };

    int GB1000[4][4] = {
            {1, 1000, 1000, 1}, //righe
            {1000, 1, 1, 1000}, //colonne
            {255, 255, 4, 4}, //massimizzare i blocchi
            {32, 32, 32, 32}, //massimizzare i thread
    };

    _gridTest("../../report/test/test2_10_cuda.csv", 10, GB10);
    _gridTest("../../report/test/test2_100_cuda.csv", 100, GB100);
    _gridTest("../../report/test/test2_1000_cuda.csv", 1000, GB1000);
}

__host__ void gridTest2() {
    printf("\n\nTEST 2: CAMBIO DIMENSIONE DELLA GRID - QUADRATI CON AREA MULTIPLA DI 32\n");
    srand(SEED);

    Image par1_1, par1_2, par1_3, par2_1, par2_2, par2_3;
    double p1_1, p1_2, p1_3, p2_1, p2_2, p2_3;

    std::ofstream csvFile;
    csvFile.open ("../../report/test/test3_cuda.csv");
    for(int size = 10; size <= 500; size+=10){
        printf("\nDimensioni = %d x %d\n", size, size);
        srand(SEED*size);
        Image original = generateImage(size, size);

        int n = (size-1)/8 + 1;
        dim3 gridA(n, n);
        dim3 blockA(8, 8);

        int m = (size-1)/16 + 1;
        dim3 gridB(m, m);
        dim3 blockB(16, 16);

        int k = (size-1)/32 + 1;
        dim3 gridC(k, k);
        dim3 blockC(32, 32);

        p1_1 = GPU_globalMem(original, par1_1, gridA, blockA);
        p2_1 = GPU_sharedMem(original, par2_1, gridA, blockA);
        p1_2 = GPU_globalMem(original, par1_2, gridB, blockB);
        p2_2 = GPU_sharedMem(original, par2_2, gridB, blockB);
        p1_3 = GPU_globalMem(original, par1_3, gridC, blockC);
        p2_3 = GPU_sharedMem(original, par2_3, gridC, blockC);

        printf("Tempo GPU1 (8x8): %f ms\n", p1_1);
        printf("Tempo GPU2 (8x8): %f ms\n", p2_1);
        printf("Tempo GPU1 (16x16): %f ms\n", p1_2);
        printf("Tempo GPU2 (16x16): %f ms\n", p2_2);
        printf("Tempo GPU1 (32x32): %f ms\n", p1_3);
        printf("Tempo GPU2 (32x32): %f ms\n", p2_3);

        freeImageHost(original);

        freeImageHost(par1_1);
        freeImageHost(par1_2);
        freeImageHost(par1_3);

        freeImageHost(par2_1);
        freeImageHost(par2_2);
        freeImageHost(par2_3);

        csvFile << size*size << ";" << p1_1 << ";" << p1_2 << ";" << p2_1 << ";" << p2_2 << ";" << p1_3 << ";" << p2_3 << "\n";
    }
    csvFile.close();
}

int main() {
    dimTest();
    gridTest1();
    gridTest2();
    return 0;
}
