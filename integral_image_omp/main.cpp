#include <cstdio>
#include <ctime>
#include <chrono>
#include <cassert>
#include <fstream>

#include "ImageController.h"
#include "IntegralGeneratorOMP.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

double CPU_omp (Image const &original, Image &result, int * threads){
    result = copyImage(original);

    auto t1 = high_resolution_clock::now();
    generateIntegralCPUomp(original.width, original.height, original.pixels, result.pixels, threads[0], threads[1]);
    auto t2 = high_resolution_clock::now();

    duration<double, std::milli> time = t2 - t1;

    return time.count();
}

void dimTest(){
    printf("\n\nTEST 1: CAMBIO DIMENSIONE DELL'IMMAGINE\n");

    Image omp;
    double o;
    int th[2] = {15, 1};

    std::ofstream csvFile;
    csvFile.open ("../../report/test/test1_omp.csv");
    for(int i = 10; i <= 500; i+=10){
        printf("\nDimensioni = %d x %d\n", i, i);
        srand(SEED*i);
        Image original = generateImage(i, i);
        o = CPU_omp(original, omp, th);

        printf("Tempo omp: %f ms\n", o);

        freeImage(original);
        freeImage(omp);

        csvFile << i*i << ";" << o << "\n";
    }
    csvFile.close();
}

void threadsTest(){
    printf("\n\nTEST 1: CAMBIO NUMERO DEI THREAD\n");

    Image omp1, omp2, omp3, omp4;
    double o1, o2, o3, o4;
    int th1[2] = {15, 1};
    int th2[2] = {1, 15};
    int th3[2] = {10, 10};
    int th4[2] = {15, 2};

    std::ofstream csvFile;
    csvFile.open ("../../report/test/test2_omp.csv");
    for(int i = 100; i <= 500; i+=10){
        printf("\nDimensioni = %d x %d\n", i, i);
        srand(SEED*i);
        Image original = generateImage(i, i);
        o1 = CPU_omp(original, omp1, th1);
        o2 = CPU_omp(original, omp2, th2);
        o3 = CPU_omp(original, omp3,th3);
        o4 = CPU_omp(original, omp4, th4);

        printf("Tempo 1: %f ms\n", o1);
        printf("Tempo 2: %f ms\n", o2);
        printf("Tempo 3: %f ms\n", o3);
        printf("Tempo 4: %f ms\n", o4);

        freeImage(original);
        freeImage(omp1);
        freeImage(omp2);
        freeImage(omp3);
        freeImage(omp4);

        csvFile << i*i << ";" << o1 << ";" << o2 << ";" << o3 << ";" << o4 << "\n";
    }
    csvFile.close();
}

int main() {
    omp_set_nested(4);
    dimTest();
    threadsTest();
    return 0;
}
