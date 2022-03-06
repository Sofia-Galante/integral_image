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

double CPU_omp (Image const &original, Image &result){
    result = copyImage(original);

    auto t1 = high_resolution_clock::now();
    generateIntegralCPUomp(original.width, original.height, original.pixels, result.pixels);
    auto t2 = high_resolution_clock::now();

    duration<double, std::milli> time = t2 - t1;

    return time.count();
}

void dimTest(){
    printf("\n\nTEST 1: CAMBIO DIMENSIONE DELL'IMMAGINE\n");

    Image omp;
    double o;

    std::ofstream csvFile;
    csvFile.open ("../../report/test/test1_omp.csv");
    for(int i = 10; i <= 500; i+=10){
        printf("\nDimensioni = %d x %d\n", i, i);
        srand(SEED*i);
        Image original = generateImage(i, i);
        o = CPU_omp(original, omp);

        printf("Tempo omp: %f ms\n", o);

        freeImage(original);
        freeImage(omp);

        csvFile << i*i << ";" << o << "\n";
    }
    csvFile.close();
}

int main() {
    dimTest();
    return 0;
}
