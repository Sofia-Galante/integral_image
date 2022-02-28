#include <cstdio>
#include <ctime>
#include <chrono>
#include <cassert>
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
int main() {
    srand(time(NULL));
    Image original = generateImage(100, 20);

    Image omp;
    double o;

    o = CPU_omp(original, omp);

    printf("Tempo impiegato: %f ms\n", o);

    freeImage(original);
    freeImage(omp);
    return 0;
}
