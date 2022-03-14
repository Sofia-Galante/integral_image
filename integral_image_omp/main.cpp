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

double CPU_omp(Image const &original, Image &result, int * threads, int level){
    result = copyImage(original);
    auto t1 = high_resolution_clock::now();
    auto t2 = t1;

    assert(level == 1 || level == 2 || level == 3);

    if(level == 1){
        t1 = high_resolution_clock::now();
        generateIntegralCPUompFirstLevel(original.width, original.height, original.pixels, result.pixels, threads[0]);
        t2 = high_resolution_clock::now();
    }
    else if(level == 2){
        t1 = high_resolution_clock::now();
        generateIntegralCPUompSecondLevel(original.width, original.height, original.pixels, result.pixels, threads[0]);
        t2 = high_resolution_clock::now();
    }
    else {
        t1 = high_resolution_clock::now();
        generateIntegralCPUompBothLevels(original.width, original.height, original.pixels, result.pixels, threads[0], threads[1]);
        t2 = high_resolution_clock::now();
    }

    duration<double, std::milli> time = t2 - t1;
    return time.count();
}
void dimTest(std::string const &testName, double * time){
    printf("\n\nTEST: CAMBIO DIMENSIONE DELL'IMMAGINE\n");
    int i = 0;
    std::ofstream csvFile;
    std::string filename = "dimTest/omp/" + testName;
    csvFile.open (filename);
    for(int size = 10; size <= 500; size+=10){
        printf("Tempo omp: %f ms\n", time[i]);
        csvFile << size*size << ";" << time[i] << "\n";
        i++;
    }

    csvFile.close();
}
void threadsTest(std::string const &testName){
    printf("\n\nTEST: LEVEL TEST\n");

    Image omp1, omp2, omp3, omp4, omp5;
    double o1, o2, o3, o4, o5;
    int th1[2] = {1, 15};
    int th2[2] = {15, 1};
    int th3[1] = {15};
    int th4[1] = {15};
    int th5[2] = {15, 15};

    std::ofstream csvFile;
    std::string filename = "threadsTest/levelsTest/" + testName;
    csvFile.open (filename);
    for(int size = 10; size <= 100; size+=5){
        printf("\nDimensioni = %d x %d\n", size, size);
        srand(SEED*size);
        Image original = generateImage(size, size);
        o1 = CPU_omp(original, omp1, th1, 3);
        o2 = CPU_omp(original, omp2, th2, 3);
        o3 = CPU_omp(original, omp3, th3, 1);
        o4 = CPU_omp(original, omp4, th4, 2);
        o5 = CPU_omp(original, omp5, th5, 3);

        printf("Tempo 1: %f ms\n", o1);
        printf("Tempo 2: %f ms\n", o2);
        printf("Tempo 3: %f ms\n", o3);
        printf("Tempo 4: %f ms\n", o4);
        printf("Tempo 5: %f ms\n", o5);

        freeImage(original);
        freeImage(omp1);
        freeImage(omp2);
        freeImage(omp3);
        freeImage(omp4);
        freeImage(omp5);

        csvFile << size*size << ";" << o1 << ";" << o2 << ";" << o3 << ";" << o4 << ";" << o5 << "\n";
    }
    csvFile.close();
}
void threadsTestV2(std::string const &testName){
    printf("\n\nTEST: CAMBIO NUMERO DEI THREAD\n");

    Image omp1, omp2, omp3;
    double o1, o2, o3;
    int th1[1] = {10};
    int th2[1] = {100};
    int th3[1] = {1000};

    double bestTimes[50];
    int i = 0;

    std::ofstream csvFile;
    std::string filename = "threadsTest/velocityTest/" + testName;
    csvFile.open (filename);
    for(int size = 10; size <= 500; size+=10){
        printf("\nDimensioni = %d x %d\n", size, size);
        srand(SEED*size);
        Image original = generateImage(size, size);
        o1 = CPU_omp(original, omp1, th1, 1);
        o2 = CPU_omp(original, omp2, th2, 1);
        o3 = CPU_omp(original, omp3, th3, 1);

        printf("Tempo 1: %f ms\n", o1);
        printf("Tempo 2: %f ms\n", o2);
        printf("Tempo 3: %f ms\n", o3);

        freeImage(original);
        freeImage(omp1);
        freeImage(omp2);
        freeImage(omp3);

        csvFile << size*size << ";" << o1 << ";" << o2 << ";" << o3 << "\n";

        bestTimes[i] = o2;
        i++;
    }
    csvFile.close();

    dimTest(testName, bestTimes);
}

int main(int argc, char *argv[]) {
    if(argc != 2){
        printf("Manca il parametro\n");
        return 1;
    }
    omp_set_nested(4);
    std::string testName = argv[1];
    threadsTest(testName);
    threadsTestV2(testName);
    return 0;
}
