//
// Created by Sofy on 25/02/2022.
//

#include "IntegralGeneratorSeq.cuh"

__host__ void generateIntegralSeq(int const width, int const height, int const * original, int * result){
    for(int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            int value = 0;
            for(int y = row; y >= 0; y--){
                for(int x = col; x >= 0; x--){
                    value += original[y * width + x];
                }
            }
            result[row * width + col] = value;
        }
    }
}

/*
__host__ void IntegralGeneratorSeq::generateIntegral() {
    for(int y = 0; y < height; y++)
        for(int x = 0; x < width; x++)
            resultImage[y * width + x] = calculate(x, y);
}

__host__ int IntegralGeneratorSeq::calculate(int row, int col) {
    int value  = 0;
    printf("CALCOLO %d, %d\n", row, col);
    for(int y = col; y >= 0; y--){
        for(int x = row; x >= 0; x--){
            value += image.getValue(x, y);
        }
    }
    return value;
}
*/