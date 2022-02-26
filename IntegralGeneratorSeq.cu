//
// Created by Sofy on 25/02/2022.
//

#include "IntegralGeneratorSeq.cuh"

__host__ void generateIntegralCPUseq(int width, int height, int const * original, int * result){
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