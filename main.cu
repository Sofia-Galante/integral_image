#include "Image.cuh"
#include <cstdio>
#include <ctime>

int main() {
    srand(time(NULL));
    Image i(10, 3);
    int * image = i.getImage();
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 10; j++)
            printf("%d ", image[i * 10 + j]);
        printf("\n");
    }
    return 0;
}
