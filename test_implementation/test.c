#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include "input_vector.h"
#include "input.h"


#define a 32
#define b 3

void avgpool(int a_width, int amount,int channel,
          float matrix_a[channel][a_width][a_width],
          float matrix_b[channel][a_width/amount][a_width/amount]);

int main(){

    struct timeval tv;

    int sizexy_pool1 = 32 / 2;
    float out_pool_1[6][sizexy_pool1][sizexy_pool1];

    gettimeofday(&tv,NULL);
    unsigned long start = 1000000 * tv.tv_sec + tv.tv_usec;

    avgpool(32,2,6,input,out_pool_1);

    gettimeofday(&tv,NULL);
    unsigned long stop = 1000000 * tv.tv_sec + tv.tv_usec;

    printf("Pooling 1\t\ttook %lu microseconds\n", stop-start);


    return 0;
}

void avgpool(int a_width, int amount,int channel,
                 float matrix_a[channel][a_width][a_width],
                 float matrix_b[channel][a_width/amount][a_width/amount]){
     float sum;
     float adding;
    for(int c = 0; c < channel; c++){
        for (int i = 0; i < a_width - amount + 1; i = i + amount){
            for (int j = 0; j < a_width - amount + 1; j = j + amount){
                matrix_b[c][i/amount][j/amount] = 0;
                for(int x = 0; x < amount; x++){
                    for(int y = 0; y < amount; y++){
                        matrix_b[c][i/amount][j/amount] += matrix_a[c][i+x][j+y];
                    }
                }
                matrix_b[c][i/amount][j/amount] = matrix_b[c][i/amount][j/amount] / (amount * amount);
                
                

            }
        }
    }

    return;
}