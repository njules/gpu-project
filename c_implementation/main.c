#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "weights.h"
#include "input.h"
#include <math.h>

#define imgsize 32
#define filter_size 5

void convolution(int a_width, int b_width, int channel_in, int channel_out,
                float matrix_a[channel_in][a_width][a_width], 
                float matrix_b[channel_out][channel_in][b_width][b_width], 
                float matrix_c[channel_out][a_width - b_width + 1][a_width - b_width + 1],
                float bias[channel_out]);

void pool(int a_width, int amount,int channel,
         float matrix_a[channel][a_width][a_width],
         float matrix_b[channel][a_width/amount][a_width/amount]);



int main(){

    // LAYER 1
    int sizexy_conv1 = imgsize-filter_size+1;
    float out_conv_1[6][sizexy_conv1][sizexy_conv1];

    convolution(imgsize,filter_size,3,6,input,conv1_weight,out_conv_1,conv1_bias);


    // LAYER 2
    int sizexy_pool1 = sizexy_conv1 / 2;
    float out_pool_1[6][sizexy_pool1][sizexy_pool1];

    pool(sizexy_conv1,2,6,out_conv_1,out_pool_1);

    // LAYER 3
    int sizexy_conv2 = sizexy_pool1-filter_size+1;
    float out_conv_2[16][sizexy_conv2][sizexy_conv2];

    convolution(sizexy_pool1,filter_size,6,16,out_pool_1,conv2_weight,out_conv_2,conv2_bias);

    // LAYER 4
    int sizexy_pool2 = sizexy_conv2 / 2;
    float out_pool_2[16][sizexy_pool2][sizexy_pool2];

    pool(sizexy_conv2,2,16,out_conv_2,out_pool_2);

    // LAYER 5 
    int sizexy_conv3 = sizexy_pool2-filter_size+1;
    float out_conv_3[120][sizexy_conv3][sizexy_conv3];


    convolution(sizexy_pool2,filter_size,16,120,out_pool_2,conv3_weight,out_conv_3,conv3_bias);

    // LAYER 6 



    return 0;
}

void convolution(int a_width, int b_width, int channel_in, int channel_out,
                float matrix_a[channel_in][a_width][a_width], 
                float matrix_b[channel_out][channel_in][b_width][b_width], 
                float matrix_c[channel_out][a_width - b_width + 1][a_width - b_width + 1],
                float bias[channel_out]){

    int kCenter = b_width / 2;
    
    for (int c_out = 0; c_out < channel_out; c_out++)
    {
        for (int i = 0; i < a_width; i++)              // rows
        {
            for (int j = 0; j < a_width; j++)          // columns
            {   
                matrix_c[c_out][i][j] = bias[c_out];

                for (int m = 0; m < b_width; m++)     // kernel rows
                {
                    int mm = b_width - 1 - m;      // row index

                    for (int n = 0; n < b_width; n++) // kernel columns
                    {
                        int nn = b_width - 1 - n;  // column index

                        // index of input signal, used for checking boundary
                        int ii = i + (m - kCenter);
                        int jj = j + (n - kCenter);

                        // ignore input samples which are out of bound
                        if (ii >= 0 && ii < a_width && jj >= 0 && jj < a_width){
                            for (int c_in = 0; c_in < channel_in; c_in++){
                                matrix_c[c_out][i][j] += matrix_a[c_in][ii][jj] * matrix_b[c_out][c_in][m][n];
                            }
                        }
                    }
                }
            }
        }
    }

    return ;
}

void pool(int a_width, int amount,int channel,
                float matrix_a[channel][a_width][a_width],
                float matrix_b[channel][a_width/amount][a_width/amount]){
    
    for(int c = 0; c < channel; c++){
        for (int i = 0; i < a_width - amount; i = i + amount){
            for (int j = 0; j < a_width - amount; j = j + amount){

                for(int x = 0; x < amount; x++){
                    for(int y = 0; y < amount; y++){
                        matrix_b[c][i/amount][j/amount] += matrix_a[c][i+x][j+y];
                    }
                }

            }
        }
    }

    return;
}

void fully_connected(int a_width, int b_width, int c_width,int matrix_a[a_width],int matrix_b[b_width],int matrix_c[c_width]){


    return;
}

void activation(int a_width, int amount,int channel_in, int channel_out,
                int matrix_a[a_width][a_width][channel_in],
                int matrix_b[a_width/amount][a_width/amount][channel_out]){


    return;
}
