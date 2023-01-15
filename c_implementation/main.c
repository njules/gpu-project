#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "weights.h"
#include "input.h"
#include "sigmoid.h"


#define imgsize 32
#define filter_size 5

void convolution(int a_width, int b_width, int channel_in, int channel_out,
                long double matrix_a[channel_in][a_width][a_width], 
                long double matrix_b[channel_out][channel_in][b_width][b_width], 
                long double matrix_c[channel_out][a_width - b_width + 1][a_width - b_width + 1],
                long double bias[channel_out]);

void avgpool(int a_width, int amount,int channel,
         long double matrix_a[channel][a_width][a_width],
         long double matrix_b[channel][a_width/amount][a_width/amount]);

void fully_connected(int a_width, int c_width,
                    double long matrix_a[a_width],
                    double long matrix_b[c_width][a_width],
                    double long matrix_c[c_width],
                    double long bias[c_width]);

void sigmoid(int a_width,int channel,
                long double matrix_a[channel][a_width][a_width],
                long double matrix_b[channel][a_width][a_width]);

void softmax(size_t input_len, long double input[input_len]);


int main(){

    // LAYER 1
    int sizexy_conv1 = imgsize-filter_size+1;
    long double out_conv_1[6][sizexy_conv1][sizexy_conv1];

    convolution(imgsize,filter_size,3,6,input,conv1_weight,out_conv_1,conv1_bias);

    printf("\n");
    

    printf("\n");

    sigmoid(sizexy_conv1,6,out_conv_1,out_conv_1);

    


    // LAYER 2
    int sizexy_pool1 = sizexy_conv1 / 2;
    long double out_pool_1[6][sizexy_pool1][sizexy_pool1];

    avgpool(sizexy_conv1,2,6,out_conv_1,out_pool_1);


    // LAYER 3
    int sizexy_conv2 = sizexy_pool1-filter_size+1;
    long double out_conv_2[16][sizexy_conv2][sizexy_conv2];

    convolution(sizexy_pool1,filter_size,6,16,out_pool_1,conv2_weight,out_conv_2,conv2_bias);


    sigmoid(sizexy_conv2,16,out_conv_2,out_conv_2);
    


    // LAYER 4
    int sizexy_pool2 = sizexy_conv2 / 2;
    long double out_pool_2[16][sizexy_pool2][sizexy_pool2];

    avgpool(sizexy_conv2,2,16,out_conv_2,out_pool_2);

    // LAYER 5 
    int sizexy_conv3 = sizexy_pool2-filter_size+1;
    long double out_conv_3[120][sizexy_conv3][sizexy_conv3];


    convolution(sizexy_pool2,filter_size,16,120,out_pool_2,conv3_weight,out_conv_3,conv3_bias);

    // FLATTEN

    long double out_conv_3_flat[120];
    for(int i = 0; i < 120; i++){
        out_conv_3_flat[i] = out_conv_3[i][0][0];
    }


    // LAYER 6

    long double out_fc_1[87];

    fully_connected(120, 87,out_conv_3_flat,fc1_weight,out_fc_1,fc1_bias);

    for(int i = 0; i < 87; i++){
        out_fc_1[i] = sigmoidl(out_fc_1[i]);
    }

    // LAYER 7

    long double out_fc_2[10];

    fully_connected(87, 10,out_fc_1,fc2_weight,out_fc_2,fc2_bias);

    softmax(10,out_fc_2);

    for(int i = 0; i < 10; i++){
        printf("%Lf out %d\n",out_fc_2[i],i);
    }

    



    return 0;
}

void convolution(int a_width, int b_width, int channel_in, int channel_out,
                long double matrix_a[channel_in][a_width][a_width], 
                long double matrix_b[channel_out][channel_in][b_width][b_width], 
                long double matrix_c[channel_out][a_width - b_width + 1][a_width - b_width + 1],
                long double bias[channel_out]){

    int kCenter = b_width / 2;
    
    for (int c_out = 0; c_out < channel_out; c_out++)
    {
        for (int i = 0; i < a_width; i++)              // rows
        {
            for (int j = 0; j < a_width; j++)          // columns
            {   
                
                if(i >= kCenter && i < a_width - kCenter && j >= kCenter && j < a_width - kCenter){
                    
                    matrix_c[c_out][i - kCenter][j - kCenter] = bias[c_out];

                    for (int m = 0; m < b_width; m++)     // kernel rows
                    {

                        for (int n = 0; n < b_width; n++) // kernel columns
                        {
                            
                            // index of input signal, used for checking boundary
                            int ii = i + m - kCenter;
                            int jj = j + n - kCenter;

                            for (int c_in = 0; c_in < channel_in; c_in++){
                                matrix_c[c_out][i- kCenter][j- kCenter] += matrix_a[c_in][ii][jj] * matrix_b[c_out][c_in][m][n];
                            }
                        }
                    }
                }
            }
        }
    }

    return ;
}

void avgpool(int a_width, int amount,int channel,
                long double matrix_a[channel][a_width][a_width],
                long double matrix_b[channel][a_width/amount][a_width/amount]){
    long double sum;
    long double adding;
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

void fully_connected(int a_width, int c_width,
                    double long matrix_a[a_width],
                    double long matrix_b[c_width][a_width],
                    double long matrix_c[c_width],
                    double long bias[c_width]){

    for(int i = 0; i < c_width; i++){
        
        matrix_c[i] = bias[i];

        for(int j = 0; j < a_width; j++){
            
            matrix_c[i] += matrix_a[j] * matrix_b[i][j];

        }
    }

    return;
}

long double sigmoidl(long double n) {

    return (1 / (1 + powf(EULER_NUMBER_L, -n)));

}

void sigmoid(int a_width,int channel,
                long double matrix_a[channel][a_width][a_width],
                long double matrix_b[channel][a_width][a_width]){
    
    for (int c = 0; c < channel; c ++){
        for (int i = 0; i < a_width; i ++){
            for (int j = 0; j < a_width; j ++){
                matrix_b[c][i][j] = sigmoidl(matrix_a[c][i][j]);
            }
        }

    }

}

void softmax(size_t input_len, long double input[input_len]) {
  assert(input);

  long double m = -INFINITY;
  for (size_t i = 0; i < input_len; i++) {
    if (input[i] > m) {
      m = input[i];
    }
  }

  long double sum = 0.0;
  for (size_t i = 0; i < input_len; i++) {
    sum += expf(input[i] - m);
  }

  long double offset = m + logf(sum);
  for (size_t i = 0; i < input_len; i++) {
    input[i] = expf(input[i] - offset);
  }
}