#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include "weights.h"
#include "input.h"
#include "sigmoid.h"


#define imgsize 32
#define filter_size 5

void convolution(int a_width, int b_width, int channel_in, int channel_out,
                 float matrix_a[channel_in][a_width][a_width], 
                 float matrix_b[channel_out][channel_in][b_width][b_width], 
                 float matrix_c[channel_out][a_width - b_width + 1][a_width - b_width + 1],
                 float bias[channel_out]);

void avgpool(int a_width, int amount,int channel,
          float matrix_a[channel][a_width][a_width],
          float matrix_b[channel][a_width/amount][a_width/amount]);

void fully_connected(int a_width, int c_width,
                    float  matrix_a[a_width],
                    float  matrix_b[c_width][a_width],
                    float  matrix_c[c_width],
                    float  bias[c_width]);

void sigmoid(int a_width,int channel,
                 float matrix_a[channel][a_width][a_width],
                 float matrix_b[channel][a_width][a_width]);

void softmax(size_t input_len,  float input[input_len]);


int main(){

    struct timeval tv;
    unsigned long start;
    unsigned long stop;


    // ----- LAYER 1
    int sizexy_conv1 = imgsize-filter_size+1;
     float out_conv_1[6][sizexy_conv1][sizexy_conv1];

    gettimeofday(&tv,NULL);
    start = 1000000 * tv.tv_sec + tv.tv_usec;

    convolution(imgsize,filter_size,3,6,input,conv1_weight,out_conv_1,conv1_bias);

    gettimeofday(&tv,NULL);
    stop = 1000000 * tv.tv_sec + tv.tv_usec;

    printf("Convolution 1\t\ttook %lu microseconds\n", stop-start);

    gettimeofday(&tv,NULL);
    start = 1000000 * tv.tv_sec + tv.tv_usec;

    sigmoid(sizexy_conv1,6,out_conv_1,out_conv_1);

    gettimeofday(&tv,NULL);
    stop = 1000000 * tv.tv_sec + tv.tv_usec;

    printf("Activation 1\t\ttook %lu microseconds\n", stop-start);

    
    // ----- LAYER 2
    int sizexy_pool1 = sizexy_conv1 / 2;
     float out_pool_1[6][sizexy_pool1][sizexy_pool1];

    gettimeofday(&tv,NULL);
    start = 1000000 * tv.tv_sec + tv.tv_usec;

    avgpool(sizexy_conv1,2,6,out_conv_1,out_pool_1);

    gettimeofday(&tv,NULL);
    stop = 1000000 * tv.tv_sec + tv.tv_usec;

    printf("Pooling 1\t\ttook %lu microseconds\n", stop-start);


    // ----- LAYER 3
    int sizexy_conv2 = sizexy_pool1-filter_size+1;
     float out_conv_2[16][sizexy_conv2][sizexy_conv2];

    gettimeofday(&tv,NULL);
    start = 1000000 * tv.tv_sec + tv.tv_usec;

    convolution(sizexy_pool1,filter_size,6,16,out_pool_1,conv2_weight,out_conv_2,conv2_bias);
    
    gettimeofday(&tv,NULL);
    stop = 1000000 * tv.tv_sec + tv.tv_usec;

    printf("Convolution 2\t\ttook %lu microseconds\n", stop-start);

    gettimeofday(&tv,NULL);
    start = 1000000 * tv.tv_sec + tv.tv_usec;

    sigmoid(sizexy_conv2,16,out_conv_2,out_conv_2);
    
    gettimeofday(&tv,NULL);
    stop = 1000000 * tv.tv_sec + tv.tv_usec;

    printf("Sigmoid 2\t\ttook %lu microseconds\n", stop-start);
    

    // ----- LAYER 4
    int sizexy_pool2 = sizexy_conv2 / 2;
     float out_pool_2[16][sizexy_pool2][sizexy_pool2];

    gettimeofday(&tv,NULL);
    start = 1000000 * tv.tv_sec + tv.tv_usec;

    avgpool(sizexy_conv2,2,16,out_conv_2,out_pool_2);
    
    gettimeofday(&tv,NULL);
    stop = 1000000 * tv.tv_sec + tv.tv_usec;

    printf("Pooling 2\t\ttook %lu microseconds\n", stop-start);


    // ----- LAYER 5 
    int sizexy_conv3 = sizexy_pool2-filter_size+1;
     float out_conv_3[120][sizexy_conv3][sizexy_conv3];

    gettimeofday(&tv,NULL);
    start = 1000000 * tv.tv_sec + tv.tv_usec;

    convolution(sizexy_pool2,filter_size,16,120,out_pool_2,conv3_weight,out_conv_3,conv3_bias);
    
    gettimeofday(&tv,NULL);
    stop = 1000000 * tv.tv_sec + tv.tv_usec;

    printf("Convolution 3\t\ttook %lu microseconds\n", stop-start);


    // ----- FLATTEN
    float out_conv_3_flat[120];

    gettimeofday(&tv,NULL);
    start = 1000000 * tv.tv_sec + tv.tv_usec;

    for(int i = 0; i < 120; i++){
        out_conv_3_flat[i] = out_conv_3[i][0][0];
    }
    
    gettimeofday(&tv,NULL);
    stop = 1000000 * tv.tv_sec + tv.tv_usec;

    printf("Flatten \t\ttook %lu microseconds\n", stop-start);


    // ----- LAYER 6
    float out_fc_1[87];

    gettimeofday(&tv,NULL);
    start = 1000000 * tv.tv_sec + tv.tv_usec;

    fully_connected(120, 87,out_conv_3_flat,fc1_weight,out_fc_1,fc1_bias);
    
    gettimeofday(&tv,NULL);
    stop = 1000000 * tv.tv_sec + tv.tv_usec;

    printf("Linear 1\t\ttook %lu microseconds\n", stop-start);

    gettimeofday(&tv,NULL);
    start = 1000000 * tv.tv_sec + tv.tv_usec;

    for(int i = 0; i < 87; i++){
        out_fc_1[i] = sigmoidl(out_fc_1[i]);
    }
    
    gettimeofday(&tv,NULL);
    stop = 1000000 * tv.tv_sec + tv.tv_usec;

    printf("Sigmoid 3\t\ttook %lu microseconds\n", stop-start);

    // ----- LAYER 7
    float out_fc_2[10];

    gettimeofday(&tv,NULL);
    start = 1000000 * tv.tv_sec + tv.tv_usec;

    fully_connected(87, 10,out_fc_1,fc2_weight,out_fc_2,fc2_bias);
    
    gettimeofday(&tv,NULL);
    stop = 1000000 * tv.tv_sec + tv.tv_usec;

    printf("Linear 2\t\ttook %lu microseconds\n", stop-start);

    gettimeofday(&tv,NULL);
    start = 1000000 * tv.tv_sec + tv.tv_usec;

    softmax(10,out_fc_2);
    
    gettimeofday(&tv,NULL);
    stop = 1000000 * tv.tv_sec + tv.tv_usec;

    printf("Softmax \t\ttook %lu microseconds\n", stop-start);

    for(int i = 0; i < 10; i++){
        printf("%lf out %d\n",out_fc_2[i],i);
    }

    



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

void fully_connected(int a_width, int c_width,
                    float  matrix_a[a_width],
                    float  matrix_b[c_width][a_width],
                    float  matrix_c[c_width],
                    float  bias[c_width]){

    for(int i = 0; i < c_width; i++){
        
        matrix_c[i] = bias[i];

        for(int j = 0; j < a_width; j++){
            
            matrix_c[i] += matrix_a[j] * matrix_b[i][j];

        }
    }

    return;
}

 float sigmoidl( float n) {

    return (1 / (1 + powf(EULER_NUMBER_F, -n)));

}

void sigmoid(int a_width,int channel,
                 float matrix_a[channel][a_width][a_width],
                 float matrix_b[channel][a_width][a_width]){
    
    for (int c = 0; c < channel; c ++){
        for (int i = 0; i < a_width; i ++){
            for (int j = 0; j < a_width; j ++){
                matrix_b[c][i][j] = sigmoidl(matrix_a[c][i][j]);
            }
        }

    }

}

void softmax(size_t input_len,  float input[input_len]) {
  assert(input);

   float m = -INFINITY;
  for (size_t i = 0; i < input_len; i++) {
    if (input[i] > m) {
      m = input[i];
    }
  }

   float sum = 0.0;
  for (size_t i = 0; i < input_len; i++) {
    sum += expf(input[i] - m);
  }

   float offset = m + logf(sum);
  for (size_t i = 0; i < input_len; i++) {
    input[i] = expf(input[i] - offset);
  }
}