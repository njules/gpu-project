#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "weights_vectors.h"
#include "input_vector.h"
#include "sigmoid.h"

#include <cuda_runtime.h>


#define imgsize 32
#define filter_size 5


__global__ void convolution(int a_width, int b_width, int channel_in, int channel_out,
                long double *matrix_a, //[channel_in][a_width][a_width]
                long double *matrix_b, //[channel_out][channel_in][b_width][b_width]
                long double *matrix_c, //[channel_out][a_width - b_width + 1][a_width - b_width + 1]
                long double *bias); //[channel_out]*/

__global__ void avgpool();

__global__ void fully_connected();

__global__ void sigmoid();

void softmax();

int main(){

//KERNEL CONFIGURATION

int threads = 5;
dim3 block(threads); //threads
dim3 grid(6); //blocks

cudaSetDevice(0);

//LAYER 1

long double *dev_input_conv1, *dev_matrix_conv1, *dev_bias1, *dev_output_conv1;

long double *output_conv1;

output_conv1 = (long double*)malloc(5*5*3*6 * sizeof(long double));

// Host to Device

cudaMalloc( (void**)&dev_input_conv1, 32*32*3 * sizeof(long double) );
cudaMalloc( (void**)&dev_matrix_conv1, 5*5*3*6 * sizeof(long double) );
cudaMalloc( (void**)&dev_output_conv1, 28*28*6 * sizeof(long double) );
cudaMalloc( (void**)&dev_bias1, 6 * sizeof(long double) );

cudaMemcpy( dev_input_conv1, input, 32*32*3 * sizeof(long double), cudaMemcpyHostToDevice);
cudaMemcpy( dev_matrix_conv1, conv1_weight, 5*5*3*6 * sizeof(long double), cudaMemcpyHostToDevice);
cudaMemcpy( dev_matrix_conv1, conv1_weight, 6 * sizeof(long double), cudaMemcpyHostToDevice);

//kernel
convolution<<<grid,block>>>(32,5,3,6,dev_input_conv1,dev_matrix_conv1,dev_output_conv1,dev_bias1);

// Device to Host
cudaMemcpy( output_conv1 , dev_output_conv1, 5*5*3*6 * sizeof(long double), cudaMemcpyDeviceToHost);

// Freeing Space

cudaFree(dev_input_conv1);
cudaFree(dev_matrix_conv1);
cudaFree(dev_output_conv1);
cudaFree(dev_bias1);


//LAYER 2








cudaDeviceReset();

}



long double sigmoidl(long double n) {

    return (1 / (1 + powf(EULER_NUMBER_L, -n)));

}

__global__ void convolution(int a_width, int b_width, int channel_in, int channel_out,
                long double *matrix_a, //[channel_in][a_width][a_width]
                long double *matrix_b, //[channel_out][channel_in][b_width][b_width]
                long double *matrix_c, //[channel_out][a_width - b_width + 1][a_width - b_width + 1]
                long double *bias){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	long double res = 0;
	
	__shared__ extern int s[];

	//to shared memory

	if(idx >= a_width)return;
	
	/*s[threadIdx.x+m/2] = a[idx];
	
	if(idx < m/2){
		s[threadIdx.x] = 0;
	}
	if(idx > n-m/2-1){
		s[threadIdx.x+m] = 0;
		
	}*/


	
	__syncthreads();

	//start computation
	
	/*if((threadIdx.x > blockDim.x - m/2 - 1 && idx < n - m/2) || (threadIdx.x < m/2 && idx > m/2)){ //its border use global
	
		for(int i = 0; i < m; i++){
			res += a[idx+i-m/2] * b[i];
		}
	
	}
	else{//its inner use shared
	
		for(int i = 0; i < m; i++){
			res += s[threadIdx.x+i] * b[i];
			
		}
	
	}*/



	//to output
	
	matrix_c[idx] = res;

}