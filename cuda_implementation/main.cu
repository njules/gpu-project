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
#define tile_size 32

__global__ void convolution(int a_width, int b_width, int channel_in, int channel_out,
                 double *matrix_a, //[channel_in][a_width][a_width]
                 double *matrix_b, //[channel_out][channel_in][b_width][b_width]
                 double *matrix_c, //[channel_out][a_width - b_width + 1][a_width - b_width + 1]
                 double *bias); //[channel_out]

__global__ void avgpool(int a_width, int amount,int channel,
          double *matrix_a, //[channel][a_width][a_width]
          double *matrix_b); //[channel][a_width/amount][a_width/amount]

/*__global__ void fully_connected();

__global__ void sigmoid();

void softmax();*/

int main(){

//KERNEL CONFIGURATION

int threads = 32;
dim3 block(threads,threads); //threads
dim3 grid(1); //blocks

cudaSetDevice(0);

//LAYER 1

 double *dev_input_conv1, *dev_matrix_conv1, *dev_bias1, *dev_output_conv1;

 double *output_conv1;

output_conv1 = ( double*)malloc(28 * 28 * 6 * sizeof( double));

// Host to Device

cudaMalloc( (void**)&dev_input_conv1, 32*32*3 * sizeof( double) );
cudaMalloc( (void**)&dev_matrix_conv1, 5*5*3*6 * sizeof( double) );
cudaMalloc( (void**)&dev_output_conv1, 28*28*6 * sizeof( double) );
cudaMalloc( (void**)&dev_bias1, 6 * sizeof( double) );

cudaMemcpy( dev_input_conv1, input, 32*32*3 * sizeof( double), cudaMemcpyHostToDevice);
cudaMemcpy( dev_matrix_conv1, conv1_weight, 5*5*3*6 * sizeof( double), cudaMemcpyHostToDevice);
cudaMemcpy( dev_bias1, conv1_bias, 6 * sizeof( double), cudaMemcpyHostToDevice);

//kernel
convolution<<<grid,block, 32 * 32 * 3 * sizeof( double)>>>(32,5,3,6,dev_input_conv1,dev_matrix_conv1,dev_output_conv1,dev_bias1);

// Device to Host
cudaMemcpy( output_conv1 , dev_output_conv1, 28 * 28 * 6 * sizeof( double), cudaMemcpyDeviceToHost);

// Freeing Space

cudaFree(dev_input_conv1);
cudaFree(dev_matrix_conv1);
cudaFree(dev_output_conv1);
cudaFree(dev_bias1);


for (int k = 0; k < 6; k++) {
	for (int j = 0; j < 28; j++) {
		for (int i = 0; i < 28; i++){
			printf("%0.2lf ",output_conv1[i + j * 28 + k * 28 * 28]);
			
		}
		printf("\n");
	}
	printf("\n");
}

//LAYER 2

free(output_conv1);



//LAYER 3

cudaDeviceReset();

}



double sigmoidl( double n) {

    return (1 / (1 + powf(EULER_NUMBER_L, -n)));

}

__global__ void convolution(int a_width, int b_width, int channel_in, int channel_out,
                 double *matrix_a, //[channel_in][a_width][a_width]
                 double *matrix_b, //[channel_out][channel_in][b_width][b_width]
                 double *matrix_c, //[channel_out][a_width - b_width + 1][a_width - b_width + 1]
                 double *bias){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	int kCenter = b_width/2;
	int out_width = a_width - b_width + 1;
	
	__shared__  extern double s[];

	//to shared memory (no ghost cells)

	if(idx >= a_width || idy >= a_width) return;
	
	for(int c_in = 0; c_in < channel_in; c_in++){
		s[ threadIdx.x + threadIdx.y * tile_size + c_in * tile_size * tile_size ] = matrix_a[ idx + idy * a_width + c_in * a_width * a_width ];
	}
	

	__syncthreads();

	//start computation

	if( threadIdx.x < tile_size && threadIdx.y < tile_size){	
		if(idx >= kCenter && idx < a_width - kCenter && idy >= kCenter && idy < a_width - kCenter){
			
		
			for(int c_out = 0; c_out < channel_out; c_out++){
				 double res = bias[c_out];

				for(int i = 0; i < b_width; i++){
					for(int j = 0; j < b_width; j++){

						int ii = threadIdx.x + i - kCenter;
						int jj = threadIdx.y + j - kCenter;

						for(int c_in = 0; c_in < channel_in; c_in++){
							res += s[ii + jj * tile_size + c_in * tile_size * tile_size] * matrix_b[i + j * b_width + c_in * b_width * b_width + c_out * channel_in * b_width * b_width];
						}
					}
				}

				matrix_c[ (idx - kCenter) + (idy - kCenter) * out_width + c_out * out_width * out_width ] = res;

			}
		}

	}

}

__global__ void avgpool(int a_width, int amount,int channel,
          double *matrix_a, //[channel][a_width][a_width]
          double *matrix_b){ //[channel][a_width/amount][a_width/amount]


	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	int out_width = a_width/amount;
	
	__shared__  extern double s[];

	//to shared memory (no ghost cells)

	if(idx >= a_width || idy >= a_width) return;
	
	for(int c_in = 0; c_in < channel; c_in++){
		s[ threadIdx.x + threadIdx.y * tile_size + c_in * tile_size * tile_size ] = matrix_a[ idx + idy * a_width + c_in * a_width * a_width ];
	}

	__syncthreads();

	//start computation



}