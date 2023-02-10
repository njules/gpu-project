#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "weights_vectors.h"
#include "input_vector.h"

#include <cuda_runtime.h>

#define EULER_NUMBER 2.71828
#define EULER_NUMBER_F 2.71828182846

#define imgsize 32
#define filter_size 5

__global__ void avgpool(int a_width, int amount,int channel,int tile_size,
          float *matrix_a, //[channel][a_width][a_width]
          float *matrix_b); //[channel][a_width/amount][a_width/amount]

__global__ void linear_layer(
	int n_infeats,
	int n_outfeats,
	float *input,
	float *weights,
	float *bias,
	float *output
);


void softmax(size_t input_len,  float *input);


int main(){

	cudaSetDevice(0);

	cudaFree(0);

	//LAYER 2-----------------------------------------------------------------------------------------------------------

	// CAREFUL TO AVOID ISSUES KEEP THIS VALUE EVEN THANKS

	int threads = 12;
	int filter = 6;
	dim3 block_pool_1(threads,threads,filter); //threads
	dim3 grid_pool_1(3,3,1); //blocks

	float output[6][16][16];

	float *dev_input_pool1, *dev_output_pool1;

	// Host to Device

	cudaMalloc( (void**)&dev_input_pool1, 32*32*6 * sizeof( float) );
	cudaMalloc( (void**)&dev_output_pool1, 16*16*6 * sizeof( float) );


	cudaMemcpy( dev_input_pool1, input_vector , 32*32*6 * sizeof( float), cudaMemcpyHostToDevice);

	//kernel
	avgpool<<<grid_pool_1,block_pool_1, threads * threads * filter * sizeof( float)>>>(32,2,6,threads,dev_input_pool1,dev_output_pool1);

	// Freeing Space


	cudaMemcpy( output, dev_output_pool1, 16*16*6 * sizeof( float), cudaMemcpyDeviceToHost);

	cudaFree(dev_input_pool1);
	cudaFree(dev_output_pool1);
for (int c = 0; c < 6; c++){
	for (int i = 0; i < 16; i++){
		for (int j = 0; j < 16; j++){
			printf("%f ",output[c][i][j]);
		}
	}
}
	cudaDeviceReset();
}

__global__ void avgpool(int a_width, int amount,int channel,int tile_size,
          float *matrix_a, //[channel][a_width][a_width]
          float *matrix_b){ //[channel][a_width/amount][a_width/amount]


	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;

	int out_width = a_width/amount;
	
	__shared__  extern float s[];

	//to shared memory (no ghost cells)

	if(!(idx >= a_width || idy >= a_width)){
		if(threadIdx.z < channel){
			s[ threadIdx.x + threadIdx.y * tile_size + threadIdx.z * tile_size * tile_size ] = matrix_a[ idx + idy * a_width + idz * a_width * a_width ];
		}
	}

	__syncthreads();

	if(idx >= a_width || idy >= a_width || idz >= channel) return;

	//start computation

	if( threadIdx.x < tile_size && threadIdx.y < tile_size && threadIdx.z < channel){
		if( threadIdx.x % 2 == 0 && threadIdx.y % 2 == 0 ){
		
				float res = 0;
				#pragma unroll
				for(int i = 0; i < amount; i++){
					#pragma unroll
					for(int j = 0; j < amount; j++){

						int ii = threadIdx.x + i;
						int jj = threadIdx.y + j;

						res += s[ii + jj * tile_size + threadIdx.z * tile_size * tile_size];
						
					}
				}
				matrix_b[ (idx/amount) + (idy/amount) * out_width + idz * out_width * out_width ] = res / (amount * amount);
			}
			
		
		//}
	}


}


__global__ void linear_layer(
	int n_infeats,
	int n_outfeats,
	float *input,
	float *weights,
	float *bias,
	float *output
){
	/*
		Dimensions:
			input		[n_infeats]
			weights		[n_infeats, n_outfeats]
			bias		[n_outfeats]
			output		[n_outfeats]
	*/

	int idx = threadIdx.x;

	if (idx >= n_outfeats) return;

	output[idx] = bias[idx];
	#pragma unroll
	for (int i=0; i<n_infeats; i++) {
		output[idx] += weights[idx*n_infeats + i] *  input[i];
	}
}


__global__ void sigmoid_activation(
	int n_feats,
	float *input,
	float *output
){
	/*
		Dimensions:
			input		[n_feats]
			output		[n_feats]
	*/

	int idx = threadIdx.x;

	if (idx >= n_feats) return;

	output[idx] = 1 / (1 + powf(EULER_NUMBER_F, -input[idx]));
}


void softmax(size_t input_len, float *input) {
	assert(input);

	float m = -INFINITY;
	#pragma unroll
	for (size_t i = 0; i < input_len; i++) {
		if (input[i] > m) {
			m = input[i];
		}
	}

	float sum = 0.0;
	#pragma unroll
	for (size_t i = 0; i < input_len; i++) {
		sum += expf(input[i] - m);
	}

	float offset = m + logf(sum);
	#pragma unroll
	for (size_t i = 0; i < input_len; i++) {
		input[i] = expf(input[i] - offset);
	}
}

/*

// IN CASE IT IS MORE CONVENIENT BUT I DOUBT IT DUE TO MEMORY BOTTLENECK

__global__ void sigmoid(int a_width,int channel,
                 float *matrix_a, 
                 float *matrix_b){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	
	__shared__  extern float s[];

	//to shared memory (no ghost cells)

	if(idx >= a_width || idy >= a_width) return;
	
	for(int c_in = 0; c_in < channel; c_in++){
		s[ threadIdx.x + threadIdx.y * tile_size + c_in * tile_size * tile_size ] = matrix_a[ idx + idy * a_width + c_in * a_width * a_width ];
	}

	__syncthreads();

	//start computation

	if( threadIdx.x < tile_size && threadIdx.y < tile_size){
		
		for(int c = 0; c < channel; c++){
			float res = 0;

			int ii = threadIdx.x;
			int jj = threadIdx.y;

			res = sigmoidl(s[ii + jj * tile_size + c * tile_size * tile_size]);
					
			matrix_b[ idx + idy * a_width + c * a_width * a_width ] = res;
		}
	}

}

*/
