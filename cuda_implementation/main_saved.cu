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

__global__ void convolution(int a_width, int b_width, int channel_in, int channel_out,int tile_size,
                 float *matrix_a, //[channel_in][a_width][a_width]
                 float *matrix_b, //[channel_out][channel_in][b_width][b_width]
                 float *matrix_c, //[channel_out][a_width - b_width + 1][a_width - b_width + 1]
                 float *bias); //[channel_out]

__global__ void convolution_nosigmoid(int a_width, int b_width, int channel_in, int channel_out,int tile_size,
                 float *matrix_a, //[channel_in][a_width][a_width]
                 float *matrix_b, //[channel_out][channel_in][b_width][b_width]
                 float *matrix_c, //[channel_out][a_width - b_width + 1][a_width - b_width + 1]
                 float *bias); //[channel_out]

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

__global__ void sigmoid_activation(
	int n_feats,
	float *input,
	float *output
);

void softmax(size_t input_len,  float *input);


int main(){

	cudaSetDevice(0);

	cudaFree(0);

	//LAYER 1-----------------------------------------------------------------------------------------------------------

	int threads = 13;
	int filter = 6;
	dim3 block_conv_1(threads,threads,filter); //threads
	dim3 grid_conv_1(3,3,1); //blocks

	float *input_dev, *weight_dev, *out_dev_1, *out_dev_2, *bias_dev;

	//allocating cuda memory

	cudaMalloc( (void**)&input_dev, 32*32*3 * sizeof( float) );
	cudaMalloc( (void**)&weight_dev, 120*16*5*5 * sizeof( float) );
	cudaMalloc( (void**)&out_dev_1, 28*28*6 * sizeof( float) );
	cudaMalloc( (void**)&out_dev_2, 28*28*6 * sizeof( float) );
	cudaMalloc( (void**)&bias_dev, 120 * sizeof( float) );

	// Host to Device

	cudaMemcpy( input_dev, input, 32*32*3 * sizeof( float), cudaMemcpyHostToDevice);
	cudaMemcpy( weight_dev, conv1_weight, 48000 * sizeof( float), cudaMemcpyHostToDevice);
	cudaMemcpy( bias_dev, conv1_bias, 6 * sizeof( float), cudaMemcpyHostToDevice);

	//kernel
	convolution<<<grid_conv_1,block_conv_1, threads * threads * filter * sizeof( float)>>>(32,5,3,6,threads,input_dev,weight_dev,out_dev_1,bias_dev);

	//LAYER 2-----------------------------------------------------------------------------------------------------------

	// CAREFUL TO AVOID ISSUES KEEP THIS VALUE EVEN THANKS

	threads = 12;
	filter = 6;
	dim3 block_pool_1(threads,threads,filter); //threads
	dim3 grid_pool_1(3,3,1); //blocks

	//kernel
	avgpool<<<grid_pool_1,block_pool_1, threads * threads * filter * sizeof( float)>>>(28,2,6,threads,out_dev_1,out_dev_2);


	//LAYER 3-----------------------------------------------------------------------------------------------------------

	threads = 8;
	filter = 16;
	dim3 block_conv_2(threads,threads,filter); //threads
	dim3 grid_conv_2(2,2,1); //blocks

	// Host to Device

	cudaMemcpy( weight_dev, conv2_weight, 5*5*6*16 * sizeof( float), cudaMemcpyHostToDevice);
	cudaMemcpy( bias_dev, conv2_bias, 16 * sizeof( float), cudaMemcpyHostToDevice);

	//kernel
	convolution<<<grid_conv_2,block_conv_2, threads * threads * filter * sizeof( float)>>>(14,5,6,16,threads,out_dev_2,weight_dev,out_dev_1,bias_dev);


	//LAYER 4-----------------------------------------------------------------------------------------------------------

	threads = 8;
	filter = 16;
	dim3 block_pool_2(threads,threads,filter); //threads
	dim3 grid_pool_2(2,2,1); //blocks

	//kernel
	avgpool<<<grid_pool_2,block_pool_2, threads * threads * filter * sizeof( float)>>>(10,2,16,threads,out_dev_1,out_dev_2);

	//LAYER 5-----------------------------------------------------------------------------------------------------------

	threads = 2;
	filter = 60;
	dim3 block_conv_3(threads,threads,filter); //threads
	dim3 grid_conv_3(4,4,2); //blocks

	// Host to Device

	cudaMemcpy( weight_dev, conv3_weight, 5*5*16*120 * sizeof( float), cudaMemcpyHostToDevice);
	cudaMemcpy( bias_dev, conv3_bias, 120 * sizeof( float), cudaMemcpyHostToDevice);

	//kernel
	convolution_nosigmoid<<<grid_conv_3,block_conv_3, threads * threads * filter * sizeof( float)>>>(5,5,16,120,threads,out_dev_2,weight_dev,out_dev_1,bias_dev);

	// -------------------------- FC1 -----------------------------------------

	// define number of threads and blocks
	threads = 84;
	int blocks = 1;

	cudaMemcpy(weight_dev, fc1_weight, 120 * 84 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bias_dev, fc1_bias, 84 * sizeof(float), cudaMemcpyHostToDevice);

	// layer computations
	linear_layer<<<blocks, threads>>>(120, 84, out_dev_1, weight_dev, bias_dev, out_dev_2);


	// -------------------------- sigmoid -------------------------------------

	// sigmoid activation function
	sigmoid_activation<<<blocks, threads>>>(84, out_dev_2, out_dev_1);

	// -------------------------- FC2 -----------------------------------------

	// define number of threads and blocks
	threads = 10;
	blocks = 1;

	cudaMemcpy(weight_dev, fc2_weight, 84 * 10 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bias_dev, fc2_bias, 10 * sizeof(float), cudaMemcpyHostToDevice);

	// layer computations
	linear_layer<<<blocks, threads>>>(84, 10, out_dev_1, weight_dev, bias_dev, out_dev_2);


	// -------------------------- softmax -------------------------------------

	// move final layer output to host
	float *fc2_out = (float*) malloc(10 * sizeof(float));
	cudaMemcpy(fc2_out , out_dev_2, 10 * sizeof(float), cudaMemcpyDeviceToHost);

	// perform softmax computation
	softmax(10, fc2_out);

	// --------------------------  output  -------------------------------------

	// print final class probabilities:
	printf("Final class probabilities:\n");
	for (int i = 0; i < 10; i++){
		printf("%lf ", fc2_out[i]);
	}
	printf("\n");
	
	free(fc2_out);

	//freeing cuda memory

	cudaFree(input_dev);
	cudaFree(weight_dev);
	cudaFree(out_dev_1);
	cudaFree(out_dev_2);
	cudaFree(bias_dev);

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
			s[ threadIdx.x + threadIdx.y * tile_size + 
			threadIdx.z * tile_size * tile_size ] = matrix_a[ idx + idy * a_width + idz * a_width * a_width ];
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


__global__ void convolution(int a_width, int b_width, int channel_in, int channel_out,int tile_size,
                 float *matrix_a, //[channel_in][a_width][a_width]
                 float *matrix_b, //[channel_out][channel_in][b_width][b_width]
                 float *matrix_c, //[channel_out][a_width - b_width + 1][a_width - b_width + 1]
                 float *bias){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;

	int kCenter = b_width/2;
	int out_width = a_width - b_width + 1;
	
	extern __shared__  float s[];

	//to shared memory (no ghost cells)

	if(!(idx >= a_width || idy >= a_width)){
		if(threadIdx.z < channel_in){
			s[ threadIdx.x + threadIdx.y * tile_size + threadIdx.z * tile_size * tile_size ] = matrix_a[ idx + idy * a_width + threadIdx.z * a_width * a_width ];
		}
	}

	__syncthreads();

	if(idx >= a_width || idy >= a_width || idz >= channel_out) return;

	//start computation

	if( threadIdx.x < tile_size && threadIdx.y < tile_size && threadIdx.z < channel_out){	
		if(idx >= kCenter && idx < a_width - kCenter && idy >= kCenter && idy < a_width - kCenter){
			
	
			float res = bias[idz];

			#pragma unroll
			for(int i = 0; i < b_width; i++){
				#pragma unroll
				for(int j = 0; j < b_width; j++){

					int ii = threadIdx.x + i - kCenter;
					int jj = threadIdx.y + j - kCenter;

					if(threadIdx.x > (b_width/2) && threadIdx.x < tile_size-(b_width/2) && threadIdx.y > (b_width/2) && threadIdx.y < tile_size-(b_width/2)){
						#pragma unroll
						for(int c_in = 0; c_in < channel_in; c_in++){
							res += s[ii + jj * tile_size + c_in * tile_size * tile_size] * matrix_b[i + j * b_width + c_in * b_width * b_width + idz * channel_in * b_width * b_width];
						}
					}
					else{
						#pragma unroll
						for(int c_in = 0; c_in < channel_in; c_in++){
							res += matrix_a[ (idx + i - kCenter) + (idy + j - kCenter) * a_width + c_in * a_width * a_width ] * matrix_b[i + j * b_width + c_in * b_width * b_width + idz * channel_in * b_width * b_width];
						}

					}
				}
			}

			res = 1 / (1 + powf(EULER_NUMBER_F, -res));

			matrix_c[ (idx - kCenter) + (idy - kCenter) * out_width + idz * out_width * out_width ] = res;

		
		}

	}

}

__global__ void convolution_nosigmoid(int a_width, int b_width, int channel_in, int channel_out,int tile_size,
                 float *matrix_a, //[channel_in][a_width][a_width]
                 float *matrix_b, //[channel_out][channel_in][b_width][b_width]
                 float *matrix_c, //[channel_out][a_width - b_width + 1][a_width - b_width + 1]
                 float *bias){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;

	int kCenter = b_width/2;
	int out_width = a_width - b_width + 1;
	
	extern __shared__  float s[];

	//to shared memory (no ghost cells)

	if(!(idx >= a_width || idy >= a_width)){
		if(threadIdx.z < channel_in){
			s[ threadIdx.x + threadIdx.y * tile_size + threadIdx.z * tile_size * tile_size ] = matrix_a[ idx + idy * a_width + threadIdx.z * a_width * a_width ];
		}
	}

	__syncthreads();

	if(idx >= a_width || idy >= a_width || idz >= channel_out) return;

	//start computation

	if( threadIdx.x < tile_size && threadIdx.y < tile_size && threadIdx.z < channel_out){	
		if(idx >= kCenter && idx < a_width - kCenter && idy >= kCenter && idy < a_width - kCenter){
			
	
			float res = bias[idz];
			#pragma unroll
			for(int i = 0; i < b_width; i++){
				#pragma unroll
				for(int j = 0; j < b_width; j++){

					int ii = threadIdx.x + i - kCenter;
					int jj = threadIdx.y + j - kCenter;

					if(threadIdx.x > (b_width/2) && threadIdx.x < tile_size-(b_width/2) && threadIdx.y > (b_width/2) && threadIdx.y < tile_size-(b_width/2)){
						#pragma unroll
						for(int c_in = 0; c_in < channel_in; c_in++){
							res += s[ii + jj * tile_size + c_in * tile_size * tile_size] * matrix_b[i + j * b_width + c_in * b_width * b_width + idz * channel_in * b_width * b_width];
						}
					}
					else{
						#pragma unroll
						for(int c_in = 0; c_in < channel_in; c_in++){
							res += matrix_a[ (idx + i - kCenter) + (idy + j - kCenter) * a_width + c_in * a_width * a_width ] * matrix_b[i + j * b_width + c_in * b_width * b_width + idz * channel_in * b_width * b_width];
						}

					}
				}
			}

			matrix_c[ (idx - kCenter) + (idy - kCenter) * out_width + idz * out_width * out_width ] = res;

		
		}

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
