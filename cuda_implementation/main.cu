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
	dim3 block_convo_1(threads,threads,filter); //threads
	dim3 grid_convo_1(3,3,1); //blocks

	float *dev_input_conv1, *dev_matrix_conv1, *dev_bias1, *dev_output_conv1;

	// Host to Device

	cudaMalloc( (void**)&dev_input_conv1, 32*32*3 * sizeof( float) );
	cudaMalloc( (void**)&dev_matrix_conv1, 5*5*3*6 * sizeof( float) );
	cudaMalloc( (void**)&dev_output_conv1, 28*28*6 * sizeof( float) );
	cudaMalloc( (void**)&dev_bias1, 6 * sizeof( float) );

	cudaMemcpy( dev_input_conv1, input, 32*32*3 * sizeof( float), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_matrix_conv1, conv1_weight, 5*5*3*6 * sizeof( float), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_bias1, conv1_bias, 6 * sizeof( float), cudaMemcpyHostToDevice);

	//kernel
	convolution<<<grid_convo_1,block_convo_1, threads * threads * filter * sizeof( float)>>>(32,5,3,6,threads,dev_input_conv1,dev_matrix_conv1,dev_output_conv1,dev_bias1);

	// Freeing Space

	cudaFree(dev_input_conv1);
	cudaFree(dev_matrix_conv1);
	cudaFree(dev_bias1);


	//LAYER 2-----------------------------------------------------------------------------------------------------------

	// CAREFUL TO AVOID ISSUES KEEP THIS VALUE EVEN THANKS

	threads = 12;
	filter = 6;
	dim3 block_pool_1(threads,threads,filter); //threads
	dim3 grid_pool_1(3,3,1); //blocks

	float *dev_input_pool1, *dev_output_pool1;

	// Host to Device

	cudaMalloc( (void**)&dev_input_pool1, 28*28*6 * sizeof( float) );
	cudaMalloc( (void**)&dev_output_pool1, 14*14*6 * sizeof( float) );

	//kernel
	avgpool<<<grid_pool_1,block_pool_1, threads * threads * filter * sizeof( float)>>>(28,2,6,threads,dev_output_conv1,dev_output_pool1);

	// Freeing Space

	cudaFree(dev_input_pool1);
	cudaFree(dev_output_conv1);


	//LAYER 3-----------------------------------------------------------------------------------------------------------

	threads = 8;
	filter = 16;
	dim3 block_convo_2(threads,threads,filter); //threads
	dim3 grid_convo_2(2,2,1); //blocks

	float *dev_input_conv2, *dev_matrix_conv2, *dev_bias2, *dev_output_conv2;


	// Host to Device

	cudaMalloc( (void**)&dev_input_conv2, 14*14*6 * sizeof( float) );
	cudaMalloc( (void**)&dev_matrix_conv2, 5*5*6*16 * sizeof( float) );
	cudaMalloc( (void**)&dev_output_conv2, 10*10*16 * sizeof( float) );
	cudaMalloc( (void**)&dev_bias2, 16 * sizeof( float) );

	cudaMemcpy( dev_matrix_conv2, conv2_weight, 5*5*6*16 * sizeof( float), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_bias2, conv2_bias, 16 * sizeof( float), cudaMemcpyHostToDevice);

	//kernel
	convolution<<<grid_convo_2,block_convo_2, threads * threads * filter * sizeof( float)>>>(14,5,6,16,threads,dev_output_pool1,dev_matrix_conv2,dev_output_conv2,dev_bias2);

	// Freeing Space

	cudaFree(dev_input_conv2);
	cudaFree(dev_matrix_conv2);
	cudaFree(dev_output_pool1);
	cudaFree(dev_bias2);


	//LAYER 4-----------------------------------------------------------------------------------------------------------

	threads = 8;
	filter = 16;
	dim3 block_pool_2(threads,threads,filter); //threads
	dim3 grid_pool_2(2,2,1); //blocks

	float *dev_input_pool2, *dev_output_pool2;

	// Host to Device

	cudaMalloc( (void**)&dev_input_pool2, 10*10*16 * sizeof( float) );
	cudaMalloc( (void**)&dev_output_pool2, 5*5*16 * sizeof( float) );

	//kernel
	avgpool<<<grid_pool_2,block_pool_2, threads * threads * filter * sizeof( float)>>>(10,2,16,threads,dev_output_conv2,dev_output_pool2);

	// Freeing Space

	cudaFree(dev_input_pool2);
	cudaFree(dev_output_conv2);

	//LAYER 5-----------------------------------------------------------------------------------------------------------

	threads = 2;
	filter = 60;
	dim3 block_convo_3(threads,threads,filter); //threads
	dim3 grid_convo_3(4,4,2); //blocks

	float *dev_input_conv3, *dev_matrix_conv3, *dev_bias3, *dev_output_conv3;


	// Host to Device

	cudaMalloc( (void**)&dev_input_conv3, 5*5*16 * sizeof( float) );
	cudaMalloc( (void**)&dev_matrix_conv3, 5*5*16*120 * sizeof( float) );
	cudaMalloc( (void**)&dev_output_conv3, 1*1*120 * sizeof( float) );
	cudaMalloc( (void**)&dev_bias3, 120 * sizeof( float) );

	cudaMemcpy( dev_matrix_conv3, conv3_weight, 5*5*16*120 * sizeof( float), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_bias3, conv3_bias, 120 * sizeof( float), cudaMemcpyHostToDevice);

	//kernel
	convolution_nosigmoid<<<grid_convo_3,block_convo_3, threads * threads * filter * sizeof( float)>>>(5,5,16,120,threads,dev_output_pool2,dev_matrix_conv3,dev_output_conv3,dev_bias3);

	// Freeing Space

	cudaFree(dev_input_conv3);
	cudaFree(dev_matrix_conv3);
	cudaFree(dev_output_pool2);
	cudaFree(dev_bias3);

	// -------------------------- FC1 -----------------------------------------

	// define layer sizes
	int NCHANNEL_CONV3 = 120;
	int NFEATS_FC1 = 84;
	int NFEATS_FC2 = 10;

	// define number of threads and blocks
	threads = NFEATS_FC1;
	int blocks = 1;

	// allocate and populate memory for fc1
	float *conv3_out_dev, *fc1_weights_dev, *fc1_bias_dev, *fc1_out_dev;

	cudaMalloc((void**)&conv3_out_dev, NCHANNEL_CONV3 * sizeof(float));
	cudaMalloc((void**)&fc1_weights_dev, NCHANNEL_CONV3 * NFEATS_FC1 * sizeof(float));
	cudaMalloc((void**)&fc1_bias_dev, NFEATS_FC1 * sizeof(float));
	cudaMalloc((void**)&fc1_out_dev, NFEATS_FC1 * sizeof(float));

	cudaMemcpy(fc1_weights_dev, fc1_weight, NCHANNEL_CONV3 * NFEATS_FC1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fc1_bias_dev, fc1_bias, NFEATS_FC1 * sizeof(float), cudaMemcpyHostToDevice);

	// layer computations
	linear_layer<<<blocks, threads>>>(NCHANNEL_CONV3, NFEATS_FC1, dev_output_conv3, fc1_weights_dev, fc1_bias_dev, fc1_out_dev);

	// free input and parameter memory on device
	cudaFree(conv3_out_dev);
	cudaFree(fc1_weights_dev);
	cudaFree(fc1_bias_dev);
	cudaFree(dev_output_conv3);

	// -------------------------- sigmoid -------------------------------------

	// allocate memory for fc1 sigmoid output
	float *fc1sigmoid_out_dev;
	cudaMalloc((void**)&fc1sigmoid_out_dev, NFEATS_FC1 * sizeof(float));

	// sigmoid activation function
	sigmoid_activation<<<blocks, threads>>>(NFEATS_FC1, fc1_out_dev, fc1sigmoid_out_dev);

	// free input memory on device
	cudaFree(fc1_out_dev);

	// -------------------------- FC2 -----------------------------------------

	// define number of threads and blocks
	threads = NFEATS_FC2;
	blocks = 1;

	// allocate and populate memory for fc2
	float *fc2_weights_dev, *fc2_bias_dev, *fc2_out_dev;

	cudaMalloc((void**)&fc2_weights_dev, NFEATS_FC1 * NFEATS_FC2 * sizeof(float));
	cudaMalloc((void**)&fc2_bias_dev, NFEATS_FC2 * sizeof(float));
	cudaMalloc((void**)&fc2_out_dev, NFEATS_FC2 * sizeof(float));

	cudaMemcpy(fc2_weights_dev, fc2_weight, NFEATS_FC1 * NFEATS_FC2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fc2_bias_dev, fc2_bias, NFEATS_FC2 * sizeof(float), cudaMemcpyHostToDevice);

	// layer computations
	linear_layer<<<blocks, threads>>>(NFEATS_FC1, NFEATS_FC2, fc1sigmoid_out_dev, fc2_weights_dev, fc2_bias_dev, fc2_out_dev);

	// free input and parameter memory on device
	cudaFree(fc1sigmoid_out_dev);
	cudaFree(fc2_weights_dev);
	cudaFree(fc2_bias_dev);

	// -------------------------- softmax -------------------------------------

	// move final layer output to host
	float *fc2_out = (float*) malloc(NFEATS_FC2 * sizeof(float));
	cudaMemcpy(fc2_out , fc2_out_dev, NFEATS_FC2 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(fc2_out_dev);

	// perform softmax computation
	softmax(NFEATS_FC2, fc2_out);

	// print final class probabilities:
	printf("Final class probabilities:\n");
	for (int i = 0; i < NFEATS_FC2; i++){
		printf("%lf ", fc2_out[i]);
	}
	printf("\n");
	
	free(fc2_out);

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
