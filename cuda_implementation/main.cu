#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "weights_vectors.h"
#include "input_vector.h"

#include "test.h"

#include <cuda_runtime.h>

#define EULER_NUMBER 2.71828
#define EULER_NUMBER_F 2.71828182846

#define imgsize 32
#define filter_size 5

__global__ void convolution(int a_width, int b_width, int channel_in, int channel_out,int tile_size,
                 double *matrix_a, //[channel_in][a_width][a_width]
                 double *matrix_b, //[channel_out][channel_in][b_width][b_width]
                 double *matrix_c, //[channel_out][a_width - b_width + 1][a_width - b_width + 1]
                 double *bias); //[channel_out]

__global__ void convolution_nosigmoid(int a_width, int b_width, int channel_in, int channel_out,int tile_size,
                 double *matrix_a, //[channel_in][a_width][a_width]
                 double *matrix_b, //[channel_out][channel_in][b_width][b_width]
                 double *matrix_c, //[channel_out][a_width - b_width + 1][a_width - b_width + 1]
                 double *bias); //[channel_out]

__global__ void avgpool(int a_width, int amount,int channel,int tile_size,
          double *matrix_a, //[channel][a_width][a_width]
          double *matrix_b); //[channel][a_width/amount][a_width/amount]


__global__ void linear_layer(
	int n_infeats,
	int n_outfeats,
	double *input,
	double *weights,
	double *bias,
	double *output
);



/*void softmax();*/

int main(){

cudaSetDevice(0);

//LAYER 1

int threads = 13;
int filter = 6;
dim3 block_convo_1(threads,threads,filter); //threads
dim3 grid_convo_1(3,3,1); //blocks

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
convolution<<<grid_convo_1,block_convo_1, threads * threads * filter * sizeof( double)>>>(32,5,3,6,threads,dev_input_conv1,dev_matrix_conv1,dev_output_conv1,dev_bias1);

// Device to Host
cudaMemcpy( output_conv1 , dev_output_conv1, 28 * 28 * 6 * sizeof( double), cudaMemcpyDeviceToHost);

// Freeing Space

cudaFree(dev_input_conv1);
cudaFree(dev_matrix_conv1);
cudaFree(dev_output_conv1);
cudaFree(dev_bias1);


/*for (int k = 0; k < 6; k++) {
	for (int j = 0; j < 28; j++) {
		for (int i = 0; i < 28; i++){
			printf("%0.3lf ",output_conv1[i + j * 28 + k * 28 * 28]);
			
		}
		printf("\n");
	}
	printf("\n");
}*/

//LAYER 2

// CAREFUL TO AVOID ISSUES KEEP THIS VALUE EVEN THANKS

threads = 12;
filter = 6;
dim3 block_pool_1(threads,threads,filter); //threads
dim3 grid_pool_1(3,3,1); //blocks

double *dev_input_pool1, *dev_output_pool1;

double *output_pool1;

output_pool1 = ( double*)malloc(14 * 14 * 6 * sizeof( double));

// Host to Device

cudaMalloc( (void**)&dev_input_pool1, 28*28*6 * sizeof( double) );
cudaMalloc( (void**)&dev_output_pool1, 14*14*6 * sizeof( double) );

cudaMemcpy( dev_input_pool1, output_conv1, 28*28*6* sizeof( double), cudaMemcpyHostToDevice);

free(output_conv1);

//kernel
avgpool<<<grid_pool_1,block_pool_1, threads * threads * filter * sizeof( double)>>>(28,2,6,threads,dev_input_pool1,dev_output_pool1);

// Device to Host
cudaMemcpy( output_pool1 , dev_output_pool1, 14 * 14 * 6 * sizeof( double), cudaMemcpyDeviceToHost);

// Freeing Space

cudaFree(dev_input_pool1);
cudaFree(dev_output_pool1);

/*for (int k = 0; k < 6; k++) {
	for (int j = 0; j < 14; j++) {
		for (int i = 0; i < 14; i++){
			printf("%lf ",output_pool1[i + j * 14 + k * 14 * 14]);
			
		}
		printf("\n");
	}
	printf("\n");
}*/


//LAYER 3

threads = 8;
filter = 16;
dim3 block_convo_2(threads,threads,filter); //threads
dim3 grid_convo_2(2,2,1); //blocks

double *dev_input_conv2, *dev_matrix_conv2, *dev_bias2, *dev_output_conv2;

double *output_conv2;

output_conv2 = ( double*)malloc(10 * 10 * 16 * sizeof( double));

// Host to Device

cudaMalloc( (void**)&dev_input_conv2, 14*14*6 * sizeof( double) );
cudaMalloc( (void**)&dev_matrix_conv2, 5*5*6*16 * sizeof( double) );
cudaMalloc( (void**)&dev_output_conv2, 10*10*16 * sizeof( double) );
cudaMalloc( (void**)&dev_bias2, 16 * sizeof( double) );

cudaMemcpy( dev_input_conv2, output_pool1, 14*14*6 * sizeof( double), cudaMemcpyHostToDevice);
cudaMemcpy( dev_matrix_conv2, conv2_weight, 5*5*6*16 * sizeof( double), cudaMemcpyHostToDevice);
cudaMemcpy( dev_bias2, conv2_bias, 16 * sizeof( double), cudaMemcpyHostToDevice);

free(output_pool1);

//kernel
convolution<<<grid_convo_2,block_convo_2, threads * threads * filter * sizeof( double)>>>(14,5,6,16,threads,dev_input_conv2,dev_matrix_conv2,dev_output_conv2,dev_bias2);

// Device to Host
cudaMemcpy( output_conv2 , dev_output_conv2, 10 * 10 * 16 * sizeof( double), cudaMemcpyDeviceToHost);

// Freeing Space

cudaFree(dev_input_conv2);
cudaFree(dev_matrix_conv2);
cudaFree(dev_output_conv2);
cudaFree(dev_bias2);

/*for (int k = 0; k < 16; k++) {
	for (int j = 0; j < 10; j++) {
		for (int i = 0; i < 10; i++){
			printf("%lf ",output_conv2[i + j * 10 + k * 10 * 10]);
			
		}
		printf("\n");
	}
	printf("\n");
}*/

//LAYER 4


threads = 8;
filter = 16;
dim3 block_pool_2(threads,threads,filter); //threads
dim3 grid_pool_2(2,2,1); //blocks

double *dev_input_pool2, *dev_output_pool2;

double *output_pool2;

output_pool2 = ( double*)malloc(5 * 5 * 16 * sizeof( double));

// Host to Device

cudaMalloc( (void**)&dev_input_pool2, 10*10*16 * sizeof( double) );
cudaMalloc( (void**)&dev_output_pool2, 5*5*16 * sizeof( double) );

cudaMemcpy( dev_input_pool2, output_conv2, 10*10*16* sizeof( double), cudaMemcpyHostToDevice);

free(output_conv2);

//kernel
avgpool<<<grid_pool_2,block_pool_2, threads * threads * filter * sizeof( double)>>>(10,2,16,threads,dev_input_pool2,dev_output_pool2);

// Device to Host
cudaMemcpy( output_pool2 , dev_output_pool2, 5 * 5 * 16 * sizeof( double), cudaMemcpyDeviceToHost);

// Freeing Space

cudaFree(dev_input_pool2);
cudaFree(dev_output_pool2);

/*for (int k = 0; k < 16; k++) {
	for (int j = 0; j < 5; j++) {
		for (int i = 0; i < 5; i++){
			printf("%lf ",output_pool2[i + j * 5 + k * 5 * 5]);
			
		}
		printf("\n");
	}
	printf("\n");
}*/

//LAYER 5

threads = 2;
filter = 60;
dim3 block_convo_3(threads,threads,filter); //threads
dim3 grid_convo_3(4,4,2); //blocks

double *dev_input_conv3, *dev_matrix_conv3, *dev_bias3, *dev_output_conv3;

double *output_conv3;

output_conv3 = ( double*)malloc(1 * 1 * 120 * sizeof( double));

// Host to Device

cudaMalloc( (void**)&dev_input_conv3, 5*5*16 * sizeof( double) );
cudaMalloc( (void**)&dev_matrix_conv3, 5*5*16*120 * sizeof( double) );
cudaMalloc( (void**)&dev_output_conv3, 1*1*120 * sizeof( double) );
cudaMalloc( (void**)&dev_bias3, 120 * sizeof( double) );

cudaMemcpy( dev_input_conv3, output_pool2, 5*5*16 * sizeof( double), cudaMemcpyHostToDevice);
cudaMemcpy( dev_matrix_conv3, conv3_weight, 5*5*16*120 * sizeof( double), cudaMemcpyHostToDevice);
cudaMemcpy( dev_bias3, conv3_bias, 120 * sizeof( double), cudaMemcpyHostToDevice);

free(output_pool2);

//kernel
convolution_nosigmoid<<<grid_convo_3,block_convo_3, threads * threads * filter * sizeof( double)>>>(5,5,16,120,threads,dev_input_conv3,dev_matrix_conv3,dev_output_conv3,dev_bias3);

// Device to Host
cudaMemcpy( output_conv3 , dev_output_conv3, 1 * 1 * 120 * sizeof( double), cudaMemcpyDeviceToHost);

// Freeing Space

cudaFree(dev_input_conv3);
cudaFree(dev_matrix_conv3);
cudaFree(dev_output_conv3);
cudaFree(dev_bias3);

// for (int i = 0; i < 120; i++){
// 	printf("%lf ",output_conv3[i]);
// }

printf("\n");

//LAYER 6

	// TODO: start Julian
	// define layer sizes
	int NCHANNEL_CONV3 = 120;
	int NFEATS_FC1 = 84;
	//int NFEATS_FC2 = 10;

	// TODO: temporary for testing:
	// output_conv3 = test_inputs; ----------this line was breaking stuff---------- 
	// just use the output_conv3 from before, OR just replace this code in line 325:
	// cudaMemcpy(conv3_out_dev, test_inputs, conv3_out_size, cudaMemcpyHostToDevice);

	for (int i = 0; i < 120; i++){
		printf("%lf ",test_inputs[i]);
	}
	printf("\n");
	printf("\n");

	// define number of threads and blocks
	threads = NFEATS_FC1;
	int blocks = 1;

	// allocate and populate memory for fc1
	double *conv3_out_dev, *fc1_weights_dev, *fc1_bias_dev, *fc1_out_dev;

	int conv3_out_size = NCHANNEL_CONV3 * sizeof(double);
	int fc1_weights_size = NCHANNEL_CONV3 * NFEATS_FC1 * sizeof(double);
	int fc1_bias_size = NFEATS_FC1 * sizeof(double);
	int fc1_out_size = NFEATS_FC1 * sizeof(double);

	cudaMalloc((void**)&conv3_out_dev, conv3_out_size);
	cudaMalloc((void**)&fc1_weights_dev, fc1_weights_size);
	cudaMalloc((void**)&fc1_bias_dev, fc1_bias_size);
	cudaMalloc((void**)&fc1_out_dev, fc1_out_size);

	cudaMemcpy(conv3_out_dev, test_inputs, conv3_out_size, cudaMemcpyHostToDevice);
	cudaMemcpy(fc1_weights_dev, fc1_weight, fc1_weights_size, cudaMemcpyHostToDevice);
	cudaMemcpy(fc1_bias_dev, fc1_bias, fc1_bias_size, cudaMemcpyHostToDevice);

	// layer computations
	linear_layer<<<blocks, threads>>>(NCHANNEL_CONV3, NFEATS_FC1, conv3_out_dev, fc1_weights_dev, fc1_bias_dev, fc1_out_dev);

	// free input and parameter memory on device
	cudaFree(conv3_out_dev);
	cudaFree(fc1_weights_dev);
	cudaFree(fc1_bias_dev);

	// TODO: temporary for testing:
	double *test_outs;
	test_outs = ( double*)malloc(fc1_out_size);
	cudaMemcpy(test_outs , fc1_out_dev, fc1_out_size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < NFEATS_FC1; i++){
		printf("%lf ",test_outs[i]);
	}


free(output_conv3);

cudaDeviceReset();

}

__global__ void avgpool(int a_width, int amount,int channel,int tile_size,
          double *matrix_a, //[channel][a_width][a_width]
          double *matrix_b){ //[channel][a_width/amount][a_width/amount]


	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;

	int out_width = a_width/amount;
	
	__shared__  extern double s[];

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
		
			//for(int c = 0; c < channel; c++){
				double res = 0;
				for(int i = 0; i < amount; i++){
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
                 double *matrix_a, //[channel_in][a_width][a_width]
                 double *matrix_b, //[channel_out][channel_in][b_width][b_width]
                 double *matrix_c, //[channel_out][a_width - b_width + 1][a_width - b_width + 1]
                 double *bias){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;

	int kCenter = b_width/2;
	int out_width = a_width - b_width + 1;
	
	extern __shared__  double s[];

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
			
	
			double res = bias[idz];

			for(int i = 0; i < b_width; i++){
				for(int j = 0; j < b_width; j++){

					int ii = threadIdx.x + i - kCenter;
					int jj = threadIdx.y + j - kCenter;

					if(threadIdx.x > (b_width/2) && threadIdx.x < tile_size-(b_width/2) && threadIdx.y > (b_width/2) && threadIdx.y < tile_size-(b_width/2)){
						for(int c_in = 0; c_in < channel_in; c_in++){
							res += s[ii + jj * tile_size + c_in * tile_size * tile_size] * matrix_b[i + j * b_width + c_in * b_width * b_width + idz * channel_in * b_width * b_width];
						}
					}
					else{

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
                 double *matrix_a, //[channel_in][a_width][a_width]
                 double *matrix_b, //[channel_out][channel_in][b_width][b_width]
                 double *matrix_c, //[channel_out][a_width - b_width + 1][a_width - b_width + 1]
                 double *bias){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int idz = blockDim.z * blockIdx.z + threadIdx.z;

	int kCenter = b_width/2;
	int out_width = a_width - b_width + 1;
	
	extern __shared__  double s[];

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
			
	
			double res = bias[idz];

			for(int i = 0; i < b_width; i++){
				for(int j = 0; j < b_width; j++){

					int ii = threadIdx.x + i - kCenter;
					int jj = threadIdx.y + j - kCenter;

					if(threadIdx.x > (b_width/2) && threadIdx.x < tile_size-(b_width/2) && threadIdx.y > (b_width/2) && threadIdx.y < tile_size-(b_width/2)){
						for(int c_in = 0; c_in < channel_in; c_in++){
							res += s[ii + jj * tile_size + c_in * tile_size * tile_size] * matrix_b[i + j * b_width + c_in * b_width * b_width + idz * channel_in * b_width * b_width];
						}
					}
					else{

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





/*

// IN CASE IT IS MORE CONVENIENT BUT I DOUBT IT DUE TO MEMORY BOTTLENECK

__global__ void sigmoid(int a_width,int channel,
                 double *matrix_a, 
                 double *matrix_b){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	
	__shared__  extern double s[];

	//to shared memory (no ghost cells)

	if(idx >= a_width || idy >= a_width) return;
	
	for(int c_in = 0; c_in < channel; c_in++){
		s[ threadIdx.x + threadIdx.y * tile_size + c_in * tile_size * tile_size ] = matrix_a[ idx + idy * a_width + c_in * a_width * a_width ];
	}

	__syncthreads();

	//start computation

	if( threadIdx.x < tile_size && threadIdx.y < tile_size){
		
		for(int c = 0; c < channel; c++){
			double res = 0;

			int ii = threadIdx.x;
			int jj = threadIdx.y;

			res = sigmoidl(s[ii + jj * tile_size + c * tile_size * tile_size]);
					
			matrix_b[ idx + idy * a_width + c * a_width * a_width ] = res;
		}
	}

}

*/

__global__ void linear_layer(
	int n_infeats,
	int n_outfeats,
	double *input,
	double *weights,
	double *bias,
	double *output
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

	/*for (int i=0; i<n_infeats; i++) {
		res += weights[idx*n_infeats + i] *  input[i];
	}
	output[idx] = res;*/
}
