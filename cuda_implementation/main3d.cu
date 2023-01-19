#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "weights_vectors.h"
#include "input_vector.h"

#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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


/*__global__ void fully_connected();



void softmax();*/

int main(){

cudaSetDevice(0);

//LAYER 1

int threads = 13;
int filters = 3;
dim3 block_convo_1(threads,threads,filters); //threads
dim3 grid_convo_1(3,3,2); //blocks


double *dev_input_conv1, *dev_matrix_conv1, *dev_bias1, *dev_output_conv1;

double *output_conv1;

output_conv1 = ( double*)malloc(28 * 28 * 6 * sizeof( double));

// Host to Device

cudaMalloc( (void**)&dev_input_conv1, 32*32*3 * sizeof( double) );
cudaMalloc( (void**)&dev_matrix_conv1, 5*5*3*6 * sizeof( double) );
cudaMalloc( (void**)&dev_output_conv1, 28*28*6 * sizeof( double) );
cudaMalloc( (void**)&dev_bias1, 6 * sizeof( double) );

//Host to Device

cudaMemcpy( dev_input_conv1, input, 32*32*3 * sizeof( double), cudaMemcpyHostToDevice);
cudaMemcpy( dev_matrix_conv1, conv1_weight, 5*5*3*6 * sizeof( double), cudaMemcpyHostToDevice);
cudaMemcpy( dev_bias1, conv1_bias, 6 * sizeof( double), cudaMemcpyHostToDevice);

//kernel
convolution<<<grid_convo_1,block_convo_1,  threads * threads * filters * sizeof( double)>>>(32,5,3,6,threads,dev_input_conv1,dev_matrix_conv1,dev_output_conv1,dev_bias1);


gpuErrchk( cudaPeekAtLastError() );
gpuErrchk( cudaDeviceSynchronize() );

// Device to Host
cudaMemcpy( output_conv1 , dev_output_conv1, 28*28*6 * sizeof( double), cudaMemcpyDeviceToHost);




for (int k = 0; k < 6; k++) {
	for (int j = 0; j < 28; j++) {
		for (int i = 0; i < 28; i++){
			printf("%0.3lf ",output_conv1[i + j * 28 + k * 28 * 28]);
			
		}
		printf("\n");
	}
	printf("\n");
}//*/


// Freeing Space

cudaFree(dev_input_conv1);
cudaFree(dev_matrix_conv1);
cudaFree(dev_output_conv1);
cudaFree(dev_bias1);
free(output_conv1);

cudaDeviceReset();

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