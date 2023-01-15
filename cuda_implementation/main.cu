#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "weights.h"
#include "input.h"
#include "sigmoid.h"

#include <cuda_runtime.h>


#define imgsize 32
#define filter_size 5


/*__global__ void convolution(int a_width, int b_width, int channel_in, int channel_out,
                long double ***matrix_a, //[channel_in][a_width][a_width]
                long double ****matrix_b, //[channel_out][channel_in][b_width][b_width]
                long double ***matrix_c, //[channel_out][a_width - b_width + 1][a_width - b_width + 1]
                long double *bias); //[channel_out]*/

__global__ void avgpool();

__global__ void fully_connected();

__global__ void sigmoid();

void softmax();

int main(){

long double *input;

input = (long double*)malloc(32 * 32 * 3 * sizeof(long double));


int threads = 5;
dim3 block(threads); //threads
dim3 grid(6); //blocks

cudaSetDevice(0);

cudaMalloc( (void**)&dev_a, N * sizeof(int) );
cudaMalloc( (void**)&dev_b, M * sizeof(int) );

cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy( dev_b, b, M * sizeof(int), cudaMemcpyHostToDevice);

//kernel
convolution<<<grid,block,(threads+M/2) * sizeof(int)>>>(dev_a,dev_b,dev_c,N,M);


cudaMemcpy( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);



free(a);
free(b);

cudaFree(dev_a);
cudaFree(dev_b);

cudaDeviceReset();


}



long double sigmoidl(long double n) {

    return (1 / (1 + powf(EULER_NUMBER_L, -n)));

}

__global__ void convolution(int a_width, int b_width, int channel_in, int channel_out,
                long double ***matrix_a, //[channel_in][a_width][a_width]
                long double ****matrix_b, //[channel_out][channel_in][b_width][b_width]
                long double ***matrix_c, //[channel_out][a_width - b_width + 1][a_width - b_width + 1]
                long double *bias){

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int res = 0;
	
	__shared__ extern int s[];
	if(idx >= n)return;
	
	s[threadIdx.x+m/2] = a[idx];
	
	if(idx < m/2){
		s[threadIdx.x] = 0;
	}
	if(idx > n-m/2-1){
		s[threadIdx.x+m] = 0;
		
	}
	
	__syncthreads();
	
	if((threadIdx.x > blockDim.x - m/2 - 1 && idx < n - m/2) || (threadIdx.x < m/2 && idx > m/2)){ //its border use global
	
		for(int i = 0; i < m; i++){
			res += a[idx+i-m/2] * b[i];
		}
	
	}
	else{//its inner use shared
	
		for(int i = 0; i < m; i++){
			res += s[threadIdx.x+i] * b[i];
			
		}
	
	}
	
	c[idx] = res;

}