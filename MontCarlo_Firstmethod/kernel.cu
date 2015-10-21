#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include <stdio.h>
#include <iostream>
#include "curand_kernel.h"
#include <ctime>

#define NE WA*HA //Total number of random numbers 
#define WA 2   // Matrix A width	
#define HA 2   // Matrix A height
#define SAMPLE 100 //Sample number
#define BLOCK_SIZE 2 //Block size

using namespace std;
size_t N = 1 << 11;
const int threadsPerBlock = 256;
const int blocksPerGrid = (32 < ((N + threadsPerBlock - 1) / threadsPerBlock)) ? 32 : ((N + threadsPerBlock - 1) / threadsPerBlock);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
__global__ void cudaRand(double *d_out)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	curandState state;
	curand_init((unsigned long long)clock() + i, 0, 0, &state);
	d_out[i] = curand_uniform_double(&state);
}
__global__ void blackSchole(double s, double mu, double V, double T, double *randnum, double *output, const int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ double cache[threadsPerBlock];
	double temp = 0;
	int cacheIndex = threadIdx.x;
	while (tid < N)
	{
		temp += s*exp(mu*T + V*sqrt(T)*randnum[tid]);
		tid += blockDim.x * gridDim.x;

	}
	cache[cacheIndex] = temp;
	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;

	}
	if (cacheIndex == 0)
		output[blockIdx.x] = cache[0];


}

int main()
{

	double *h_v = new double[N];
	double S = 1, mu = 0.5, V = 3, T = 10;
	double *d_out;
	double *output;
	cudaMalloc((void**)&d_out, N * sizeof(double));
	cudaMalloc((void**)&output, N * sizeof(double));

	// generate random numbers
	cudaRand << < blocksPerGrid, threadsPerBlock >> > (d_out);
	cudaThreadSynchronize();
	blackSchole << < blocksPerGrid, threadsPerBlock >> >(S, mu, V, T, d_out, output, N);


	cudaMemcpy(h_v, output, N * sizeof(double), cudaMemcpyDeviceToHost);
	double result = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		result += h_v[i];
		printf("out %d: %f\n", i, h_v[i]);
	}
	result /= blocksPerGrid;
	printf("finally: %f\n", result);


	getchar();
	cudaFree(d_out);
	cudaFree(output);
	delete[] h_v;

	return 0;

}

