#include "common.cuh"

__global__ void gemm_naive(float* a, float* b, float* c, size_t a_x, size_t b_x)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum(0);
	for (int k(0); k < a_x; ++k)
	{
		sum += a[row * a_x + k] * b[k * b_x + col];
	}
	c[row * b_x + col] = sum;
}
