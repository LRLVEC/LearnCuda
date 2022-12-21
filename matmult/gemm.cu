#include "common.cuh"

// calculates block(blockIdx.x, blockIdx.y) in the result matrix c
// note that a block's size is also TILE_DIM^2
__global__ void gemm(float* a, float* b, float* c, size_t a_x, size_t b_x)
{
	__shared__ float aTile[TILE_DIM][TILE_DIM], bTile[TILE_DIM][TILE_DIM];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	for (int c0(0); c0 < a_x; c0 += TILE_DIM)
	{
		aTile[threadIdx.y][threadIdx.x] = a[row * a_x + c0 + threadIdx.x];
		bTile[threadIdx.y][threadIdx.x] = b[(c0 + threadIdx.y) * b_x + col];
		__syncthreads();
		for (int i = 0; i < TILE_DIM; i++)
		{
			// sum = __fmaf_rn(aTile[threadIdx.y][i], bTile[i][threadIdx.x], sum);

			//  ok!
			// aTile's read is broadcasted
			// bTile may suffer two way bank conflict when tile size is 16
			// sum = __fmaf_ieee_rn(aTile[threadIdx.y][i], bTile[i][threadIdx.x], sum);
			sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
		}
		__syncthreads();
	}
	c[row * b_x + col] = sum;
}
