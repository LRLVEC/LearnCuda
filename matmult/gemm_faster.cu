#include "common.cuh"

__global__ void gemm_faster(float* a, float* b, float* c, size_t a_x, size_t b_x)
{
	__shared__ float ta[TileSize_faster][RollLength_faster], tb[RollLength_faster][TileSize_faster];
	int row = blockIdx.y * blockDim.y * KernelSize_faster;
	int col = blockIdx.x * blockDim.x * KernelSize_faster;
	float ar[KernelSize_faster][KernelLength_faster];
	float br[KernelSize_faster][KernelLength_faster];
	float cr[KernelSize_faster][KernelSize_faster] = { 0 };
	for (int c0(0); c0 < a_x; c0 += RollLength_faster)
	{
		// read: 128*16*2
		// calc: 128*128*16
		// ratio: 128/2 / 4 = 16
		int id = threadIdx.x + threadIdx.y * KernelNum_faster;
		for (int c1(0); c1 < TileSize_faster; c1 += ThreadNum_faster / RollLength_faster)
		{
			int x = id % RollLength_faster;
			int y = c1 + id / RollLength_faster;
			ta[y][x] = a[(row + y) * a_x + c0 + x];
		}
		for (int c1(0); c1 < RollLength_faster; c1 += ThreadNum_faster / TileSize_faster)
		{
			int x = id % TileSize_faster;
			int y = c1 + id / TileSize_faster;
			tb[y][x] = b[(c0 + y) * b_x + col + x];
		}
		__syncthreads();
		for (int c1(0); c1 < RollLength_faster; c1 += KernelLength_faster)
		{
			for (int i(0); i < KernelSize_faster; ++i)
			{
				for (int j(0); j < KernelLength_faster; ++j)
				{
					ar[i][j] = ta[threadIdx.y * KernelSize_faster + i][c1 + j];
					br[i][j] = tb[c1 + j][threadIdx.x * KernelSize_faster + i];
				}
			}
			for (int i(0); i < KernelSize_faster; ++i)
				for (int k(0); k < KernelLength_faster; ++k)
					for (int j(0); j < KernelSize_faster; ++j)
						cr[i][j] += ar[i][k] * br[j][k];
		}
		__syncthreads();
	}
	for (int c0(0); c0 < KernelSize_faster; ++c0)
	{
		for (int c1(0); c1 < KernelSize_faster; ++c1)
		{
			c[b_x * (row + threadIdx.y * KernelSize_faster + c0) + col + threadIdx.x * KernelSize_faster + c1] = cr[c0][c1];
		}
	}
}
