#include "common.cuh"

// block: [32, 4, 2]
__global__ void gemm_ultra(float* a, float* b, float* c, size_t a_x, size_t b_x)
{
	__shared__ float ta[TileSizeA_ultra][RollLength_ultra];
	__shared__ float tb[RollLength_ultra][TileSizeB_ultra];
	int row = blockIdx.y * TileSizeA_ultra;
	int col = blockIdx.x * TileSizeB_ultra;
	int offset_x = WarpSizeX_ultra * threadIdx.y + (threadIdx.x & 3) * KernelSize_ultra;
	int offset_y = WarpSizeY_ultra * threadIdx.z + (threadIdx.x >> 2) * KernelSize_ultra;
	float ar[2][KernelSize_ultra];
	float br[2][KernelSize_ultra];
	float cr[4][KernelSize_ultra][KernelSize_ultra] = { 0 };
	for (int c0(0); c0 < a_x; c0 += RollLength_ultra)
	{
		// read: (128 + 256) * 16
		// calc: 128 * 256 * 16
		// ratio: 128 * 256 / (4 * (128 + 256)) = 21.3
		int id = threadIdx.x + 32 * (threadIdx.y + WarpNumX_ultra * threadIdx.z);
		int x = id % RollLength_ultra;
		for (int c1(0); c1 < TileSizeA_ultra; c1 += ThreadNum_ultra / RollLength_ultra)
		{
			int y = c1 + id / RollLength_ultra;
			ta[y][x] = a[(row + y) * a_x + c0 + x];
		}
		x = id % TileSizeB_ultra;
		// int new_x = ((x >> 3) << 3) + ((x + (x >> 3)) & 7);
		for (int c1(0); c1 < RollLength_ultra; c1 += ThreadNum_ultra / TileSizeB_ultra)
		{
			int y = c1 + id / TileSizeB_ultra;
			tb[y][x] = b[(c0 + y) * b_x + col + x];
		}
		__syncthreads();
		for (int c1(0); c1 < RollLength_ultra; c1++)
		{
			// read from shared
			for (int i(0); i < 2; ++i)
			{
				for (int j(0); j < 4; ++j)
				{
					ar[i][j] = ta[offset_y + i * HalfWarpSizeY_ultra + j][c1];
					br[i][j] = tb[c1][offset_x + i * HalfWarpSizeX_ultra + j];
				}
			}
			for (int c2(0); c2 < 2; ++c2)
				for (int c3(0); c3 < 2; ++c3)
					for (int i(0); i < 4; ++i)
						for (int j(0); j < 4; ++j)
							cr[c2 * 2 + c3][i][j] += ar[c2][i] * br[c3][j];
		}
		__syncthreads();
	}
	for (int i(0); i < 2; ++i)
	{
		for (int j(0); j < 2; ++j)
		{
			for (int c0(0); c0 < KernelSize_ultra; ++c0)
			{
				for (int c1(0); c1 < KernelSize_ultra; ++c1)
				{
					c[b_x * (row + offset_y + i * HalfWarpSizeY_ultra + c0)
						+ col + offset_x + j * HalfWarpSizeX_ultra + c1] = cr[2 * i + j][c0][c1];
				}
			}
		}
	}
}
