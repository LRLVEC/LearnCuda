#include "common.cuh"

__forceinline__ __device__ float load_global(float* ptr, bool pred_guard)
{
	unsigned data;
	asm volatile(
		"{\n"
		"  .reg .pred p;\n"
		"  setp.ne.b32 p, %2, 0;\n"
		"  mov.b32 %0, %3;\n"
		"  @p ld.global.L2::128B.u32 %0, [%1];\n"
		"}\n"
		: "=r"(data)
		: "l"(ptr), "r"((int)pred_guard), "r"(data));
	return *(float*)&data;
}

__forceinline__ __device__ float4 load_shared(float* ptr)
{
	unsigned addr = __cvta_generic_to_shared(ptr);
	uint4 v;
	asm volatile ("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" :
	"=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
	return *(float4*)&v;
}

__global__ void gemm_extreme(float* a, float* b, float* c, size_t a_x, size_t b_x)
{
	__shared__ float ta[TileSizeA_extreme][RollLength_extreme], tb[RollLength_extreme][TileSizeB_extreme];
	// sizeof(ta);
	// sizeof(tb);
	int row = blockIdx.y * blockDim.y * KernelSize_extreme;
	int col = blockIdx.x * blockDim.x * KernelSize_extreme;
	float ar[KernelSize_extreme][KernelLength_extreme];
	float br[KernelSize_extreme][KernelLength_extreme];
	float cr[KernelSize_extreme][KernelSize_extreme] = { 0 };
	int id = threadIdx.x + threadIdx.y * KernelNumX_extreme;
	for (int c0(0); c0 < a_x; c0 += RollLength_extreme)
	{
		// read: (128 + 256) * 16
		// calc: 128 * 256 * 16
		// ratio: 128 * 256 / (4 * (128 + 256)) = 21.3
		int x = id % RollLength_extreme;
		int x8 = (x >> 3) << 3;
		for (int c1(0); c1 < TileSizeA_extreme; c1 += ThreadNum_extreme / RollLength_extreme)
		{
			int y = c1 + id / RollLength_extreme;
			int new_x_0 = x8 + ((x + ((y >> 3) & 1)) & 7);
			// ta[y][new_x_0] = a[(row + y) * a_x + c0 + x];
			ta[y][new_x_0] = load_global(a + (row + y) * a_x + c0 + x, true);
		}
		x = id % TileSizeB_extreme;
		x8 = (x >> 3) << 3;
		int new_x = x8 + ((x + (x >> 5)) & 7);
		for (int c1(0); c1 < RollLength_extreme; c1 += ThreadNum_extreme / TileSizeB_extreme)
		{
			int y = c1 + id / TileSizeB_extreme;
			// make sure that no new bank conflict is introduced here: yes
			// tb[y][new_x] = b[(c0 + y) * b_x + col + x];
			tb[y][new_x] = load_global(b + (c0 + y) * b_x + col + x, true);
		}
		__syncthreads();
#pragma unroll
		for (int c1(0); c1 < RollLength_extreme; c1 += KernelLength_extreme)
		{
			for (int i(0); i < KernelSize_extreme; ++i)
			{
				int new_i = (i + (threadIdx.x >> 2)) & 7;
				for (int j(0); j < KernelLength_extreme; ++j)
				{
					int new_j = (j + (threadIdx.y & 1)) & 7;
					// broadcast and 2 way conflict: threadIdx.y = (2 * n) and (2 * n + 1)
					ar[i][j] = ta[threadIdx.y * KernelSize_extreme + i][c1 + new_j];
					// if use the original sampling method:
					// 4 way conflict:
					// 4 threads from threadIdx.x 0, 4, 8, 12 access the same bank
					// 4 threads from threadIdx.x 1, 5, 9, 13 access the same bank
					// we need to make thread 0, 4 access bank 0, 1
					br[i][j] = tb[c1 + j][threadIdx.x * KernelSize_extreme + new_i];
				}
			}
#pragma unroll
			for (int i(0); i < KernelSize_extreme; ++i)
#pragma unroll
				for (int j(0); j < KernelSize_extreme; ++j)
#pragma unroll
					for (int k(0); k < KernelLength_extreme; ++k)
						cr[i][j] += ar[i][k] * br[j][k];
		}
		__syncthreads();
	}
	for (int c0(0); c0 < KernelSize_extreme; ++c0)
	{
		for (int c1(0); c1 < KernelSize_extreme; ++c1)
		{
			c[b_x * (row + threadIdx.y * KernelSize_extreme + c0) + col + threadIdx.x * KernelSize_extreme + c1] = cr[c0][c1];
		}
	}
}
