#include "common.cuh"

__global__ void gemm_fast(float* a, float* b, float* c, size_t a_x, size_t b_x)
{
	// threadIdx.x: [0, RollWidth_gemm_fast - 1]
	// id: [0, VecSize_gemm_fast - 1]
	unsigned int id = threadIdx.x + threadIdx.y * VecWarp_gemm_fast;
	a += a_x * (blockIdx.x * VecSize_gemm_fast + id);
	b += b_x * threadIdx.y + blockIdx.y * RollWidth_gemm_fast + threadIdx.x;
	c += b_x * (blockIdx.x * VecSize_gemm_fast + id) + blockIdx.y * RollWidth_gemm_fast;
	__shared__ float bs[RollTimes_gemm_fast][RollWidth_gemm_fast + 1];
	float cs[RollWidth_gemm_fast] = { 0 };
	int cnt(0);
	do
	{
		// read: 8 + 64
		// calc: 4096
		// ratio: 4096 / (4*72) = 14.2
		for (int i(0); i < RollTimes_gemm_fast; i += VecWarpNum_gemm_fast)
		{
			bs[threadIdx.y + i][threadIdx.x] = b[i * b_x];
		}
		b += RollTimes_gemm_fast * b_x;
		cnt += RollTimes_gemm_fast;
		__syncthreads();
		for (int i(0); i < RollTimes_gemm_fast; ++i, ++a)
		{
			float a0 = a[0];
			for (int j(0); j < RollWidth_gemm_fast; ++j)
			{
				// slow! need to read one from shared each time:
				// fma.rn.f32     %f454, %f266, %f269, %f454;
				// ld.shared.f32  %f270, [%r23+16];
				// fma.rn.f32     %f453, %f266, %f270, %f453;
				// ld.shared.f32  %f271, [%r23+20];
				cs[j] += a0 * bs[i][j];
			}
		}
		__syncthreads();
	} while (cnt < a_x);
	for (int i(0); i < RollWidth_gemm_fast; ++i)
	{
		c[i] = cs[i];
	}
}