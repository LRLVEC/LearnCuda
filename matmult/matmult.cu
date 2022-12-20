﻿#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <random>
#include <_Time.h>
#include <_BLAS.h>
#include <cutlass/gemm/device/gemm.h>

// All matrices are row-major

void generate_random_matrix(BLAS::mat& ref, std::mt19937 mt, float* data = nullptr, size_t row = 0, size_t col = 0)
{
	std::uniform_real_distribution<float> rd(-1, 1);
	BLAS::randomMat(ref, mt, rd);
	if (data)
	{
		for (size_t c0(0); c0 < row; ++c0)
		{
			for (size_t c1(0); c1 < col; ++c1)
			{
				data[row * c0 + c1] = ref(c0, c1);
			}
		}
	}
}

void write_to_float(float* mat_host, BLAS::mat& mat)
{
	for (size_t c0(0); c0 < mat.height; ++c0)
	{
		for (size_t c1(0); c1 < mat.width; ++c1)
		{
			mat_host[c0 * mat.width + c1] = mat(c0, c1);
		}
	}
}

void check(float* answer, BLAS::mat const& ref, size_t accu)
{
	BLAS::mat a(ref);
	for (size_t c0(0); c0 < a.height; ++c0)
	{
		for (size_t c1(0); c1 < a.width; ++c1)
		{
			a(c0, c1) = answer[a.width * c0 + c1];
		}
	}
	//a.print();
	//ref.print();
	a -= ref;
	//a.print();
	double eps(0);
	for (size_t c0(0); c0 < a.height; ++c0)
	{
		double rowEps(a.row(c0).norm1());
		eps += rowEps;
		// if (rowEps / (a.width * accu) > 1e-6)
		// {
		// 	printf("%d row eps: %e\n", c0, rowEps / (a.width * accu));
		// }
	}
	printf("Error: %e\n", eps / (a.width * a.height * accu));
}

constexpr unsigned int TILE_DIM = 16;

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
			sum = __fmaf_ieee_rn(aTile[threadIdx.y][i], bTile[i][threadIdx.x], sum);
			// sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
		}
		__syncthreads();
	}
	c[row * b_x + col] = sum;
}

constexpr unsigned int VecSize_gemm_fast = 256;
constexpr unsigned int VecWarp_gemm_fast = 64;
constexpr unsigned int VecWarpNum_gemm_fast = VecSize_gemm_fast / VecWarp_gemm_fast;
constexpr unsigned int RollWidth_gemm_fast = 64;
constexpr unsigned int RollTimes_gemm_fast = VecWarp_gemm_fast;

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


constexpr unsigned int TileSize = 128;
constexpr unsigned int RollLength = 16;
constexpr unsigned int KernelSize = 8;
constexpr unsigned int KernelLength = 8;
constexpr unsigned int KernelNum = TileSize / KernelSize;
constexpr unsigned int ThreadNum = KernelNum * KernelNum;

// launch: [32, 32, 1]
__global__ void gemm_faster(float* a, float* b, float* c, size_t a_x, size_t b_x)
{
	__shared__ float ta[TileSize][RollLength], tb[RollLength][TileSize];
	int row = blockIdx.y * blockDim.y * KernelSize;
	int col = blockIdx.x * blockDim.x * KernelSize;
	float ar[KernelSize][KernelLength];
	float br[KernelSize][KernelLength];
	float cr[KernelSize][KernelSize] = { 0 };
	for (int c0(0); c0 < a_x; c0 += RollLength)
	{
		// read: 128*16*2
		// calc: 128*128*16
		// ratio: 128/2 / 4 = 16
		int id = threadIdx.x + threadIdx.y * KernelNum;
		for (int c1(0); c1 < TileSize; c1 += ThreadNum / RollLength)
		{
			int x = id % RollLength;
			int y = c1 + id / RollLength;
			ta[y][x] = a[(row + y) * a_x + c0 + x];
		}
		for (int c1(0); c1 < RollLength; c1 += ThreadNum / TileSize)
		{
			int x = id % TileSize;
			int y = c1 + id / TileSize;
			tb[y][x] = b[(c0 + y) * b_x + col + x];
		}
		__syncthreads();
		for (int c1(0); c1 < RollLength; c1 += KernelLength)
		{
			for (int i(0); i < KernelSize; ++i)
			{
				for (int j(0); j < KernelLength; ++j)
				{
					ar[i][j] = ta[threadIdx.y * KernelSize + i][c1 + j];
					br[i][j] = tb[c1 + j][threadIdx.x * KernelSize + i];
				}
			}
			for (int i(0); i < KernelSize; ++i)
				for (int k(0); k < KernelLength; ++k)
					for (int j(0); j < KernelSize; ++j)
						cr[i][j] += ar[i][k] * br[j][k];
		}
		__syncthreads();
	}
	for (int c0(0); c0 < KernelSize; ++c0)
	{
		for (int c1(0); c1 < KernelSize; ++c1)
		{
			c[b_x * (row + threadIdx.y * KernelSize + c0) + col + threadIdx.x * KernelSize + c1] = cr[c0][c1];
		}
	}
}


cudaError_t gemm_cutlass(float* a, float* b, float* c, size_t a_x, size_t b_x, size_t a_y)
{
	using RowMajor = cutlass::layout::RowMajor;
	using CutlassGemm = cutlass::gemm::device::Gemm<
		float, RowMajor,
		float, RowMajor,
		float, RowMajor>;

	CutlassGemm gemm_operator;
	CutlassGemm::Arguments args({ int(b_x), int(a_y), int(a_x) },
		{ a, a_x },
		{ b, b_x },
		{ c, b_x },
		{ c, b_x },
		{ 1.f, 0.f });
	cutlass::Status status = gemm_operator(args);
	if (status != cutlass::Status::kSuccess)
	{
		return cudaErrorUnknown;
	}
	return cudaSuccess;
}


int main()
{
	constexpr unsigned int loop_num(1);
	constexpr bool check_result(true);
	std::mt19937 mt(114514);
	constexpr unsigned int a_row = 2048;
	constexpr unsigned int a_col = 2048;
	constexpr unsigned int b_row = 2048;
	constexpr unsigned int b_col = 2048;
	constexpr unsigned int c_row = a_row;
	constexpr unsigned int c_col = b_col;
	printf("%dx%d * %dx%d -> %dx%d\n", a_row, a_col, b_row, b_col, c_row, c_col);
	Timer timer;
	timer.begin();
	BLAS::mat a(a_col, a_row);
	BLAS::mat b(b_col, b_row);
	BLAS::mat c(c_col, c_row);
	timer.end();
	timer.print("malloc matrices:");

	timer.begin();
	generate_random_matrix(a, mt);
	timer.end();
	timer.print("gen rand mat a:");

	mt.discard(1llu << 20);

	timer.begin();
	generate_random_matrix(b, mt);
	timer.end();
	timer.print("gen rand mat b:");

	if (check_result)
	{
		timer.begin();
		a(b, c);
		timer.end();
		timer.print("cpu mult");
	}

	float* a_host;
	float* b_host;
	float* c_host;
	float* a_device;
	float* b_device;
	float* c_device;
	constexpr size_t a_size = sizeof(float) * a_row * a_col;
	constexpr size_t b_size = sizeof(float) * b_row * b_col;
	constexpr size_t c_size = sizeof(float) * c_row * c_col;

	timer.begin();
	a_host = (float*)malloc(a_size);
	b_host = (float*)malloc(b_size);
	c_host = (float*)malloc(c_size);
	timer.end();
	timer.print("malloc host:");

	timer.begin();
	cudaMalloc(&a_device, a_size);
	cudaMalloc(&b_device, b_size);
	cudaMalloc(&c_device, c_size);
	timer.end();
	timer.print("malloc device:");

	write_to_float(a_host, a);
	write_to_float(b_host, b);

	timer.begin();
	cudaMemcpy(a_device, a_host, a_size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_device, b_host, b_size, cudaMemcpyHostToDevice);
	cudaMemset(c_device, 0, c_size);
	timer.end();
	timer.print("copy to device:");

	dim3 block = { TILE_DIM, TILE_DIM, 1 };
	dim3 grid = { c_col / TILE_DIM, c_row / TILE_DIM, 1 };
	dim3 block_fast = { VecWarp_gemm_fast, VecWarpNum_gemm_fast, 1 };
	dim3 grid_fast = { c_row / VecSize_gemm_fast, c_col / RollWidth_gemm_fast, 1 };
	dim3 block_faster = { KernelNum, KernelNum, 1 };
	dim3 grid_faster = { c_row / TileSize, c_col / TileSize, 1 };
	printf("Launch grid: [%d, %d, %d]\n", grid.x, grid.y, grid.z);
	printf("Launch grid fast: [%d, %d, %d]\n", grid_fast.x, grid_fast.y, grid_fast.z);
	printf("Launch grid faster: [%d, %d, %d]\n", grid_faster.x, grid_faster.y, grid_faster.z);

	for (int c0(0); c0 < loop_num; ++c0)
	{
		cudaDeviceSynchronize();
		timer.begin();
		gemm << <grid, block >> > (a_device, b_device, c_device, a_col, b_col);
		cudaDeviceSynchronize();
		timer.end();
		timer.print("cuda mult:");
		printf("flops: %.3f T\n", double(a_col) * c_row * c_col / (timer.deltaT() * 1e12));
	}
	if (check_result)
	{
		cudaMemcpy(c_host, c_device, c_size, cudaMemcpyDeviceToHost);
		cudaMemset(c_device, 0, c_size);
		check(c_host, c, a_col);
	}

	for (int c0(0); c0 < loop_num; ++c0)
	{
		cudaDeviceSynchronize();
		timer.begin();
		gemm_fast << <grid_fast, block_fast >> > (a_device, b_device, c_device, a_col, b_col);
		cudaDeviceSynchronize();
		timer.end();
		timer.print("cuda mult fast:");
		printf("flops: %.3f T\n", double(a_col) * c_row * c_col / (timer.deltaT() * 1e12));
	}
	if (check_result)
	{
		cudaMemcpy(c_host, c_device, c_size, cudaMemcpyDeviceToHost);
		cudaMemset(c_device, 0, c_size);
		check(c_host, c, a_col);
	}

	for (int c0(0); c0 < loop_num; ++c0)
	{
		cudaDeviceSynchronize();
		timer.begin();
		gemm_faster << <grid_faster, block_faster >> > (a_device, b_device, c_device, a_col, b_col);
		cudaDeviceSynchronize();
		timer.end();
		timer.print("cuda mult faster:");
		printf("flops: %.3f T\n", double(a_col) * c_row * c_col / (timer.deltaT() * 1e12));
	}
	if (check_result)
	{
		cudaMemcpy(c_host, c_device, c_size, cudaMemcpyDeviceToHost);
		cudaMemset(c_device, 0, c_size);
		check(c_host, c, a_col);
	}

	for (int c0(0); c0 < loop_num; ++c0)
	{
		cudaDeviceSynchronize();
		timer.begin();
		gemm_cutlass(a_device, b_device, c_device, a_col, b_col, a_row);
		cudaDeviceSynchronize();
		timer.end();
		timer.print("cuda mult cutlass:");
		printf("flops: %.3f T\n", double(a_col) * c_row * c_col / (timer.deltaT() * 1e12));
	}
	if (check_result)
	{
		cudaMemcpy(c_host, c_device, c_size, cudaMemcpyDeviceToHost);
		cudaMemset(c_device, 0, c_size);
		check(c_host, c, a_col);
	}

	free(a_host);
	free(b_host);
	free(c_host);
	cudaFree(a_device);
	cudaFree(b_device);
	cudaFree(c_device);

	// a.printToTableTxt("E:/files/C++/CUDA/LearnCuda/matmult/a.txt");
	// b.printToTableTxt("E:/files/C++/CUDA/LearnCuda/matmult/b.txt");
	// c.printToTableTxt("E:/files/C++/CUDA/LearnCuda/matmult/c.txt");

	return 0;
}