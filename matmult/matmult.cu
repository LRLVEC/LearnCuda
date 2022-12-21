#include <_Time.h>
#include <_BLAS.h>
#include <random>
#include "common.cuh"

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

int main()
{
	constexpr unsigned int loop_num(1);
	constexpr bool check_result(false);
	std::mt19937 mt(114514);
	constexpr unsigned int a_row = 8192;
	constexpr unsigned int a_col = 8192;
	constexpr unsigned int b_row = 8192;
	constexpr unsigned int b_col = 8192;
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

	dim3 block_naive = { 32, 32, 1 };
	dim3 grid_naive = { c_col / 32, c_row / 32, 1 };

	dim3 block = { TILE_DIM, TILE_DIM, 1 };
	dim3 grid = { c_col / TILE_DIM, c_row / TILE_DIM, 1 };

	dim3 block_fast = { VecWarp_gemm_fast, VecWarpNum_gemm_fast, 1 };
	dim3 grid_fast = { c_row / VecSize_gemm_fast, c_col / RollWidth_gemm_fast, 1 };

	dim3 block_faster = { KernelNum_faster, KernelNum_faster, 1 };
	dim3 grid_faster = { c_row / TileSize_faster, c_col / TileSize_faster, 1 };

	dim3 block_extreme = { KernelNumX_extreme, KernelNumY_extreme, 1 };
	dim3 grid_extreme = { c_row / TileSizeB_extreme, c_col / TileSizeA_extreme, 1 };

	dim3 block_ultra = { 32, WarpNumX_ultra, WarpNumY_ultra };
	dim3 grid_ultra = { c_row / TileSizeB_ultra, c_col / TileSizeB_ultra, 1 };

	printf("Launch grid: [%d, %d, %d]\n", grid.x, grid.y, grid.z);
	printf("Launch grid fast: [%d, %d, %d]\n", grid_fast.x, grid_fast.y, grid_fast.z);
	printf("Launch grid faster: [%d, %d, %d]\n", grid_faster.x, grid_faster.y, grid_faster.z);
	printf("Launch grid extreme: [%d, %d, %d]\n", grid_extreme.x, grid_extreme.y, grid_extreme.z);
	printf("Launch grid ultra: [%d, %d, %d]\n", grid_ultra.x, grid_ultra.y, grid_ultra.z);

	for (int c0(0); c0 < loop_num; ++c0)
	{
		cudaDeviceSynchronize();
		timer.begin();
		gemm_naive << <grid_naive, block_naive >> > (a_device, b_device, c_device, a_col, b_col);
		cudaDeviceSynchronize();
		timer.end();
		timer.print("cuda mult naive:");
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
		gemm_extreme << <grid_extreme, block_extreme >> > (a_device, b_device, c_device, a_col, b_col);
		cudaDeviceSynchronize();
		timer.end();
		timer.print("cuda mult extreme:");
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

	for (int c0(0); c0 < loop_num; ++c0)
	{
		cudaDeviceSynchronize();
		timer.begin();
		gemm_ultra << <grid_ultra, block_ultra >> > (a_device, b_device, c_device, a_col, b_col);
		cudaDeviceSynchronize();
		timer.end();
		timer.print("cuda mult like but not cutlass:");
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