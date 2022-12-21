#include "common.cuh"
#include <cutlass/gemm/device/gemm.h>


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
