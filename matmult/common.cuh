#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// gemm
constexpr unsigned int TILE_DIM = 32;

// gemm_fast
constexpr unsigned int VecSize_gemm_fast = 256;
constexpr unsigned int VecWarp_gemm_fast = 64;
constexpr unsigned int VecWarpNum_gemm_fast = VecSize_gemm_fast / VecWarp_gemm_fast;
constexpr unsigned int RollWidth_gemm_fast = 64;
constexpr unsigned int RollTimes_gemm_fast = VecWarp_gemm_fast;

// gemm_faster
constexpr unsigned int TileSize_faster = 128;
constexpr unsigned int RollLength_faster = 16;
constexpr unsigned int KernelSize_faster = 8;
constexpr unsigned int KernelLength_faster = 8;
constexpr unsigned int KernelNum_faster = TileSize_faster / KernelSize_faster;
constexpr unsigned int ThreadNum_faster = KernelNum_faster * KernelNum_faster;

// gemm_extreme
constexpr unsigned int TileSizeA_extreme = 256;// y
constexpr unsigned int TileSizeB_extreme = 128;// x
constexpr unsigned int RollLength_extreme = 16;
constexpr unsigned int KernelSize_extreme = 8;
constexpr unsigned int KernelLength_extreme = 8;
constexpr unsigned int KernelNumX_extreme = TileSizeB_extreme / KernelSize_extreme;
constexpr unsigned int KernelNumY_extreme = TileSizeA_extreme / KernelSize_extreme;
constexpr unsigned int ThreadNum_extreme = KernelNumX_extreme * KernelNumY_extreme;

// gemm_ultra
constexpr unsigned int TileSizeA_ultra = 128;// y
constexpr unsigned int TileSizeB_ultra = 128;// x
constexpr unsigned int RollLength_ultra = 16;
constexpr unsigned int WarpNumX_ultra = 4;
constexpr unsigned int WarpNumY_ultra = 2;
constexpr unsigned int WarpSizeX_ultra = TileSizeB_ultra / WarpNumX_ultra;
constexpr unsigned int WarpSizeY_ultra = TileSizeA_ultra / WarpNumY_ultra;
constexpr unsigned int HalfWarpSizeX_ultra = WarpSizeX_ultra / 2;
constexpr unsigned int HalfWarpSizeY_ultra = WarpSizeY_ultra / 2;
constexpr unsigned int KernelSize_ultra = 4;
constexpr unsigned int WarpKernelNumX_ultra = WarpSizeX_ultra / (KernelSize_ultra * 2);
constexpr unsigned int WarpKernelNumY_ultra = WarpSizeY_ultra / (KernelSize_ultra * 2);
constexpr unsigned int ThreadNum_ultra = 32 * WarpNumX_ultra * WarpNumY_ultra;


__global__ void gemm_naive(float* a, float* b, float* c, size_t a_x, size_t b_x);
__global__ void gemm(float* a, float* b, float* c, size_t a_x, size_t b_x);
__global__ void gemm_fast(float* a, float* b, float* c, size_t a_x, size_t b_x);
__global__ void gemm_faster(float* a, float* b, float* c, size_t a_x, size_t b_x);
__global__ void gemm_extreme(float* a, float* b, float* c, size_t a_x, size_t b_x);
__global__ void gemm_ultra(float* a, float* b, float* c, size_t a_x, size_t b_x);
cudaError_t gemm_cutlass(float* a, float* b, float* c, size_t a_x, size_t b_x, size_t a_y);
