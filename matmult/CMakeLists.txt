list(APPEND MATMULT_SOURCES
	"matmult.cu"
	"gemm.cu"
	"gemm_naive.cu"
	"gemm_fast.cu"
	"gemm_faster.cu"
	"gemm_extreme.cu"
	"gemm_ultra.cu"
	"gemm_cutlass.cu"
)
add_executable(matmult ${MATMULT_SOURCES})
set_target_properties(matmult PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(matmult PUBLIC
	"../dependencies/tiny-stl/include"
	"../dependencies/cutlass/include"
)
# target_link_libraries(matmult cublas)

add_library(matmult_ptx OBJECT ${MATMULT_SOURCES})
set_property(TARGET matmult_ptx PROPERTY CUDA_PTX_COMPILATION ON)
target_include_directories(matmult_ptx PUBLIC
	"../dependencies/tiny-stl/include"
	"../dependencies/cutlass/include"
)
# target_link_libraries(matmult_ptx cublas)
