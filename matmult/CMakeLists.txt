add_executable(matmult matmult.cu)
set_target_properties(matmult PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(matmult PUBLIC
	"../dependencies/tiny-stl/include"
	"../dependencies/cutlass/include"
)
add_library(matmult_ptx OBJECT matmult.cu)
set_property(TARGET matmult_ptx PROPERTY CUDA_PTX_COMPILATION ON)
target_include_directories(matmult_ptx PUBLIC
	"../dependencies/tiny-stl/include"
	"../dependencies/cutlass/include"
)