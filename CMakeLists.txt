cmake_minimum_required(VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS True)
project(learn-cuda
	VERSION 0.0.1
	DESCRIPTION "Learn CUDA"
	LANGUAGES C CXX CUDA
)


if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/tiny-stl/CMakeLists.txt")
	message(FATAL_ERROR
		"Some learn-cuda dependencies are missing. "
		"If you forgot the \"--recursive\" flag when cloning this project, "
		"this can be fixed by calling \"git submodule update --init --recursive\"."
	)
endif()


if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
if (MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-bigobj")
endif()

list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")


include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
set(GLFRAMEWORK_CUDA_ARCHITECTURES "" CACHE STRING "Build glframework for a specific GPU architecture.")
if (DEFINED ENV{GLFRAMEWORK_CUDA_ARCHITECTURES})
	message(STATUS "Obtained target architecture from environment variable GLFRAMEWORK_CUDA_ARCHITECTURES=$ENV{GLFRAMEWORK_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES $ENV{GLFRAMEWORK_CUDA_ARCHITECTURES})
elseif (GLFRAMEWORK_CUDA_ARCHITECTURES)
	message(STATUS "Obtained target architecture from CMake variable GLFRAMEWORK_CUDA_ARCHITECTURES=${GLFRAMEWORK_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES ${GLFRAMEWORK_CUDA_ARCHITECTURES})
else()
	set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
	# set(CMAKE_CUDA_ARCHITECTURES 75)
endif()


add_subdirectory(matmult)