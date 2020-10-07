cmake_minimum_required( VERSION 2.8.0 )

project( dev_func_fft )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDA QUIET REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)

set(INC_DIRS
   "include"
   "${CUDA_INCLUDE_DIRECTORIES}"
)

cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)

LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
LIST(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

set(EXEC_NAME "${CMAKE_PROJECT_NAME}")

set(SRCS
   "src/my_utils.cpp"
   "src/my_file_io_funcs.cpp"
   "src/my_cufft_utils.cpp"
   "src/simple_dsp_kernels.cu"
   "src/SimpleDSP.cu"
   #"src/dammit.cu"
   "src/main.cpp"
   )

include_directories( ${INC_DIRS} )
cuda_add_executable( ${EXEC_NAME} ${SRCS} )
target_link_libraries( ${EXEC_NAME} ${CUDA_cudadevrt_LIBRARY} ${CUDA_cufft_LIBRARY} m )


