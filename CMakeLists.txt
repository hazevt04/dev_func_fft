cmake_minimum_required( VERSION 3.8 )

project( dev_func_fft LANGUAGES CXX CUDA )

set(CUDA_VERBOSE_BUILD ON)

find_package(CUDA REQUIRED)
cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)

LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
LIST(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
LIST(APPEND CUDA_NVCC_FLAGS "--maxrregcount=64")
IF(CMAKE_BUILD_TYPE MATCHES Debug)
   message(status "debug mode")
   #LIST(APPEND CUDA_NVCC_FLAGS "--device-debug")
   LIST(APPEND CUDA_NVCC_FLAGS "--generate-line-info")
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

include_directories(
   "include"
   "${CUDA_INCLUDE_DIRS}"
)

set(CUDA_SEPARABLE_COMPILATION ON)

set(INC_DIRS
   "include"
   "${CUDA_INCLUDE_DIRECTORIES}"
)

set(SRCS
   "src/my_utils.cpp"
   "src/my_file_io_funcs.cpp"
   "src/my_cufft_utils.cpp"
   "src/sincos_table.cu"
   "src/simple_dsp_kernels.cu"
   "src/SimpleDSP.cu"
   "src/main.cpp"
   )

set(EXEC_NAME "${CMAKE_PROJECT_NAME}")

add_executable( ${EXEC_NAME} ${SRCS} )
target_link_libraries( ${EXEC_NAME} ${CUDA_cudadevrt_LIBRARY} m )

set_target_properties( ${EXEC_NAME}
   PROPERTIES
   CXX_STANDARD 11
   CXX_STANDARD_REQUIRED YES
   CXX_EXTENSIONS NO
   CUDA_SEPARABLE_COMPILATION ON
   CMAKE_CUDA_STANDARD 11
   CMAKE_CUDA_STANDARD_REQUIRED ON
)


