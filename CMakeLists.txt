cmake_minimum_required( VERSION 3.8.0 )

project( dev_func_fft LANGUAGES CXX CUDA )

#set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 CACHE STRING "CUDA architectures" FORCE)
include(my_cuda_module.cmake)

set(EXEC_NAME "${CMAKE_PROJECT_NAME}")

set(SRCS
   "src/my_utils.cpp"
   "src/my_file_io_funcs.cpp"
   "src/simple_dsp_kernels.cu"
   "src/main.cpp"
   )

add_executable( "${EXEC_NAME}" "${SRCS}" )
target_link_libraries( "${EXEC_NAME}" m )

set_target_properties( ${EXEC_NAME}
    PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        CUDA_STANDARD 11
        CUDA_STANDARD_REQUIRED YES
        CUDA_EXTENSIONS NO
        CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(${EXEC_NAME} PRIVATE
   "include"
)

# I SHOULD NOT HAVE TO DO THIS!!! HOW THE $@#$@$ DO I SET THE MICROARCH OTHERWISE?!!!
# Come on CMake!
#find_library(CUDA REQUIRED)
#CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
#LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})

#target_compile_options(${EXEC_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_72,code=sm_72>)
