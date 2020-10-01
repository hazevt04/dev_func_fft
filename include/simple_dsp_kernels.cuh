#pragma omce

//#include <cuda_runtime.h>

//#include <cufft.h>
#include <cuComplex.h>

#include "my_cuda_utils.hpp"
#include "my_cufft_utils.hpp"

// FFT Implementation from C++ Cookbook:
// https://www.oreilly.com/library/view/c-cookbook/0596007612/ch11s18.html#cplusplusckbk-CHP-11-EX-33
__device__ 
unsigned int bit_reverse(unsigned int x, int log2n);

// Complex exponential function: result = e^(val)
__device__ __forceinline__ 
cufftComplex complex_exponential(cufftComplex val);

// Complex Phase Angle function
__device__ 
float complex_phase_angle(const cufftComplex& val) { return atan2( cuCimagf(val), cuCrealf(val)); } 

__global__ 
void cookbook_fft(cufftComplex* frequencies, const cufftComplex* __restrict__ samples, const int num_bits);

__global__
void calc_con_sqrs(cufftComplex* __restrict__ con_sqrs, const cufftComplex* __restrict__ frequencies, const int num_frequencies);

__global__
void calc_psds(float* __restrict__ psds, const cufftComplex* __restrict__ con_sqrs, const int num_con_sqrs, const float log10num_con_sqrs);

__global__
void simple_dsp_kernel(float* __restrict__ psds, cufftComplex* frequencies, const cufftComplex* __restrict__ samples, const int num_bits, const float log10num_con_sqrs);
