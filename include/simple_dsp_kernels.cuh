#pragma omce

#include <cufft.h>

#define NUM_FFT_SIZE_BITS 6
#define FFT_SIZE (1u << (NUM_FFT_SIZE_BITS))
#define PI 3.1415926536f

// FFT Implementation from C++ Cookbook:
// https://www.oreilly.com/library/view/c-cookbook/0596007612/ch11s18.html#cplusplusckbk-CHP-11-EX-33
__device__ 
unsigned int bit_reverse(unsigned int x, int log2n);

// Complex exponential function: result = e^(val)
//__device__ __forceinline__
//cufftComplex complex_exponential(cufftComplex val);

 // Complex exponential function: result = e^(val)
// TODO: Look into optimizing the sincosf
// Check out: https://www.drdobbs.com/cpp/a-simple-and-efficient-fft-implementatio/199500857?pgno=3
// The template metaprogramming from there for sincos series might work here?
__device__ __forceinline__ 
cufftComplex complex_exponential(cufftComplex val) {
   cufftComplex result;
   float temp_exp = expf(val.x);
   sincosf(val.y, &result.y, &result.x);
   result.x *= temp_exp;
   result.y *= temp_exp;
   return result;
}

// Complex Phase Angle function
__device__ 
float complex_phase_angle(const cufftComplex& val);


__global__
void calc_con_sqrs(cufftComplex* __restrict__ con_sqrs, const cufftComplex* __restrict__ frequencies, const int num_frequencies);

__global__
void calc_psds(float* __restrict__ psds, const cufftComplex* __restrict__ con_sqrs, const int num_con_sqrs, const float log10num_con_sqrs);

__device__ 
void cookbook_fft64(cufftComplex* frequencies, const cufftComplex* __restrict__ samples, const int num_samples);

__global__
void simple_dsp_kernel(float* __restrict__ psds, cufftComplex* __restrict__ con_sqrs, cufftComplex* frequencies, const cufftComplex* __restrict__ samples,  const int num_samples, const float log10num_con_sqrs);
