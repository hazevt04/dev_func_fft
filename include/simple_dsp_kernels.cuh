#pragma omce

#include <cufft.h>

#ifndef NUM_FFT_SIZE_BITS
#define NUM_FFT_SIZE_BITS 6
#endif

#ifndef FFT_SIZE
#define FFT_SIZE (1u << (NUM_FFT_SIZE_BITS))
#endif

#ifndef HALF_FFT_SIZE
#define HALF_FFT_SIZE (1u << ((NUM_FFT_SIZE_BITS)-1))
#endif

// Complex exponential function: result = e^(val)
// TODO: Look into optimizing the sincosf
// Check out: https://www.drdobbs.com/cpp/a-simple-and-efficient-fft-implementatio/199500857?pgno=3
// The template metaprogramming from there for sincos series might work here?
/*__device__ __forceinline__ */
/*cufftComplex complex_exponential(cufftComplex val) {*/
/*   cufftComplex result;*/
/*   float temp_exp = expf(val.x);*/
/*   sincosf(val.y, &result.y, &result.x);*/
/*   result.x *= temp_exp;*/
/*   result.y *= temp_exp;*/
/*   return result;*/
/*}*/

// Complex Phase Angle function
__device__ 
float complex_phase_angle(const cufftComplex& val);

__device__
void calc_con_sqrs(float* __restrict__ sh_con_sqrs, const cufftComplex* __restrict__ sh_frequencies);

__device__
void calc_psds(float* __restrict__ sh_psds, const float* __restrict__ sh_con_sqrs, const float log10num_con_sqrs);


