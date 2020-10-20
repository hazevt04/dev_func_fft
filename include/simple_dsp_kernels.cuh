#pragma once

#include "sm_fft.cuh"

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

__device__
void calc_con_sqrs(float* __restrict__ sh_con_sqrs, const cufftComplex* __restrict__ sh_frequencies);

__device__
void calc_psds(float* __restrict__ sh_psds, const float* __restrict__ sh_con_sqrs, const float log10num_con_sqrs);


template<class const_params>
__global__ void simple_dsp_kernel(float* __restrict__ d_psds, float* __restrict__ d_con_sqrs, cufftComplex* d_sfrequencies, 
   const cufftComplex* __restrict__ d_samples, const int num_samples, const float log10num_con_sqrs) {

	__shared__ cufftComplex sh_samples[const_params::fft_sm_required];
	__shared__ float sh_con_sqrs[const_params::fft_sm_required];
	__shared__ float sh_psds[const_params::fft_sm_required];

	sh_samples[threadIdx.x]                                           = d_samples[threadIdx.x + blockIdx.x*const_params::fft_length];
	sh_samples[threadIdx.x + const_params::fft_length_quarter]        = d_samples[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_quarter];
	sh_samples[threadIdx.x + const_params::fft_length_half]           = d_samples[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_half];
	sh_samples[threadIdx.x + const_params::fft_length_three_quarters] = d_samples[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_three_quarters];
	
	__syncthreads();
	do_SMFFT_CT_DIT<const_params>(sh_samples);
	__syncthreads();
	calc_con_sqrs(sh_con_sqrs, sh_samples);
	__syncthreads();
	calc_psds(sh_psds, sh_con_sqrs, log10num_con_sqrs);
	__syncthreads();

	d_sfrequencies[threadIdx.x + blockIdx.x*const_params::fft_length]                                           = sh_samples[threadIdx.x];
	d_sfrequencies[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_quarter]        = sh_samples[threadIdx.x + const_params::fft_length_quarter];
	d_sfrequencies[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_half]           = sh_samples[threadIdx.x + const_params::fft_length_half];
	d_sfrequencies[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_three_quarters] = sh_samples[threadIdx.x + const_params::fft_length_three_quarters];

	d_con_sqrs[threadIdx.x + blockIdx.x*const_params::fft_length]                                           = sh_con_sqrs[threadIdx.x];
	d_con_sqrs[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_quarter]        = sh_con_sqrs[threadIdx.x + const_params::fft_length_quarter];
	d_con_sqrs[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_half]           = sh_con_sqrs[threadIdx.x + const_params::fft_length_half];
	d_con_sqrs[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_three_quarters] = sh_con_sqrs[threadIdx.x + const_params::fft_length_three_quarters];
	
	d_psds[threadIdx.x + blockIdx.x*const_params::fft_length]                                           = sh_psds[threadIdx.x];
	d_psds[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_quarter]        = sh_psds[threadIdx.x + const_params::fft_length_quarter];
	d_psds[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_half]           = sh_psds[threadIdx.x + const_params::fft_length_half];
	d_psds[threadIdx.x + blockIdx.x*const_params::fft_length + const_params::fft_length_three_quarters] = sh_psds[threadIdx.x + const_params::fft_length_three_quarters];
	
}
