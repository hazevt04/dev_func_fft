#include <stdio.h>
#include <cuComplex.h>
#include <cooperative_groups.h>

#include "simple_dsp_kernels.cuh"
#define WARP_SIZE 32

__device__ 
float complex_phase_angle(const cufftComplex& val) { 
   return atan2( cuCimagf(val), cuCrealf(val)); 
} 

namespace cg = cooperative_groups;  

__device__
void calc_con_sqrs(float* __restrict__ sh_con_sqrs, const cufftComplex* __restrict__ sh_frequencies) {

	int local_id = threadIdx.x & (WARP_SIZE - 1);
	int warp_id = threadIdx.x/WARP_SIZE;

   int A_index = local_id + (warp_id<<2)*WARP_SIZE;
   int B_index = local_id + (warp_id<<2)*WARP_SIZE + WARP_SIZE;
   int C_index = local_id + (warp_id<<2)*WARP_SIZE + 2*WARP_SIZE;
   int D_index = local_id + (warp_id<<2)*WARP_SIZE + 3*WARP_SIZE;

	cufftComplex A_freq = sh_frequencies[A_index];
	cufftComplex B_freq = sh_frequencies[B_index];
	cufftComplex C_freq = sh_frequencies[C_index];
	cufftComplex D_freq = sh_frequencies[D_index];
   __syncthreads();

   float A_con_sqr = __fmaf_ieee_rn( A_freq.x, A_freq.x, __fmul_rn( A_freq.y, A_freq.y ) );
   float B_con_sqr = __fmaf_ieee_rn( B_freq.x, B_freq.x, __fmul_rn( B_freq.y, B_freq.y ) );
   float C_con_sqr = __fmaf_ieee_rn( C_freq.x, C_freq.x, __fmul_rn( C_freq.y, C_freq.y ) );
   float D_con_sqr = __fmaf_ieee_rn( D_freq.x, D_freq.x, __fmul_rn( D_freq.y, D_freq.y ) );

   /*float A_con_sqr = A_freq.x * A_freq.x + A_freq.y * A_freq.y;*/
   /*float B_con_sqr = B_freq.x * B_freq.x + B_freq.y * B_freq.y;*/
   /*float C_con_sqr = C_freq.x * C_freq.x + C_freq.y * C_freq.y;*/
   /*float D_con_sqr = D_freq.x * D_freq.x + D_freq.y * D_freq.y;*/

   sh_con_sqrs[A_index] = A_con_sqr;
   sh_con_sqrs[B_index] = B_con_sqr;
   sh_con_sqrs[C_index] = C_con_sqr;
   sh_con_sqrs[D_index] = D_con_sqr;
}


__device__ 
void calc_psds(float* __restrict__ sh_psds, const float* __restrict__ sh_con_sqrs, const float log10num_con_sqrs) {
   
	int local_id = threadIdx.x & (WARP_SIZE - 1);
	int warp_id = threadIdx.x/WARP_SIZE;

   int A_index = local_id + (warp_id<<2)*WARP_SIZE;
   int B_index = local_id + (warp_id<<2)*WARP_SIZE + WARP_SIZE;
   int C_index = local_id + (warp_id<<2)*WARP_SIZE + 2*WARP_SIZE;
   int D_index = local_id + (warp_id<<2)*WARP_SIZE + 3*WARP_SIZE;
      
   float A_con_sqr = sh_con_sqrs[A_index];
   float B_con_sqr = sh_con_sqrs[B_index];
   float C_con_sqr = sh_con_sqrs[C_index];
   float D_con_sqr = sh_con_sqrs[D_index];
   __syncthreads();

   float A_psd = 10*__log10f( A_con_sqr ) - log10num_con_sqrs;
   float B_psd = 10*__log10f( B_con_sqr ) - log10num_con_sqrs;
   float C_psd = 10*__log10f( C_con_sqr ) - log10num_con_sqrs;
   float D_psd = 10*__log10f( D_con_sqr ) - log10num_con_sqrs;

   sh_psds[A_index] = A_psd;
   sh_psds[B_index] = B_psd;
   sh_psds[C_index] = C_psd;
   sh_psds[D_index] = D_psd;

   //psds[index] = 10*__log10f( cuCabsf(con_sqrs[index]) ) - log10num_con_sqrs;
}


