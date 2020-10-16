#include <stdio.h>
#include <cuComplex.h>
#include <cooperative_groups.h>

#include "simple_dsp_kernels.cuh"


__device__ 
float complex_phase_angle(const cufftComplex& val) { 
   return atan2( cuCimagf(val), cuCrealf(val)); 
} 

namespace cg = cooperative_groups;  

__device__
void calc_con_sqrs(cufftComplex* __restrict__ con_sqrs, const cufftComplex* __restrict__ frequencies, const int num_frequencies) {

   auto group = cg::this_thread_block();
   
   for (int index = group.thread_rank(); index < num_frequencies; index += group.size() ) {
      cufftComplex conj = cuConjf(frequencies[index]);
      con_sqrs[index] = cuCmulf( conj, conj );
   }
}


__device__ 
void calc_psds(float* __restrict__ psds, const cufftComplex* __restrict__ con_sqrs, const int num_con_sqrs, const float log10num_con_sqrs) {
   
   auto group = cg::this_thread_block();
   
   for (int index = group.thread_rank(); index < num_con_sqrs; index += group.size() ) {
      psds[index] = 10*__log10f( cuCabsf(con_sqrs[index]) ) - log10num_con_sqrs;
   }

}

__device__ 
void cufft_shift(cufftComplex* __restrict__ shifted_frequencies,
   const cufftComplex* __restrict__ frequencies, const int num_frequencies) {

   int thread_index = threadIdx.x;
   int global_index = threadIdx.x + blockIdx.x * blockDim.x;

   __shared__ cufftComplex sh_frequencies[FFT_SIZE];
   sh_frequencies[thread_index] = frequencies[global_index];
   __syncthreads();

   if ( global_index < num_frequencies ) {

      __syncthreads();
      if (thread_index < HALF_FFT_SIZE) {
         sh_frequencies[thread_index] = sh_frequencies[thread_index + HALF_FFT_SIZE];
      } else if ((thread_index >= HALF_FFT_SIZE) && (thread_index < FFT_SIZE)) {
         sh_frequencies[thread_index] = sh_frequencies[thread_index - HALF_FFT_SIZE];
      }
      __syncthreads();

      shifted_frequencies[global_index] = sh_frequencies[thread_index];
   }
} // end of cufft_shift


