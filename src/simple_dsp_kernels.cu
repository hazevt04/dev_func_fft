#include <stdio.h>
#include <cuComplex.h>
#include <cooperative_groups.h>

#include "simple_dsp_kernels.cuh"


// FFT Implementation from C++ Cookbook:
// https://www.oreilly.com/library/view/c-cookbook/0596007612/ch11s18.html#cplusplusckbk-CHP-11-EX-33
__device__ 
unsigned int bit_reverse(unsigned int x, int log2n) {
   unsigned int n = 0;
   for (int i = 0; i < log2n; i++) {
      n <<= 1;
      n |= (x & 1);
      x >>= 1;
   }
   return n;
}


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

      if (thread_index < HALF_FFT_SIZE) {
         sh_frequencies[thread_index] = sh_frequencies[thread_index + HALF_FFT_SIZE];
      } else if ((thread_index >= HALF_FFT_SIZE) && (thread_index < FFT_SIZE)) {
         sh_frequencies[thread_index] = sh_frequencies[thread_index - HALF_FFT_SIZE];
      }
      __syncthreads();

      shifted_frequencies[global_index] = sh_frequencies[thread_index];
   }
} // end of cufft_shift


__device__
void cookbook_fft64(cufftComplex* frequencies, cufftComplex* __restrict__ sh_samples, const int num_samples) {
   int global_index = threadIdx.x + blockIdx.x * blockDim.x;
   int thread_index = threadIdx.x;

   if ( global_index < num_samples ) {

      if ( thread_index == 0 ) {
         int br_index = bit_reverse( (int)thread_index, NUM_FFT_SIZE_BITS );
         sh_samples[thread_index].x = sh_samples[br_index].x;
         sh_samples[thread_index].y = sh_samples[br_index].y;
         
         const cufftComplex J = make_cuComplex(0,-1);
         for (int s = 1; s <= NUM_FFT_SIZE_BITS; ++s) {
            unsigned int m = (1 << s);
            unsigned int m2 = (m >> 1);
            cufftComplex w = make_cuComplex(1, 0);
            cufftComplex wm = complex_exponential( cuCmulf( J, make_cuComplex( (PI / m2), 0 ) ) );
            for (unsigned int j = 0; j != m2; ++j) {
               for (int k = j; k < FFT_SIZE; k += m) {
                  cufftComplex t = cuCmulf( w, sh_samples[k + m2] );
                  cufftComplex u = make_cuComplex( sh_samples[k].x, sh_samples[k].y );
                  sh_samples[k] = cuCaddf( u, t );
                  sh_samples[k + m2] = cuCsubf( u, t );
               }
               w = cuCmulf( w, wm );
            } // end of for (unsigned int j = 0; j != m2; ++j) {
         } // end of for (int s = 1; s <= NUM_FFT_SIZE_BITS; ++s) {
      } // if ( thread_index == 0 ) {
      __syncthreads();
      
      frequencies[global_index].x = sh_samples[thread_index].x;
      frequencies[global_index].y = sh_samples[thread_index].y;
   } // if ( global_index < num_samples ) {
   
} // end of cookbook_fft64


__global__
void simple_dsp_kernel(float* __restrict__ psds, cufftComplex* __restrict__ con_sqrs, cufftComplex* sfrequencies, cufftComplex* frequencies,
      const cufftComplex* __restrict__ samples, const int num_samples, const float log10num_con_sqrs) {
  
   extern __shared__ cufftComplex sh_samples[];
   
   int thread_index = threadIdx.x;
   int global_index = threadIdx.x + blockIdx.x * blockDim.x;

   sh_samples[thread_index] = samples[global_index];
   __syncthreads();

   cookbook_fft64( frequencies, sh_samples, num_samples );
   __syncthreads();
   cufft_shift( sfrequencies, frequencies, num_samples );
   //__syncthreads();
   /*calc_con_sqrs( con_sqrs, sfrequencies, num_samples );*/
   /*calc_psds( psds, con_sqrs, num_samples, log10num_con_sqrs);*/
}
