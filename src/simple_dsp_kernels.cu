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


__global__
void calc_con_sqrs(cufftComplex* __restrict__ con_sqrs, const cufftComplex* __restrict__ frequencies, const int num_frequencies) {

   //Assuming one stream
   int global_index = ( blockIdx.x * blockDim.x ) + threadIdx.x;
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   
   for (int index = global_index; index < num_frequencies; index+=stride) {   
      cufftComplex conj = cuConjf(frequencies[index]);
      con_sqrs[index] = cuCmulf( conj, conj );
   }
}


__global__ 
void calc_psds(float* __restrict__ psds, const cufftComplex* __restrict__ con_sqrs, const int num_con_sqrs, const float log10num_con_sqrs) {
   
   // Assuming one stream
   int global_index = ( blockIdx.x * blockDim.x ) + threadIdx.x;
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   
   for (int index = global_index; index < num_con_sqrs; index+=stride) {   
      psds[index] = 10*__log10f( cuCabsf(con_sqrs[index]) ) - log10num_con_sqrs;
      
   }

}

namespace cg = cooperative_groups;  

__device__
void cookbook_fft64(cufftComplex* frequencies, cufftComplex* __restrict__ sh_samples, const int num_samples) {
   auto group = cg::this_thread_block();

   for (int index = group.thread_rank(); index < num_samples; index += group.size() ) {

      int br_index = (int)bit_reverse((int)index, NUM_FFT_SIZE_BITS);
      sh_samples[index].x = sh_samples[br_index].x;
      sh_samples[index].y = sh_samples[br_index].y;

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
               group.sync();
               sh_samples[k] = cuCaddf( u, t );
               sh_samples[k + m2] = cuCsubf( u, t );
               group.sync();
            }
            w = cuCmulf( w, wm );
         } // end of for (unsigned int j = 0; j != m2; ++j) {
      } // end of for (int s = 1; s <= NUM_FFT_SIZE_BITS; ++s) {
      frequencies[index].x = sh_samples[index].x;
      frequencies[index].y = sh_samples[index].y;
   } // end of for (int index = grid.thread_rank(); index < num_samples; index += grid.size() ) {
   
} // end of cookbook_fft64


__global__
void simple_dsp_kernel(float* __restrict__ psds, cufftComplex* __restrict__ con_sqrs, cufftComplex* frequencies, 
      const cufftComplex* __restrict__ samples, const int num_samples, const float log10num_con_sqrs) {
  
   extern __shared__ cufftComplex sh_samples[];
   
   auto group = cg::this_thread_block();
   int thread_index = group.thread_rank();

   sh_samples[thread_index] = samples[thread_index];

   //if ( group.thread_rank() == 0 ) {
   cookbook_fft64( frequencies, sh_samples, num_samples );
   //}
}
