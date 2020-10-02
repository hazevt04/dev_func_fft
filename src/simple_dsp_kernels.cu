#include <stdio.h>
#include <cuComplex.h>

#include "simple_dsp_kernels.cuh"

// FFT Implementation from C++ Cookbook:
// https://www.oreilly.com/library/view/c-cookbook/0596007612/ch11s18.html#cplusplusckbk-CHP-11-EX-33
__device__ 
unsigned int bit_reverse(unsigned int x, int log2n) {
   int n = 0;
   for (int i = 0; i < log2n; i++) {
      n <<= 1;
      n |= (x & 1);
      x >>= 1;
   }
   return n;
}


__device__ 
float complex_phase_angle(const cufftComplex& val) { return atan2( cuCimagf(val), cuCrealf(val)); } 


__global__
void cookbook_fft64(cufftComplex* frequencies, const cufftComplex* __restrict__ samples, const int num_samples) {
   int thread_index = threadIdx.x;

   const cufftComplex J = make_cuComplex(0,-1);

   if ( thread_index == 0 ) printf( "cookbook_fft64- Here1\n" );

   __shared__ cufftComplex sh_samples[FFT_SIZE];
   int sh_index = bit_reverse(thread_index, NUM_FFT_SIZE_BITS);
   sh_samples[thread_index] = samples[sh_index];
   __syncthreads();
   
   if ( thread_index == 0 ) printf( "cookbook_fft64- Here2\n" );
      
   for (int s = 1; s <= NUM_FFT_SIZE_BITS; ++s) {
      if ( thread_index == 0 ) printf( "cookbook_fft64- In outer for loop-Here3\n" );
      unsigned int m = 1 << s;
      unsigned int m2 = m >> 1;
      cufftComplex w = make_cuComplex(1, 0);
      cufftComplex wm = complex_exponential( cuCmulf( J, make_cuComplex( (PI / m2), 0 ) ) );
      for (unsigned int j = 0; j != m2; ++j) {
         if ( thread_index == 0 ) printf( "cookbook_fft64- In inner for loop-Here4\n" );
         for (int k = j; k < FFT_SIZE; k += m) {
            cufftComplex t = cuCmulf( w, sh_samples[k + m2] );
            cufftComplex u = make_cuComplex( sh_samples[k].x, sh_samples[k].y );
            sh_samples[k] = cuCaddf( u, t );
            sh_samples[k + m2] = cuCsubf( u, t );
         }
         w = cuCmulf( w, wm );
      }
   } // end of for (int s = 1; s <= log2n; ++s) {   

   __syncthreads();
   if ( thread_index == 0 ) printf( "cookbook_fft64- Here5\n" );
   frequencies[thread_index] = sh_samples[thread_index];
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


__global__
void simple_dsp_kernel(float* __restrict__ psds, cufftComplex* __restrict__ con_sqrs, cufftComplex* frequencies, const cufftComplex* __restrict__ samples, const int num_samples, const float log10num_con_sqrs) {
  
   if( threadIdx.x == 0 ) { 
      printf( "simple_dsp_kernel Here1\n");
      cookbook_fft64<<<1,FFT_SIZE>>>(frequencies, samples, num_samples);
   }
   cudaDeviceSynchronize();
   __syncthreads();
   if( threadIdx.x == 0 ) printf( "simple_dsp_kernel Here2\n");
   
   //calc_con_sqrs<<<1,64>>>(con_sqrs, frequencies, num_samples);
   //calc_psds<<<1,64>>>(psds, con_sqrs, num_samples, log10num_con_sqrs);    
}
