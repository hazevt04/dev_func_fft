#include <stdio.h>
#include <cuComplex.h>
#include <cooperative_groups.h>

#include "simple_dsp_kernels.cuh"


// FFT Implementation from C++ Cookbook:
// https://www.oreilly.com/library/view/c-cookbook/0596007612/ch11s18.html#cplusplusckbk-CHP-11-EX-33
__device__ 
unsigned int bit_reverse(unsigned int x, int log2n) {
   unsigned int n = 0;
   printf( "Bit reverse: x = %u\n", x );
   for (int i = 0; i < log2n; i++) {
      n <<= 1;
      n |= (x & 1);
      x >>= 1;
   }
   printf( "Bit reverse: n = %u\n", n );
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
void cookbook_fft64(cufftComplex* frequencies, const cufftComplex* __restrict__ samples, const int num_samples) {
   auto grid = cg::this_grid();

   for (int index = grid.thread_rank(); index < num_samples; index += grid.size() ) {
      const cufftComplex J = make_cuComplex(0,-1);

      int br_index = (int)bit_reverse(index, NUM_FFT_SIZE_BITS);
      printf( "cookbook_fft64- Here1- br_index is %d\n", br_index );
      frequencies[index].x = samples[br_index].x;
      frequencies[index].y = samples[br_index].y;
      // Need sync here
      
      printf( "cookbook_fft64- Here2\n" );
         
      for (int s = 1; s <= NUM_FFT_SIZE_BITS; ++s) {
         printf( "cookbook_fft64- In outer for loop-Here3\n" );
         unsigned int m = 1 << s;
         unsigned int m2 = m >> 1;
         cufftComplex w = make_cuComplex(1, 0);
         cufftComplex wm = complex_exponential( cuCmulf( J, make_cuComplex( (PI / m2), 0 ) ) );
         for (unsigned int j = 0; j != m2; ++j) {
            printf( "cookbook_fft64- In inner for loop-Here4\n" );
            for (int k = j; k < FFT_SIZE; k += m) {
               cufftComplex t = cuCmulf( w, frequencies[k + m2] );
               printf( "cookbook_fft64- In inner for loop-after cuCmulf()\n" );
               cufftComplex u = make_cuComplex( frequencies[k].x, frequencies[k].y );
               printf( "cookbook_fft64- In inner for loop-after creating u: {%f, %f}\n", u.x, u.y );
               frequencies[k] = cuCaddf( u, t );
               printf( "cookbook_fft64- In inner for loop-after cuCaddf()\n" );
               frequencies[k + m2] = cuCsubf( u, t );
               grid.sync();
               printf( "cookbook_fft64- In inner for loop- after cuCsubf()\n" );
            }
            w = cuCmulf( w, wm );
            grid.sync();
            printf( "after innermost grid.sync();\n" );
         }
         grid.sync();
         printf( "after second innermost grid.sync();\n" );
      } // end of for (int s = 1; s <= log2n; ++s) {   
      grid.sync();
      printf( "after third innermost grid.sync();\n" );
   } // end of for (int index = grid.thread_rank(); index < srcSize; index += grid.size() ) {
   grid.sync();
   printf( "after outermost grid.sync();\n" );
}



__global__
void simple_dsp_kernel(float* __restrict__ psds, cufftComplex* __restrict__ con_sqrs, cufftComplex* frequencies, 
      const cufftComplex* __restrict__ samples, const int num_samples, const float log10num_con_sqrs) {
  
   cookbook_fft64( frequencies, samples, num_samples );
}
