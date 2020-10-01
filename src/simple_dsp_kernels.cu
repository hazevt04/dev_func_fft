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


__global__
void cookbook_fft(cufftComplex* frequencies, const cufftComplex* __restrict__ samples, const int num_bits) {
   
   const cufftComplex J = make_cuComplex(0,-1);
   unsigned int fft_size = 1u << num_bits;

   const double PI = 3.1415926536;
   for (unsigned int index = 0; index != fft_size; ++index) {
      unsigned int br_index = bit_reverse(index, num_bits);
      frequencies[br_index] = samples[index];
   }

   for (int s = 1; s <= num_bits; ++s) {
      unsigned int m = 1 << s;
      unsigned int m2 = m >> 1;
      cufftComplex w = make_cuComplex(1, 0);
      cufftComplex wm = complex_exponential( cuCmulf( J, make_cuComplex( (PI / m2), 0 ) ) );
      for (unsigned int j = 0; j != m2; ++j) {
         for (int k = j; k < (int)fft_size; k += m) {
            cufftComplex t = cuCmulf( w, frequencies[k + m2] );
            cufftComplex u = make_cuComplex( frequencies[k].x, frequencies[k].y );
            frequencies[k] = cuCaddf( u, t );
            frequencies[k + m2] = cuCsubf( u, t );
         }
         w = cuCmulf( w, wm );
      }
   } // end of for (int s = 1; s <= log2n; ++s) {   
}


__global__
void calc_con_sqrs(cufftComplex* __restrict__ con_sqrs, const cufftComplex* __restrict__ frequencies, const int num_frequencies) {

   // Assuming one stream
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
void simple_dsp_kernel(float* __restrict__ psds, cufftComplex* __restrict__ con_sqrs, cufftComplex* frequencies, const cufftComplex* __restrict__ samples, const int num_bits, const int num_samples, const float log10num_con_sqrs) {
  
   cookbook_fft<<<1,64>>>(frequencies, samples, num_bits);
   calc_con_sqrs<<<1,64>>>(con_sqrs, frequencies, num_samples);
   calc_psds<<<1,64>>>(psds, con_sqrs, num_samples, log10num_con_sqrs);    

}
