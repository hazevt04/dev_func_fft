#include <cuComplex.h>

#include "simple_dsp_kernels.cuh"

// FFT Implementation from C++ Cookbook:
// https://www.oreilly.com/library/view/c-cookbook/0596007612/ch11s18.html#cplusplusckbk-CHP-11-EX-33
__device__ __host__
unsigned int bit_reverse(unsigned int x, int log2n) {
   int n = 0;
   for (int i = 0; i < log2n; i++) {
      n <<= 1;
      n |= (x & 1);
      x >>= 1;
   }
   return n;
}

__device__ __host__ 
void cookbook_fft(cufftComplex* frequencies, const cufftComplex* __restrict__ samples, const int num_bits) {
   
   const cufftComplex J(0,1);
   unsigned int fft_size = 1u << num_bits;

   for (unsigned int index = 0; index != fft_size; ++index) {
      unsigned int br_index = bit_reverse(index, log2n);
      frequencies[br_index] = samples[index];
   }

   for (int s = 1; s <= num_bits; ++s) {
      unsigned int m = 1 << s;
      unsigned int m2 = m >> 1;
      cufftComplex w(1, 0);
      cufftComplex wm = exp(-J * (T(PI) / m2));
      for (unsigned int j = 0; j != m2; ++j) {
         for (int k = j; k < (int)fft_size; k += m) {
            cufftComplex t = w * frequencies[k + m2];
            cufftComplex u = frequencies[k];
            frequencies[k] = u + t;
            frequencies[k + m2] = u - t;
         }
         w *= wm;
      }
   } // end of for (int s = 1; s <= log2n; ++s) {   
}


__device__ __host__
void calc_con_sqrs(cufftComplex* __restrict__ con_sqrs, const cufftComplex* __restrict__ frequencies, const int num_frequencies);
   
   // Assuming one stream
   int global_index = ( blockIdx.x * blockDim.x ) + threadIdx.x;
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   
   for (int index = global_index; index < num_vals; index+=stride) {   
      cufftComplex conj = cuConjf(frequencies[index]);
      con_sqrs[index] = cuCmulf( conj * conj );
   }
}


__device__ __host__
void calc_psds(float* __restrict__ psds, const cufftComplex* __restrict__ con_sqrs, const int num_con_sqrs, const float log10num_con_sqrs) {
   
   // Assuming one stream
   int global_index = ( blockIdx.x * blockDim.x ) + threadIdx.x;
   // stride is set to the total number of threads in the grid
   int stride = blockDim.x * gridDim.x;
   
   for (int index = global_index; index < num_con_sqrs; index+=stride) {   
      psds[index] = 10*log10f( cuCabsf(con_sqrs[index]) ) - 10log10num_con_sqrs;
   }

}


__global__
void simple_dsp_kernel(float* __restrict__ psds, cufftComplex* __restrict__ con_sqrs, cufftComplex* frequencies, const cufftComplex* __restrict__ samples, const int num_bits, const int num_samples) {
  
   cookbook_fft<<<1,1>>>(frequencies, samples, num_bits);
   calc_con_sqrs<<<1,1>>>(con_sqrs, frequencies, num_samples);
   calc_psds<<<1,1>>>(psds, con_sqrs, num_samples, log10num_con_sqrs);    

}
