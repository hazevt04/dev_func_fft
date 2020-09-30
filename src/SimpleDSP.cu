#include <cuda_runtime>
#include <cufft.h>

#include "my_cuda_utils.hpp"
#include "my_cufft_utils.hpp"

#include "simple_dsp_kernels.cuh"

#include "SimpleDSP.cuh"


SimpleDSP::SimpleDSP( int new_num_bits ):
   num_bits( new_num_bits ) {
   
   try {
      cudaError_t cerror = cudaSuccess;
      // Since num_samples will always be a power of 2, just left shift
      // for multiplication.
      size_t num_bytes = sizeof( cufftComplex ) << num_bits;
      size_t num_psd_bytes = sizeof( float ) << num_bits;
      
      num_samples = ( 1u << new_num_bits );
      log10num_con_sqrs = (float)std::log10( num_samples );


      try_cuda_func_throw( cerror, cudaMallocHost( (void**)&samples, num_bytes );
      try_cuda_func_throw( cerror, cudaMallocHost( (void**)&frequencies), num_bytes );
      try_cuda_func_throw( cerror, cudaMallocHost( (void**)&con_sqrs, num_bytes );
      try_cuda_func_throw( cerror, cudaMallocHost( (void**)&psds, num_float_bytes );

      try_cuda_func_throw( cerror, cudaDeviceReset() );
   } catch (std::exception& ex) {
      throw std::runtime_error{
         std::string{"SampleProcFunctor::" + std::string{__func__} + "(): " + ex.what()}};
   }
}

void SimpleDSP::operator()() {
   try {
      cudaError_t cerror = cudaSuccess;
      Duration duration_ms;
      float gpu_milliseconds = 0;

      // Typedef for Time_Point is in my_utils.hpp
      Time_Point start = Steady_Clock::now();

      simple_dsp_kernel<<<1,64>>>(psds, con_sqrs, frequencies, samples, num_bits, num_samples, log10num_con_sqrs);

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );

      duration_ms = Steady_Clock::now() - start;
      gpu_milliseconds = duration_ms.count();
      std::cout << "GPU processing took " << gpu_milliseconds << " milliseconds\n";

   } catch (std::exception& ex) {
      std::cout << "ERROR: SampleProcFunctor::" << __func__ << "(): " << ex.what() << " Exiting.\n";
   }
}


SimpleDSP::~SimpleDSP() {
   if (samples) cudaFree(samples);

   if (frequencies) cudaFree(frequencies);

   if (con_sqrs) cudaFree(con_sqrs);

   if (psds) cudaFree(psds);
   
}
