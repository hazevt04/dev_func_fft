#include <cuda_runtime>
#include <cufft.h>


#include "my_cuda_utils.hpp"
#include "my_cufft_utils.hpp"

#include "my_utils.hpp"
#include "my_file_io_funcs.hpp"

#include "simple_dsp_kernels.cuh"

#include "SimpleDSP.cuh"

SimpleDSP::SimpleDSP( 
      const int new_num_bits,
      const bool new_debug ):
   num_bits( new_num_bits ),
   debug( new_debug ) {
   
   try {
      cudaError_t cerror = cudaSuccess;
      // Since num_samples will always be a power of 2, just left shift
      // for multiplication.
      size_t num_bytes = sizeof( cufftComplex ) << num_bits;
      size_t num_psd_bytes = sizeof( float ) << num_bits;
      
      num_samples = ( 1u << new_num_bits );
      log10num_con_sqrs = (float)std::log10( num_samples );

      try_cuda_func_throw( cerror, cudaMallocHost( (void**)&samples, num_bytes ) );
      try_cuda_func_throw( cerror, cudaMallocHost( (void**)&frequencies, num_bytes ) );
      try_cuda_func_throw( cerror, cudaMallocHost( (void**)&con_sqrs, num_bytes ) );
      try_cuda_func_throw( cerror, cudaMallocHost( (void**)&psds, num_float_bytes ) );

      //gen_cufftComplexes( samples, num_samples, -100.0, 100.0 );
      read_binary_file<cufftComplex>(samples,
         "../testdataBPSKcomplex.bin",
         num_samples,
         debug );

      if ( debug ) {
         const char delim[] = " ";
         const char suffix[] = "\n";
         print_vals<cufftComplex>(samples, "Samples from testfile: ", delim, suffix);
      }

      std::memset( frequencies, 0, num_bytes );
      std::memset( con_sqrs, 0, num_bytes );
      std::memset( psds, 0, num_float_bytes );

      try_cuda_func_throw( cerror, cudaDeviceReset() );
   } catch (std::exception& ex) {
      throw std::runtime_error{
         std::string{"SampleProcFunctor::" + std::string{__func__} + "(): " + ex.what()}};
   }
}


void SimpleDSP::operator()() {
   try {
      run();
   } catch (std::exception& ex) {
      std::cout << "ERROR: SampleProcFunctor::" << __func__ << "(): " << ex.what() << " Exiting.\n";
   }
}


void SimpleDSP::run() {
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
      
      float max_diff = 1e-3;
      bool all_are_close = cufftComplexes_are_close( frequencies, expected_frequencies, num_samples, max_diff debug );
   
   } catch (std::exception& ex) {
      throw std::runtime_error{
         std::string{"SampleProcFunctor::" + std::string{__func__} + "(): " + ex.what()}};
   }
      

}


SimpleDSP::~SimpleDSP() {
   if (samples) cudaFree(samples);

   if (frequencies) cudaFree(frequencies);

   if (con_sqrs) cudaFree(con_sqrs);

   if (psds) cudaFree(psds);
   
}
