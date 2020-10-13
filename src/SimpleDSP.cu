#include "SimpleDSP.cuh"
#include "expected_frequencies.hpp"


SimpleDSP::SimpleDSP( 
      const int new_num_samples,
      const bool new_debug ):
   num_samples( new_num_samples ),
   debug( new_debug ) {
   
   try {
      cudaError_t cerror = cudaSuccess;
      // Since num_samples will always be a power of 2, just left shift
      // for multiplication.
      size_t num_bytes = sizeof( cufftComplex ) * num_samples;
      size_t num_float_bytes = sizeof( float ) * num_samples;
      
      log10num_con_sqrs = (float)std::log10( num_samples );

      dout << __func__ << "(): num_samples is " << num_samples << "\n"; 
      dout << __func__ << "(): FFT_SIZE is " << FFT_SIZE << "\n"; 
      dout << __func__ << "(): NUM_FFT_SIZE_BITS is " << NUM_FFT_SIZE_BITS << "\n"; 
      dout << __func__ << "(): log10num_con_sqrs is " << log10num_con_sqrs << "\n"; 
      dout << __func__ << "(): num_bytes is " << num_bytes << "\n"; 
      dout << __func__ << "(): num_float_bytes is " << num_float_bytes << "\n"; 

      try_cuda_func_throw( cerror, cudaSetDevice(0) );
      
      try_cuda_func_throw( cerror, cudaDeviceReset() );

      // Instruct CUDA to yield its thread when waiting for results from the device. 
      // This can increase latency when waiting for the device, but can increase the 
      // performance of CPU threads performing work in parallel with the device.
      try_cuda_func_throw( cerror, cudaSetDeviceFlags( cudaDeviceScheduleYield ) );

      try_cuda_func_throw( cerror, cudaMallocManaged( (void**)&samples, num_bytes * 2 ) );
      try_cuda_func_throw( cerror, cudaMallocManaged( (void**)&frequencies, num_bytes ) );
      try_cuda_func_throw( cerror, cudaMallocManaged( (void**)&sfrequencies, num_bytes ) );
      try_cuda_func_throw( cerror, cudaMallocManaged( (void**)&con_sqrs, num_bytes ) );
      try_cuda_func_throw( cerror, cudaMallocManaged( (void**)&psds, num_float_bytes ) );

      //gen_cufftComplexes( samples, num_samples, -100.0, 100.0 );
      read_binary_file<cufftComplex>(samples,
         "../testdataBPSKcomplex.bin",
         num_samples,
         debug );

      if ( debug ) {
         const char delim[] = " ";
         const char suffix[] = "\n";
         print_cufftComplexes(samples, num_samples, "Samples from testfile: ", delim, suffix);
      }

      for( int index = 0; index < num_samples; ++index ) {
         frequencies[index].x = 0;
         frequencies[index].y = 0;
         sfrequencies[index].x = 0;
         sfrequencies[index].y = 0;
         con_sqrs[index].x = 0;
         con_sqrs[index].y = 0;
         psds[index] = 0;
      } 

      //try_cuda_func_throw( cerror, cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte ) );

   } catch (std::exception& ex) {
      throw std::runtime_error{
         std::string{"SimpleDSP::" + std::string{__func__} + "(): " + ex.what()}};
   }
}


void SimpleDSP::operator()() {
   try {
      run();
   } catch (std::exception& ex) {
      std::cout << "ERROR: SimpleDSP::" << __func__ << "(): " << ex.what() << " Exiting.\n";
   }
}


void SimpleDSP::run() {
   try {
      cudaError_t cerror = cudaSuccess;
      Duration_ms duration_ms;
      float gpu_milliseconds = 0;
      
      int threads_per_block = FFT_SIZE;
      int num_blocks = (num_samples + threads_per_block -1)/threads_per_block;
      size_t num_shared_bytes = num_blocks * FFT_SIZE * sizeof( cufftComplex );

      dout << __func__ << "(): num_samples = " << num_samples << "\n";
      dout << __func__ << "(): threads_per_block = " << threads_per_block << "\n";
      dout << __func__ << "(): num_blocks = " << num_blocks << "\n\n";
      
      // Typedef for Time_Point is in my_utils.hpp
      Time_Point start = Steady_Clock::now();

      dout << __func__ << "(): Launching simple_dsp_kernel()...\n";
      // Launch the kernel
      simple_dsp_kernel<<<num_blocks, threads_per_block, num_shared_bytes>>>(psds, con_sqrs, 
         sfrequencies, frequencies, samples, num_samples, log10num_con_sqrs);

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      dout << __func__ << "(): Done with simple_dsp_kernel...\n\n"; 

      duration_ms = Steady_Clock::now() - start;
      gpu_milliseconds = duration_ms.count();
      std::cout << "GPU processing took " << gpu_milliseconds << " milliseconds\n\n";
      
      if ( debug ) {
         const char delim[] = " ";
         const char suffix[] = "\n";
         print_cufftComplexes(sfrequencies, num_samples, "Shifted Frequencies from GPU: ", delim, suffix);
         print_cufftComplexes(expected_frequencies, num_samples, "Expected Frequencies: ", delim, suffix);
      }
      
      float max_diff = 1e-3;
      dout << __func__ << "(): Comparing first " << FFT_SIZE << " (FFT Size) results with expected\n\n";

      bool all_are_close = cufftComplexes_are_close( sfrequencies, expected_frequencies, FFT_SIZE, max_diff, debug );
      if (!all_are_close) { 
         throw std::runtime_error( "ERROR: Not all of the frequencies were close to the expected." );
      }
      std::cout << "All Frequencies computed on the GPU were close to the expected.\n\n"; 
      
      /*if ( debug ) {*/
         /*const char space[] = " ";*/
         /*const char comma_space[] = ", ";*/
         /*const char newline[] = "\n";*/
         /*print_cufftComplexes(con_sqrs, num_samples, "Conjugate Squares from GPU: ", space, newline);*/
         /*print_vals<float>(psds, num_samples, "PSDs from GPU: ", comma_space, newline);*/
      /*}*/

   } catch (std::exception& ex) {
      throw std::runtime_error{
         std::string{"SimpleDSP::" + std::string{__func__} + "(): " + ex.what()}};
   }
      

}


SimpleDSP::~SimpleDSP() {
   if (samples) cudaFree(samples);

   if (frequencies) cudaFree(frequencies);

   if (con_sqrs) cudaFree(con_sqrs);

   if (psds) cudaFree(psds);
   
}
