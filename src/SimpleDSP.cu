#include "SimpleDSP.cuh"
#include "simple_dsp_kernels.cuh"

SimpleDSP::SimpleDSP( 
      const int new_num_samples,
      const bool new_debug ):
   num_samples( new_num_samples ),
   debug( new_debug ) {
   
   try {
      cudaError_t cerror = cudaSuccess;

      num_ffts = num_samples >> NUM_FFT_SIZE_BITS;

      // Since num_samples will always be a power of 2, just left shift
      // for multiplication.
      size_t num_bytes = sizeof( cufftComplex ) * num_samples;
      size_t num_float_bytes = sizeof( float ) * num_samples;
      
      threads_per_block = FFT_SIZE/2;
      num_blocks = num_ffts/2;

      dout << __func__ << "(): num_samples is " << num_samples << "\n"; 
      dout << __func__ << "(): num_ffts is " << num_ffts << "\n\n"; 

      dout << __func__ << "(): threads_per_block = " << threads_per_block << "\n";
      dout << __func__ << "(): num_blocks = " << num_blocks << "\n";

      dout << __func__ << "(): FFT_SIZE is " << FFT_SIZE << "\n"; 
      dout << __func__ << "(): NUM_FFT_SIZE_BITS is " << NUM_FFT_SIZE_BITS << "\n";
      dout << __func__ << "(): TEN_LOG_TEN_FFT_SIZE is " << TEN_LOG_TEN_FFT_SIZE << "\n\n";

      dout << __func__ << "(): num_bytes is " << num_bytes << "\n"; 
      dout << __func__ << "(): num_float_bytes is " << num_float_bytes << "\n\n"; 

      size_t free_gpu_memory_bytes = 0u;
      size_t total_gpu_memory_bytes = 0u;
      int device_id = 0;
      try_cuda_func_throw( cerror, cudaSetDevice(device_id) );
      try_cuda_func_throw( cerror, cudaSetDeviceFlags( cudaDeviceScheduleYield ) );

      // Check GPU Memory
      try_cuda_func_throw( cerror, cudaMemGetInfo( &free_gpu_memory_bytes, &total_gpu_memory_bytes ) );
      dout << __func__ << "(): GPU has " << ((float)free_gpu_memory_bytes)/1048576.f 
         << " MiB free out of " << ((float)total_gpu_memory_bytes)/1048576.f << " MiB available\n";
      size_t total_needed_bytes = 2 * num_bytes + 2 * num_float_bytes;
      if ( total_needed_bytes > free_gpu_memory_bytes ) {
         throw std::runtime_error{ std::string{"ERROR: Not enough free GPU memory. Need "} + 
            std::to_string(((float)total_needed_bytes)/1048576.f) + 
            std::string{" MiB for processing "} + std::to_string(num_samples) + " samples." };
      }

      // Instruct CUDA to yield its thread when waiting for results from the device. 
      // This can increase latency when waiting for the device, but can increase the 
      // performance of CPU threads performing work in parallel with the device.

      try_cuda_func_throw( cerror, cudaHostAlloc( (void**)&samples, num_bytes, cudaHostAllocMapped ) );
      try_cuda_func_throw( cerror, cudaHostAlloc( (void**)&sfrequencies, num_bytes, cudaHostAllocMapped ) );
      try_cuda_func_throw( cerror, cudaHostAlloc( (void**)&con_sqrs, num_float_bytes, cudaHostAllocMapped ) );
      try_cuda_func_throw( cerror, cudaHostAlloc( (void**)&psds, num_float_bytes, cudaHostAllocMapped ) );

      try_cuda_func_throw( cerror, cudaHostGetDevicePointer( (void**)&d_samples, (void*)samples, 0 ) );
      try_cuda_func_throw( cerror, cudaHostGetDevicePointer( (void**)&d_sfrequencies, (void*)sfrequencies, 0 ) );
      try_cuda_func_throw( cerror, cudaHostGetDevicePointer( (void**)&d_con_sqrs, (void*)con_sqrs, 0 ) );
      try_cuda_func_throw( cerror, cudaHostGetDevicePointer( (void**)&d_psds, (void*)psds, 0 ) );
	
      try_cuda_func_throw( cerror, cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) );
	   try_cuda_func_throw( cerror, cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) );
      
      for( int index = 0; index < num_samples; ++index ) {
         sfrequencies[index].x = 0;
         sfrequencies[index].y = 0;
         con_sqrs[index] = 0;
         psds[index] = 0;
      } 

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
      
      // Typedef for Time_Point is in my_utils.hpp
      Time_Point start = Steady_Clock::now();

      dout << __func__ << "(): Launching simple_dsp_kernel()...\n";
      // Launch the kernel
      simple_dsp_kernel<FFT_64_forward><<<num_blocks, threads_per_block>>>(
         d_psds, d_con_sqrs, d_sfrequencies, d_samples);

      try_cuda_func_throw( cerror, cudaDeviceSynchronize() );
      dout << __func__ << "(): Done with simple_dsp_kernel...\n\n"; 

      duration_ms = Steady_Clock::now() - start;
      gpu_milliseconds = duration_ms.count();
      std::cout << "GPU processing took " << gpu_milliseconds << " milliseconds\n\n";
      
      if ( debug ) {
         const char delim[] = ", ";
         const char suffix[] = "\n";
         print_cufftComplexes(expected_sfrequencies, num_samples, "Expected Shifted Frequencies: ", delim, suffix);
         print_cufftComplexes(sfrequencies, num_samples, "Shifted Frequencies from GPU: ", delim, suffix);
         print_vals<float>(expected_con_sqrs, num_samples, "Expected Conjugate Squares: ", delim, suffix);
         print_vals<float>(con_sqrs, num_samples, "Conjugate Squares: ", delim, suffix);
         print_vals<float>(expected_psds, num_samples, "Expected PSDs: ", delim, suffix);
         print_vals<float>(psds, num_samples, "PSDs: ", delim, suffix);
      }
      
      float max_diff = 1e-2;
      dout << __func__ << "(): Comparing " << num_samples << " shifted frequencies with expected\n";

      bool all_are_close = cufftComplexes_are_close( sfrequencies, expected_sfrequencies, num_samples, max_diff, debug );
      if (!all_are_close) { 
         throw std::runtime_error( "Not all of the shifted frequencies were close to the expected." );
      }
      std::cout << "All Shifted Frequencies computed on the GPU were close to the expected.\n"; 

      max_diff = 1;
      dout << __func__ << "(): Comparing " << num_samples << " conjugate squares with expected\n";
      all_are_close = vals_are_close<float>( con_sqrs, expected_con_sqrs, num_samples, max_diff, debug );
      if (!all_are_close) { 
         throw std::runtime_error( "Not all of the conjugate squares were close to the expected." );
      }
      std::cout << "All Conjugate Squares computed on the GPU were close to the expected.\n"; 

      dout << __func__ << "(): Comparing " << num_samples << " psds with expected\n\n";
      all_are_close = vals_are_close<float>( psds, expected_psds, num_samples, max_diff, debug );
      if (!all_are_close) { 
         throw std::runtime_error( "Not all of the psds were close to the expected." );
      }
      std::cout << "All PSDs computed on the GPU were close to the expected.\n";
      std::cout << "\n"; 

   } catch (std::exception& ex) {
      throw std::runtime_error{
         std::string{"SimpleDSP::" + std::string{__func__} + "(): " + ex.what()}};
   }
      

}

// Don't throw exceptions from the 
// destructor.
SimpleDSP::~SimpleDSP() {
   if (samples) cudaFreeHost(samples);

   if (frequencies) cudaFreeHost(sfrequencies);

   if (con_sqrs) cudaFreeHost(con_sqrs);

   if (psds) cudaFreeHost(psds);

   cudaDeviceReset();
}
