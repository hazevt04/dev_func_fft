#include "SimpleDSP.cuh"


SimpleDSP::SimpleDSP( 
      const int new_num_samples,
      const bool new_debug ):
   num_samples( new_num_samples ),
   debug( new_debug ) {
   
   try {
      cudaError_t cerror = cudaSuccess;

      num_streams = default_num_streams;
      
      if ( const char* pNUM_STREAMS_ENV = std::getenv( "NUM_STREAMS" ) ) {
         char* pEnd = nullptr;
         int t_num_streams = std::strtol( pNUM_STREAMS_ENV, &pEnd, 10 );
         if ( '\0' == *pEnd ) {
            num_streams = t_num_streams;
         } else {
            std::cout << __func__ << "(): WARNING: NUM_STREAMS environment variable set to invalid value: "
               << std::string{pNUM_STREAMS_ENV} << "\n"
               << "Using the default value for number of streams " << default_num_streams << "\n"; 
         }
      }

      threads_per_block = FFT_SIZE;
      num_blocks = (num_samples + threads_per_block -1)/threads_per_block;
      num_stream_samples = num_samples/num_streams;
      num_stream_blocks = (num_blocks + num_streams -1)/num_streams;

      num_shared_bytes = 0;
      
      // Since num_samples will always be a power of 2, just left shift
      // for multiplication.
      num_bytes = sizeof( cufftComplex ) * num_samples;
      num_float_bytes = sizeof( float ) * num_samples;
      num_stream_bytes = num_bytes/num_streams;
      num_stream_float_bytes = num_float_bytes/num_streams;
      
      log10num_con_sqrs = (float)std::log10( num_samples );

      dout << __func__ << "(): FFT_SIZE is " << FFT_SIZE << "\n"; 
      dout << __func__ << "(): NUM_FFT_SIZE_BITS is " << NUM_FFT_SIZE_BITS << "\n"; 
      
      dout << __func__ << "(): num_samples is " << num_samples << "\n"; 
      dout << __func__ << "(): threads_per_block = " << threads_per_block << "\n";
      dout << __func__ << "(): num_blocks = " << num_blocks << "\n\n";

      dout << __func__ << "(): num_streams is " << num_streams << "\n"; 
      dout << __func__ << "(): num_stream_samples is " << num_stream_samples << "\n"; 
      dout << __func__ << "(): num_stream_blocks = " << num_stream_blocks << "\n\n";
      
      dout << __func__ << "(): log10num_con_sqrs is " << log10num_con_sqrs << "\n\n";
      
      dout << __func__ << "(): num_bytes is " << num_bytes << "\n"; 
      dout << __func__ << "(): num_float_bytes is " << num_float_bytes << "\n"; 
      dout << __func__ << "(): num_stream_bytes is " << num_stream_bytes << "\n"; 
      dout << __func__ << "(): num_stream_float_bytes is " << num_stream_float_bytes << "\n\n"; 

      int device = 0;
      try_cuda_func_throw( cerror, cudaGetDevice(&device) );
      
      try_cuda_func_throw( cerror, cudaDeviceReset() );
      
      // CUDA will yield its thread when waiting for results from the GPU
      // Increases performance of CPU threads working in parallel with the GPU
      // but also increases latency when waiting for the GPU
      try_cuda_func_throw( cerror, cudaSetDeviceFlags( cudaDeviceScheduleYield ) );
    
      for( int stream_num = 0; stream_num < num_streams; ++stream_num ) {
         try_cuda_func_throw( cerror, cudaStreamCreate( &streams[stream_num] ) );
      } 

      try_cuda_func_throw( cerror, cudaHostAlloc( (void**)&samples, num_bytes, cudaHostAllocDefault ) );
      try_cuda_func_throw( cerror, cudaHostAlloc( (void**)&frequencies, num_bytes, cudaHostAllocDefault ) );
      /*try_cuda_func_throw( cerror, cudaHostAlloc( (void**)&con_sqrs, num_bytes, cudaHostAllocDefault ) );*/
      /*try_cuda_func_throw( cerror, cudaHostAlloc( (void**)&psds, num_bytes, cudaHostAllocDefault ) );*/

      try_cuda_func_throw( cerror, cudaMalloc( (void**)&d_samples, num_bytes ) );
      try_cuda_func_throw( cerror, cudaMalloc( (void**)&d_frequencies, num_bytes ) );
      /*try_cuda_func_throw( cerror, cudaMalloc( (void**)&d_con_sqrs, num_bytes ) );*/
      /*try_cuda_func_throw( cerror, cudaMalloc( (void**)&d_psds, num_float_bytes ) );*/

      /*try_cuda_func_throw( cerror, cudaHostGetDevicePointer( (void**)&d_samples, (void*)samples, 0 ) );*/
      /*try_cuda_func_throw( cerror, cudaHostGetDevicePointer( (void**)&d_frequencies, (void*)frequencies, 0 ) );*/
      /*try_cuda_func_throw( cerror, cudaHostGetDevicePointer( (void**)&d_con_sqrs, (void*)con_sqrs, 0 ) );*/
      /*try_cuda_func_throw( cerror, cudaHostGetDevicePointer( (void**)&d_psds, (void*)psds, 0 ) );*/

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
      } 
      /*for( int index = 0; index < num_samples; ++index ) {*/
         /*con_sqrs[index].x = 0;*/
         /*con_sqrs[index].y = 0;*/
      /*} */
      /*for( int index = 0; index < num_samples; ++index ) {*/
         /*psds[index] = 0;*/
      /*} */

      //try_cuda_func_throw( cerror, cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte ) );

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
      Duration_ms duration_ms;
      float gpu_milliseconds = 0;
      
      // Typedef for Time_Point is in my_utils.hpp
      Time_Point start = Steady_Clock::now();

      for( int stream_num = 0; stream_num < num_streams; ++stream_num ) {
      
         int stream_index = num_stream_samples * stream_num;
         try_cuda_func_throw( cerror, cudaMemcpyAsync( &d_samples[stream_index], &samples[stream_index], num_stream_bytes, cudaMemcpyHostToDevice, streams[stream_num] ) );

         // Launch the kernel
         /*simple_dsp_kernel<<<num_stream_blocks, threads_per_block, num_shared_bytes, streams[stream_num]>>>( */
            /*&d_psds[stream_index], &d_con_sqrs[stream_index], &d_frequencies[stream_index], &d_samples[stream_index], num_stream_samples, log10num_con_sqrs );*/
         cookbook_fft64<<<num_stream_blocks, threads_per_block, num_shared_bytes, streams[stream_num]>>>( 
            &d_frequencies[stream_index], &d_samples[stream_index], num_stream_samples );
         
         try_cuda_func_throw( cerror, cudaMemcpyAsync( &frequencies[stream_index], &d_frequencies[stream_index], num_stream_bytes, cudaMemcpyDeviceToHost, streams[stream_num] ) );
      }

      try_cuda_func_throw( cerror, cudaStreamSynchronize( streams[num_streams-1] ) );
      //dout << __func__ << "(): Done with simple_dsp_kernel with " << NUM_CHUNKS << "\n"; 

      duration_ms = Steady_Clock::now() - start;
      gpu_milliseconds = duration_ms.count();
      std::cout << "GPU processing took " << gpu_milliseconds << " milliseconds\n";
      
      if ( debug ) {
         const char delim[] = " ";
         const char suffix[] = "\n";
         print_cufftComplexes(frequencies, num_samples, "Frequencies from GPU: ", delim, suffix);
      }
      
      // float max_diff = 1e-3;
      // const cufftComplex expected_frequencies[] = {
         // {-132., -240.},          {-85.94402282, -36.09659629},
           // {-81.35411222, -0.91108112},   {84.94564024, -187.26012091},
             // {6.83723648, -27.00435032},  {-24.07475218, -150.84853419},
           // {-50.59636575, +29.90044203},   {-4.80805541, -73.01125682},
           // {-48.76955262, -69.85786438},  {-51.8021418, -59.20386129},
           // {-23.61138123, -64.37262007},  {104.73821491, -43.24309453},
           // {-16.74151197, -175.16605276},  {-27.61689685, +76.52360648},
          // {-125.05633239, +34.02646551},   {80.79678323, -48.6777785},
           // {-72., -84.},          {-45.52931596, +54.15275963},
           // {125.79232289, +56.25780044},   {57.22641, -15.69231809},
           // {-83.99643814, -80.9538002},   {-18.54727802, -155.81469514},
            // {13.03395819, -97.52749052},   {27.10334384, +5.76524363},
           // {-53.45584412, +17.74011537},  {-38.93389753, -6.62313144},
           // {-43.226213, -105.64023583},   {18.01908495, +45.23527691},
            // {-2.87496529, -25.8473992},    {55.45470935, -16.56101907},
           // {123.45218242, -103.60809949},  {-11.39947084, -14.98496058},
            // {44., +40.},          {-10.92177239, +15.03216534},
           // {-23.93041029, +3.57839443},  {115.6990709, -37.44050401},
            // {25.73130601, -43.05452418},   {54.0160334, -1.24250993},
            // {29.06745814, -36.73234781},   {36.88721264, -0.69300137},
            // {24.76955262, -98.14213562},  {-44.02764984, +42.79225072},
            // {89.15365045, -14.57386059},  {118.94062029, -10.8452387},
           // {108.29071849, -41.16699172},   {67.17722805, +66.17455353},
           // {129.83575362, -19.3857114},    {25.53325289, +72.75019775},
            // {48., +12.},          {141.75599176, +5.25427001},
            // {59.49219962, +24.95713725},   {64.05621207, +49.31729165},
             // {3.42789564, -56.9873253},   {127.79222818, -14.7950111},
           // {100.14170239, +13.37873228},   {39.96257108, +40.42186716},
            // {-2.54415588, -89.74011537},   {76.22705162, -26.95460967},
            // {57.68394378, +132.70446549},  {146.53116761, +75.81095868},
           // {343.32575878, +194.18044368},  {656.97448503, +592.2103624},
          // {-443.87835664, -364.05199061},  {-72.23205842, -241.45256226}
      // };
//
      //if ( debug ) std::cout << __func__ << "(): Comparing first " << FFT_SIZE << " (FFT Size) results with expected\n";

      //bool all_are_close = cufftComplexes_are_close( frequencies, expected_frequencies, FFT_SIZE, max_diff, debug );
   
   } catch (std::exception& ex) {
      throw std::runtime_error{
         std::string{"SampleProcFunctor::" + std::string{__func__} + "(): " + ex.what()}};
   }
      

}


SimpleDSP::~SimpleDSP() {
   if (samples) cudaFreeHost(samples);

   if (frequencies) cudaFreeHost(frequencies);

   if (con_sqrs) cudaFreeHost(con_sqrs);

   if (psds) cudaFreeHost(psds);
   
}
