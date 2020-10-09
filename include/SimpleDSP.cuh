#pragma once
#include <cstdlib>
#include <cstring>

#include "my_cuda_utils.hpp"
#include "my_cufft_utils.hpp"

#include "my_file_io_funcs.hpp"

#include "simple_dsp_kernels.cuh"


constexpr int default_num_streams = 4;
constexpr int default_num_stream_blocks = 4;
constexpr int max_num_streams = 8;

class SimpleDSP {
   public:
      SimpleDSP():
         samples( nullptr ),
         frequencies( nullptr ),
         con_sqrs( nullptr ),
         psds( nullptr ),
         d_samples( nullptr ),
         d_frequencies( nullptr ),
         d_con_sqrs( nullptr ),
         d_psds( nullptr ),
         num_samples(0),
         num_streams(0),
         num_stream_samples(0),
         threads_per_block(0),
         num_blocks(0),
         num_stream_blocks(0),
         log10num_con_sqrs(0),
         num_bytes(0),
         num_float_bytes(0),
         num_stream_bytes(0),
         num_stream_float_bytes(0),
         debug(false) {}

      SimpleDSP( const int new_num_samples, const bool new_debug );
      
      void run();
      void operator()();
      
      ~SimpleDSP();

   private:
      cufftComplex* samples;
      cufftComplex* frequencies;
      cufftComplex* con_sqrs;
      float* psds;
      
      cufftComplex* d_samples;
      cufftComplex* d_frequencies;
      cufftComplex* d_con_sqrs;
      float* d_psds;

      cudaStream_t streams[max_num_streams];
      cudaEvent_t kernel_done;


      int num_samples;
      int num_streams;
      int num_stream_samples;
      int threads_per_block;
      int num_blocks;
      int num_stream_blocks;
      size_t num_bytes;
      size_t num_float_bytes;
      size_t num_stream_bytes;
      size_t num_stream_float_bytes;
      size_t num_shared_bytes;
      float log10num_con_sqrs;
      bool debug;
      
};
