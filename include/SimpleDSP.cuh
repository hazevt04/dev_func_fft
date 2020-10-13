#pragma once
#include <cstring>

#include "my_cuda_utils.hpp"
#include "my_cufft_utils.hpp"

#include "my_file_io_funcs.hpp"

#include "simple_dsp_kernels.cuh"

constexpr float max_diff = 1e-3;

class SimpleDSP {
   public:
      SimpleDSP():
         samples( nullptr ),
         frequencies( nullptr ),
         sfrequencies( nullptr ),
         con_sqrs( nullptr ),
         psds( nullptr ),
         num_samples(0),
         threads_per_block(0),
         num_blocks(0),
         num_shared_bytes(0),
         log10num_con_sqrs(0),
         debug(false) {}

      SimpleDSP( const int new_num_samples, const bool new_debug );
      
      void run();
      void operator()();
      
      ~SimpleDSP();

   private:
      cufftComplex* samples;
      cufftComplex* frequencies;
      cufftComplex* sfrequencies;
      cufftComplex* con_sqrs;
      float* psds;

      int num_samples;
      int threads_per_block;
      int num_blocks;
      size_t num_shared_bytes;
      float log10num_con_sqrs;
      bool debug;
};
