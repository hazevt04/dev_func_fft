#pragma once

#include "my_utils.hpp"
#include "my_file_io_funcs.hpp"

#include "my_cuda_utils.hpp"
#include "my_cufft_utils.hpp"

// Tables for checking output
#include "expected_sfrequencies.hpp"
#include "expected_con_sqrs.hpp"
#include "expected_psds.hpp"

class SimpleDSP {
   public:
      SimpleDSP():
         samples( nullptr ),
         frequencies( nullptr ),
         sfrequencies( nullptr ),
         con_sqrs( nullptr ),
         psds( nullptr ),
         d_samples( nullptr ),
         d_frequencies( nullptr ),
         d_sfrequencies( nullptr ),
         d_con_sqrs( nullptr ),
         d_psds( nullptr ),
         num_samples(0),
         threads_per_block(0),
         num_blocks(0),
         num_bytes(0),
         num_float_bytes(0),
         debug(false) {}

      SimpleDSP( const int new_num_samples, const bool new_debug );
      
      void run();
      void operator()();
      
      ~SimpleDSP();

   private:
      cufftComplex* samples;
      cufftComplex* frequencies;
      cufftComplex* sfrequencies;
      float* con_sqrs;
      float* psds;
      
      cufftComplex* d_samples;
      cufftComplex* d_frequencies;

      cufftComplex* d_sfrequencies;
      float* d_con_sqrs;
      float* d_psds;

      int num_samples;
      int num_ffts;
      int threads_per_block;
      int num_blocks;

      size_t num_bytes;
      size_t num_float_bytes;
      bool debug;

};
