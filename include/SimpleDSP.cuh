#pragma once

#include <cufft.h>

constexpr float max_diff = 1e-3;

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
      
      cufftComplex* d_samples;
      cufftComplex* d_frequencies;
      cufftComplex* d_sfrequencies;
      cufftComplex* d_con_sqrs;
      float* d_psds;

      int num_samples;
      int num_ffts;
      int threads_per_block;
      int num_blocks;
      float log10num_con_sqrs;
      bool debug;

};
